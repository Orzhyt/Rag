import math
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from common.logger import setup_logger

load_dotenv()
logger = setup_logger("retrieval.embedder")


def _detect_device() -> str:
    """Auto-detect best available compute device: CUDA > NPU > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, RuntimeError):
        pass

    try:
        import torch_npu  # noqa: F401 -- registers torch.npu
        import torch
        if torch.npu.is_available():
            return "npu"
    except (ImportError, RuntimeError, AttributeError):
        pass

    return "cpu"


def _normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        return vec
    return [v / norm for v in vec]


class Embedder:
    """文本向量化模型，支持 local (sentence-transformers) 和 maas (华为云 MaaS API) 两种后端

    模型加载优先级 (provider=local):
    1. 若模型路径为本地目录且存在，直接加载本地模型
    2. 否则通过 HuggingFace 下载（可设置 HF_ENDPOINT 环境变量使用镜像，
       如 https://hf-mirror.com）

    provider=maas 时，调用华为云 MaaS embedding API，无需本地模型文件。
    """

    def __init__(self, model_name: str = None, device: str = None, provider: str = None):
        self.provider = provider or os.getenv("EMBEDDING_PROVIDER", "local")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "models/Qwen3-Embedding-0.6B")

        if self.provider == "maas":
            self._maas_model = os.getenv("MAAS_EMBEDDING_MODEL", "bge-m3")
            self._maas_dim = int(os.getenv("MAAS_EMBEDDING_DIM", "1024"))
            self._maas_api_url = self._resolve_maas_embedding_url()
            self._maas_api_key = os.getenv("MAAS_API_KEY", "")
            self.device = "cloud"
            self._model = None
            self._dim = self._maas_dim
            logger.info(
                "Embedding provider=maas, model=%s, dim=%d, url=%s",
                self._maas_model, self._maas_dim, self._maas_api_url,
            )
            return

        device = device or os.getenv("EMBEDDING_DEVICE", "auto")
        if device == "auto":
            device = _detect_device()
            logger.info("Auto-detected embedding device: %s", device)
        self.device = device
        self._model = None
        self._dim = None

    @staticmethod
    def _resolve_maas_embedding_url() -> str:
        """从 MAAS_API_URL 推导 embedding API 地址

        MAAS_API_URL 示例: https://api.modelarts-maas.com/v2/chat/completions
        去掉 /v2/chat/completions 后拼接 /v1/embeddings
        """
        api_url = os.getenv("MAAS_API_URL", "https://api.modelarts-maas.com/v2/chat/completions")
        # 去掉路径部分，只保留 scheme + host
        from urllib.parse import urlparse
        parsed = urlparse(api_url)
        return f"{parsed.scheme}://{parsed.netloc}/v1/embeddings"

    @property
    def model(self):
        if self.provider == "maas":
            return None
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            local_path = Path(self.model_name)
            if local_path.exists() and local_path.is_dir():
                logger.info("Loading embedding model from local: %s", self.model_name)
            else:
                hf_endpoint = os.getenv("HF_ENDPOINT", "")
                if hf_endpoint:
                    logger.info("Using HF mirror: %s", hf_endpoint)
                logger.info("Loading embedding model: %s on %s", self.model_name, self.device)

            model_kwargs = {}
            if self.device in ("cuda", "npu"):
                model_kwargs["torch_dtype"] = torch.float16

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=model_kwargs if model_kwargs else None,
            )
            self._dim = self._model.get_embedding_dimension()
            dtype_str = "float16" if model_kwargs.get("torch_dtype") == torch.float16 else "float32"
            logger.info("Embedding model loaded, dimension: %d, device: %s, dtype: %s", self._dim, self.device, dtype_str)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            _ = self.model
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量将文本转为向量，返回 List[List[float]]"""
        if not texts:
            return []
        if self.provider == "maas":
            return self._encode_maas(texts, batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def _encode_maas(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """调用华为云 MaaS embedding API"""
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._maas_api_key}",
        }

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            data = {
                "model": self._maas_model,
                "input": batch,
                "encoding_format": "float",
            }
            try:
                resp = requests.post(
                    self._maas_api_url,
                    headers=headers,
                    json=data,
                    verify=False,
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                logger.error("MaaS embedding API call failed: %s", e)
                raise

            # 响应格式: {"data": [{"embedding": [...], "index": 0}, ...]}
            items = result.get("data", [])
            # 按 index 排序确保顺序一致
            items.sort(key=lambda x: x.get("index", 0))
            for item in items:
                vec = item.get("embedding", [])
                all_embeddings.append(_normalize(vec))

        return all_embeddings
