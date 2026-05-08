import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from common.logger import setup_logger

load_dotenv()
logger = setup_logger("milvus.embedder")


class EmbeddingModel:
    """文本向量化模型，基于 sentence-transformers

    模型加载优先级:
    1. 若模型路径为本地目录且存在，直接加载本地模型
    2. 否则通过 HuggingFace 下载（可设置 HF_ENDPOINT 环境变量使用镜像，
       如 https://hf-mirror.com）
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")
        self._model = None
        self._dim = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            local_path = Path(self.model_name)
            if local_path.exists() and local_path.is_dir():
                logger.info("Loading embedding model from local: %s", self.model_name)
            else:
                hf_endpoint = os.getenv("HF_ENDPOINT", "")
                if hf_endpoint:
                    logger.info("Using HF mirror: %s", hf_endpoint)
                logger.info("Loading embedding model: %s on %s", self.model_name, self.device)

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding model loaded, dimension: %d", self._dim)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            _ = self.model  # trigger lazy load
        return self._dim

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量将文本转为向量，返回 List[List[float]]"""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
