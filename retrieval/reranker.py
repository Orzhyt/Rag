import json
import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

from common.logger import setup_logger

load_dotenv()
logger = setup_logger("retrieval.reranker")


@dataclass
class RerankResult:
    index: int
    relevance_score: float


class Reranker:
    """华为云 MaaS rerank API 客户端"""

    def __init__(self, api_url: str = None, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("MAAS_API_KEY", "")
        self.model = model or os.getenv("MAAS_RERANK_MODEL", "bge-reranker-v2-m3")

        maas_url = api_url or os.getenv("MAAS_API_URL", "https://api.modelarts-maas.com/v2/chat/completions")
        from urllib.parse import urlparse
        parsed = urlparse(maas_url)
        self.rerank_url = f"{parsed.scheme}://{parsed.netloc}/v1/rerank"

        logger.info("Reranker initialized: model=%s, url=%s", self.model, self.rerank_url)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """调用 MaaS rerank API，返回按 relevance_score 降序排列的结果"""
        if not documents:
            return []

        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            data["top_n"] = top_n

        try:
            resp = requests.post(
                self.rerank_url,
                headers=headers,
                json=data,
                verify=False,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            logger.error("MaaS rerank API call failed: %s", e)
            raise

        # 响应格式: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
        raw_results = result.get("results", [])
        rerank_results = [
            RerankResult(index=r["index"], relevance_score=r["relevance_score"])
            for r in raw_results
        ]
        rerank_results.sort(key=lambda x: x.relevance_score, reverse=True)

        if top_n is not None:
            rerank_results = rerank_results[:top_n]

        return rerank_results
