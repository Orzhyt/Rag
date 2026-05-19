import os

from common.logger import setup_logger
from milvus.client import MilvusClient
from retrieval.embedder import Embedder
from retrieval.profile import DEFAULT_RAG_PROFILE
from retrieval.reranker import Reranker
from retrieval.service import MilvusService

logger = setup_logger("evaluation.runner")


def init_services() -> tuple:
    """初始化评估所需的服务，返回 (MilvusService, EmbeddingModel)"""
    embedder = Embedder()
    _ = embedder.dim  # 触发模型加载

    client = MilvusClient()
    client.connect()

    reranker = None
    if os.getenv("MAAS_RERANK_ENABLED", "false").lower() in ("true", "1", "yes"):
        reranker = Reranker()

    service = MilvusService(
        client=client, embedder=embedder, profile=DEFAULT_RAG_PROFILE,
        reranker=reranker,
    )
    return service, embedder
