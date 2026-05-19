import os
from typing import Optional

from common.logger import setup_logger
from milvus.client import MilvusClient
from retrieval.embedder import Embedder
from retrieval.profile import DEFAULT_RAG_PROFILE
from retrieval.reranker import Reranker
from retrieval.service import MilvusService
from llm.client import LLMClient
from llm.service import RAGChatService

logger = setup_logger("api.deps")

_milvus_client: Optional[MilvusClient] = None
_milvus_service: Optional[MilvusService] = None
_embedder: Optional[Embedder] = None
_reranker: Optional[Reranker] = None
_llm_client: Optional[LLMClient] = None
_rag_chat_service: Optional[RAGChatService] = None


def init_singletons():
    global _milvus_client, _milvus_service, _embedder, _reranker, _llm_client, _rag_chat_service

    logger.info("Loading embedding model...")
    _embedder = Embedder()
    _ = _embedder.dim  # 触发模型加载（maas 模式下直接返回配置维度）
    logger.info("Embedding model loaded, dim=%d, provider=%s", _embedder.dim, _embedder.provider)

    # 初始化神经重排序
    if os.getenv("MAAS_RERANK_ENABLED", "false").lower() in ("true", "1", "yes"):
        _reranker = Reranker()
        logger.info("Neural reranker initialized (model=%s)", _reranker.model)
    else:
        logger.info("Neural reranker disabled")

    logger.info("Connecting to Milvus...")
    _milvus_client = MilvusClient()
    _milvus_client.connect()
    logger.info("Milvus connected")

    _milvus_service = MilvusService(
        client=_milvus_client, embedder=_embedder, profile=DEFAULT_RAG_PROFILE,
        reranker=_reranker,
    )

    logger.info("Initializing LLM client...")
    _llm_client = LLMClient()
    _rag_chat_service = RAGChatService(llm_client=_llm_client, milvus_service=_milvus_service)
    logger.info("RAG chat service initialized")


def cleanup_singletons():
    global _milvus_client
    if _milvus_client and _milvus_client.connected:
        _milvus_client.disconnect()
        logger.info("Milvus disconnected")


def get_milvus_client() -> MilvusClient:
    if _milvus_client is None:
        raise RuntimeError("MilvusClient not initialized")
    return _milvus_client


def get_milvus_service() -> MilvusService:
    if _milvus_service is None:
        raise RuntimeError("MilvusService not initialized")
    return _milvus_service


def get_embedder() -> Embedder:
    if _embedder is None:
        raise RuntimeError("EmbeddingModel not initialized")
    return _embedder


def get_reranker() -> Optional[Reranker]:
    return _reranker


def get_llm_service() -> RAGChatService:
    if _rag_chat_service is None:
        raise RuntimeError("RAGChatService not initialized")
    return _rag_chat_service
