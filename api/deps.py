from typing import Optional

from common.logger import setup_logger
from milvus.client import MilvusClient
from milvus.embedder import EmbeddingModel
from retrieval.profile import DEFAULT_RAG_PROFILE
from retrieval.service import MilvusService

logger = setup_logger("api.deps")

_milvus_client: Optional[MilvusClient] = None
_milvus_service: Optional[MilvusService] = None
_embedder: Optional[EmbeddingModel] = None


def init_singletons():
    global _milvus_client, _milvus_service, _embedder

    logger.info("Loading embedding model...")
    _embedder = EmbeddingModel()
    _ = _embedder.dim  # 触发模型加载
    logger.info("Embedding model loaded, dim=%d", _embedder.dim)

    logger.info("Connecting to Milvus...")
    _milvus_client = MilvusClient()
    _milvus_client.connect()
    logger.info("Milvus connected")

    _milvus_service = MilvusService(
        client=_milvus_client, embedder=_embedder, profile=DEFAULT_RAG_PROFILE,
    )


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


def get_embedder() -> EmbeddingModel:
    if _embedder is None:
        raise RuntimeError("EmbeddingModel not initialized")
    return _embedder
