import os

# 避免 torchvision C 扩展 DLL 加载崩溃（Windows 下 torch/torchvision 版本不匹配时）
os.environ.setdefault("TORCHVISION_DISABLE_EXTENSION", "1")

import pytest
from dotenv import load_dotenv
from milvus import MilvusClient, EmbeddingModel, MilvusService

load_dotenv()
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "")
TEST_COLLECTION = "test_rag_chunks"


@pytest.fixture(scope="session")
def embedder():
    """Session 级别的 embedding 模型，只加载一次"""
    return EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)


@pytest.fixture
def milvus_client():
    """每个测试独立的 MilvusClient 连接"""
    client = MilvusClient(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        user=MILVUS_USER,
        password=MILVUS_PASSWORD,
        collection_name=TEST_COLLECTION,
    )
    client.connect()
    yield client
    # 清理：删除测试集合（如果存在）
    if client.has_collection(TEST_COLLECTION):
        client.drop_collection()
    client.disconnect()


@pytest.fixture
def milvus_service(milvus_client, embedder):
    """组合 client + embedder 的 MilvusService，自动初始化和清理"""
    service = MilvusService(client=milvus_client, embedder=embedder)
    service.init_collection(drop_if_exists=True, enable_bm25=True)
    yield service
    # 清理：释放集合
    try:
        milvus_client.release_collection()
    except Exception:
        pass
