from fastapi import APIRouter, Depends

from api.deps import get_embedder, get_milvus_client
from api.schemas import HealthResponse
from milvus.client import MilvusClient
from milvus.embedder import EmbeddingModel

router = APIRouter()


@router.get("", response_model=HealthResponse)
def health_check(
    client: MilvusClient = Depends(get_milvus_client),
    embedder: EmbeddingModel = Depends(get_embedder),
):
    milvus_ok = client.connected
    model_loaded = embedder._model is not None
    status = "ok" if (milvus_ok and model_loaded) else "degraded"
    return HealthResponse(
        status=status,
        milvus_connected=milvus_ok,
        embedding_model_loaded=model_loaded,
        embedding_device=embedder.device,
    )
