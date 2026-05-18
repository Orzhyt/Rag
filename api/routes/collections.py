from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.deps import get_embedder, get_milvus_client, get_milvus_service
from api.schemas import (
    BM25SupportResponse,
    CollectionExistsResponse,
    CreateCollectionRequest,
    DescribeCollectionResponse,
    DropCollectionRequest,
    InitCollectionRequest,
    MessageResponse,
    TruncateCollectionsRequest,
    VectorIndexParams,
)
from milvus.client import MilvusClient
from milvus.embedder import EmbeddingModel
from retrieval.service import MilvusService

router = APIRouter()


def _switch_db(client: MilvusClient, database: Optional[str]):
    if database and database != client.database:
        client.using_database(database)


@router.get("", response_model=list[str])
def list_collections(
    database: Optional[str] = Query(None),
    client: MilvusClient = Depends(get_milvus_client),
):
    _switch_db(client, database)
    return client.list_collections()


@router.get("/{collection_name}/schema", response_model=DescribeCollectionResponse)
def get_collection_schema(
    collection_name: str,
    database: Optional[str] = Query(None),
    client: MilvusClient = Depends(get_milvus_client),
):
    _switch_db(client, database)
    info = client.describe_collection(collection_name)
    return DescribeCollectionResponse(
        name=info.get("name", collection_name),
        description=info.get("description", ""),
        fields=info.get("fields", []),
    )


@router.get("/exists", response_model=CollectionExistsResponse)
def collection_exists(
    collection_name: Optional[str] = Query(None),
    database: Optional[str] = Query(None),
    client: MilvusClient = Depends(get_milvus_client),
):
    _switch_db(client, database)
    name = collection_name or client.collection_name
    exists = client.has_collection(name)
    return CollectionExistsResponse(exists=exists)


@router.get("/describe", response_model=DescribeCollectionResponse)
def describe_collection(
    collection_name: Optional[str] = Query(None),
    database: Optional[str] = Query(None),
    client: MilvusClient = Depends(get_milvus_client),
):
    _switch_db(client, database)
    name = collection_name or client.collection_name
    info = client.describe_collection(name)
    return DescribeCollectionResponse(
        name=info.get("name", name),
        description=info.get("description", ""),
        fields=info.get("fields", []),
    )


@router.get("/bm25-support", response_model=BM25SupportResponse)
def bm25_support(
    collection_name: Optional[str] = Query(None),
    database: Optional[str] = Query(None),
    client: MilvusClient = Depends(get_milvus_client),
):
    _switch_db(client, database)
    col_name = collection_name or client.collection_name
    if not client.has_collection(col_name):
        return BM25SupportResponse(has_bm25_support=False)
    info = client._client.describe_collection(col_name)
    has_support = any(
        f.get("type").name == "SPARSE_FLOAT_VECTOR"
        for f in info.get("fields", [])
        if hasattr(f.get("type"), "name")
    )
    return BM25SupportResponse(has_bm25_support=has_support)


@router.post("/create", response_model=MessageResponse)
def create_collection(
    req: CreateCollectionRequest,
    client: MilvusClient = Depends(get_milvus_client),
    embedder: EmbeddingModel = Depends(get_embedder),
):
    _switch_db(client, req.database)

    dim = req.dim if req.dim is not None else embedder.dim

    vector_index = None
    if req.vector_index is not None:
        vector_index = [
            {"field_name": vi.field_name, "index_type": vi.index_type,
             "metric_type": vi.metric_type, "params": vi.params}
            for vi in req.vector_index
        ]

    client.create_collection(
        dim=dim,
        drop_if_exists=req.drop_if_exists,
        collection_name=req.collection_name,
        fields=req.fields,
        vector_index=vector_index,
        description=req.description,
    )

    effective_name = req.collection_name or client.collection_name
    return MessageResponse(message=f"Collection '{effective_name}' created (dim={dim})")


@router.post("/init", response_model=MessageResponse)
def init_collection(
    req: InitCollectionRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    _switch_db(service.client, req.database)

    vector_index = None
    if req.vector_index is not None:
        vector_index = [
            {
                "field_name": vi.field_name,
                "index_type": vi.index_type,
                "metric_type": vi.metric_type,
                "params": vi.params,
            }
            for vi in req.vector_index
        ]

    service.init_collection(
        drop_if_exists=req.drop_if_exists,
        collection_name=req.collection_name,
        fields=req.fields,
        vector_index=vector_index,
        description=req.description,
    )

    effective_name = req.collection_name or service.client.collection_name
    return MessageResponse(message=f"Collection '{effective_name}' initialized")


@router.delete("/drop", response_model=MessageResponse)
def drop_collection(
    req: DropCollectionRequest,
    client: MilvusClient = Depends(get_milvus_client),
):
    effective_name = req.collection_name or client.collection_name
    client.drop_collection(collection_name=req.collection_name, database=req.database)
    return MessageResponse(message=f"Collection '{effective_name}' dropped")


@router.post("/truncate", response_model=MessageResponse)
def truncate_collections(
    req: TruncateCollectionsRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    _switch_db(service.client, req.database)
    total_deleted = 0
    for col_name in req.collection_names:
        total_deleted += service.truncate_collection(collection_name=col_name)
    return MessageResponse(
        message=f"Truncated {len(req.collection_names)} collections, total {total_deleted} rows deleted"
    )
