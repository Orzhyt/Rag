from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.deps import get_embedder, get_milvus_client, get_milvus_service
from api.schemas import (
    BM25SupportResponse,
    CollectionExistsResponse,
    CreateCollectionRequest,
    DescribeCollectionResponse,
    InitCollectionRequest,
    MessageResponse,
)
from milvus.client import MilvusClient, build_field_schema
from milvus.embedder import EmbeddingModel
from retrieval.service import MilvusService

router = APIRouter()


@router.get("/exists", response_model=CollectionExistsResponse)
def collection_exists(
    collection_name: Optional[str] = Query(None),
    client: MilvusClient = Depends(get_milvus_client),
):
    name = collection_name or client.collection_name
    exists = client.has_collection(name)
    return CollectionExistsResponse(exists=exists)


@router.get("/describe", response_model=DescribeCollectionResponse)
def describe_collection(
    collection_name: Optional[str] = Query(None),
    client: MilvusClient = Depends(get_milvus_client),
):
    name = collection_name or client.collection_name
    info = client.describe_collection(name)
    return DescribeCollectionResponse(
        name=info.get("name", name),
        description=info.get("description", ""),
        fields=info.get("fields", []),
    )


@router.get("/bm25-support", response_model=BM25SupportResponse)
def bm25_support(client: MilvusClient = Depends(get_milvus_client)):
    return BM25SupportResponse(has_bm25_support=client.has_bm25_support)


@router.post("/create", response_model=MessageResponse)
def create_collection(
    req: CreateCollectionRequest,
    client: MilvusClient = Depends(get_milvus_client),
    embedder: EmbeddingModel = Depends(get_embedder),
):
    dim = req.dim if req.dim is not None else embedder.dim

    field_schemas = None
    if req.fields is not None:
        field_schemas = [build_field_schema(fd, dim=dim) for fd in req.fields]

    vector_index = None
    if req.vector_index is not None:
        vector_index = {
            "field_name": req.vector_index.field_name,
            "index_type": req.vector_index.index_type,
            "metric_type": req.vector_index.metric_type,
            "params": req.vector_index.params,
        }

    bm25_config = None
    if req.bm25_config is not None:
        bm25_config = {
            "text_field_name": req.bm25_config.text_field_name,
            "sparse_field_name": req.bm25_config.sparse_field_name,
            "function_name": req.bm25_config.function_name,
        }

    client.create_collection(
        dim=dim,
        drop_if_exists=req.drop_if_exists,
        enable_bm25=req.enable_bm25,
        collection_name=req.collection_name,
        fields=field_schemas,
        vector_index=vector_index,
        bm25_config=bm25_config,
        description=req.description,
        embedding_field_name=req.embedding_field_name,
    )

    effective_name = req.collection_name or client.collection_name
    return MessageResponse(message=f"Collection '{effective_name}' created (dim={dim})")


@router.post("/init", response_model=MessageResponse)
def init_collection(
    req: InitCollectionRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    field_schemas = None
    if req.fields is not None:
        field_schemas = [build_field_schema(fd) for fd in req.fields]

    vector_index = None
    if req.vector_index is not None:
        vector_index = {
            "field_name": req.vector_index.field_name,
            "index_type": req.vector_index.index_type,
            "metric_type": req.vector_index.metric_type,
            "params": req.vector_index.params,
        }

    bm25_config = None
    if req.bm25_config is not None:
        bm25_config = {
            "text_field_name": req.bm25_config.text_field_name,
            "sparse_field_name": req.bm25_config.sparse_field_name,
            "function_name": req.bm25_config.function_name,
        }

    service.init_collection(
        drop_if_exists=req.drop_if_exists,
        enable_bm25=req.enable_bm25,
        collection_name=req.collection_name,
        fields=field_schemas,
        vector_index=vector_index,
        bm25_config=bm25_config,
        description=req.description,
        embedding_field_name=req.embedding_field_name,
    )

    effective_name = req.collection_name or service.client.collection_name
    return MessageResponse(message=f"Collection '{effective_name}' initialized")


@router.delete("/drop", response_model=MessageResponse)
def drop_collection(client: MilvusClient = Depends(get_milvus_client)):
    client.drop_collection()
    return MessageResponse(message=f"Collection '{client.collection_name}' dropped")


@router.post("/load", response_model=MessageResponse)
def load_collection(client: MilvusClient = Depends(get_milvus_client)):
    client.load_collection()
    return MessageResponse(message=f"Collection '{client.collection_name}' loaded")


@router.post("/release", response_model=MessageResponse)
def release_collection(client: MilvusClient = Depends(get_milvus_client)):
    client.release_collection()
    return MessageResponse(message=f"Collection '{client.collection_name}' released")
