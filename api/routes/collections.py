from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.deps import get_embedder, get_milvus_client, get_milvus_service
from api.schemas import (
    BM25SupportResponse,
    CollectionExistsResponse,
    CreateCollectionRequest,
    DescribeCollectionResponse,
    DropCollectionRequest,
    FieldDefinition,
    InitCollectionRequest,
    MessageResponse,
    TruncateCollectionsRequest,
    VectorIndexParams,
)
from milvus.client import MilvusClient, build_field_schema
from milvus.embedder import EmbeddingModel
from retrieval.service import MilvusService

router = APIRouter()


@router.get("/exists", response_model=CollectionExistsResponse)
def collection_exists(
    collection_name: Optional[str] = Query(None),
    database: Optional[str] = Query(None),
    client: MilvusClient = Depends(get_milvus_client),
):
    original_db = None
    if database:
        original_db = client.database
        client.using_database(database)
    try:
        name = collection_name or client.collection_name
        exists = client.has_collection(name)
    finally:
        if original_db is not None:
            client.using_database(original_db)
    return CollectionExistsResponse(exists=exists)


@router.get("/describe", response_model=DescribeCollectionResponse)
def describe_collection(
    collection_name: Optional[str] = Query(None),
    database: Optional[str] = Query(None),
    client: MilvusClient = Depends(get_milvus_client),
):
    original_db = None
    if database:
        original_db = client.database
        client.using_database(database)
    try:
        name = collection_name or client.collection_name
        info = client.describe_collection(name)
    finally:
        if original_db is not None:
            client.using_database(original_db)
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
    original_db = None
    if database:
        original_db = client.database
        client.using_database(database)
    try:
        col_name = collection_name or client.collection_name
        if not client.has_collection(col_name):
            return BM25SupportResponse(has_bm25_support=False)
        info = client._client.describe_collection(col_name)
        has_support = any(
            f.get("type").name == "SPARSE_FLOAT_VECTOR"
            for f in info.get("fields", [])
            if hasattr(f.get("type"), "name")
        )
    finally:
        if original_db is not None:
            client.using_database(original_db)
    return BM25SupportResponse(has_bm25_support=has_support)


@router.post("/create", response_model=MessageResponse)
def create_collection(
    req: CreateCollectionRequest,
    client: MilvusClient = Depends(get_milvus_client),
    embedder: EmbeddingModel = Depends(get_embedder),
):
    # 切换到指定数据库
    original_db = None
    if req.database:
        original_db = client.database
        client.using_database(req.database)

    try:
        dim = req.dim if req.dim is not None else embedder.dim

        # 构建自定义字段列表
        field_schemas = None
        if req.fields is not None:
            field_schemas = [build_field_schema(fd, dim=dim) for fd in req.fields]

        # 构建向量索引
        vector_index = None
        if req.vector_index is not None:
            vector_index = [
                {"field_name": vi.field_name, "index_type": vi.index_type,
                 "metric_type": vi.metric_type, "params": vi.params}
                for vi in req.vector_index
            ]

        # BM25 配置
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
    finally:
        # 恢复原始数据库
        if original_db is not None:
            client.using_database(original_db)

    return MessageResponse(message=f"Collection '{effective_name}' created (dim={dim})")


@router.post("/init", response_model=MessageResponse)
def init_collection(
    req: InitCollectionRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    # 切换到指定数据库
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)

    try:
        field_schemas = None
        if req.fields is not None:
            field_schemas = [build_field_schema(fd) for fd in req.fields]

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
    finally:
        # 恢复原始数据库
        if original_db is not None:
            service.client.using_database(original_db)

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
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)
    try:
        total_deleted = 0
        for col_name in req.collection_names:
            total_deleted += service.truncate_collection(collection_name=col_name)
    finally:
        if original_db is not None:
            service.client.using_database(original_db)
    return MessageResponse(
        message=f"Truncated {len(req.collection_names)} collections, total {total_deleted} rows deleted"
    )
