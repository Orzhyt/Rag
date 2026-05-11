from fastapi import APIRouter, Depends

from api.deps import get_milvus_service
from api.schemas import (
    CountRequest,
    CountResponse,
    DeleteByChunkIdsRequest,
    DeleteByFieldRequest,
    DeleteBySourceRequest,
    DeleteResponse,
    HybridSearchRequest,
    MessageResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    TruncateCollectionsRequest,
    TruncateDatabaseRequest,
)
from retrieval.service import MilvusService, SearchResult

router = APIRouter()


def _to_item(r: SearchResult) -> SearchResultItem:
    return SearchResultItem(score=r.score, fields=r.fields)


@router.post("/vector", response_model=SearchResponse)
def vector_search(
    req: SearchRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    # 切换到指定数据库
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)

    try:
        results = service.search(
            query=req.query,
            top_k=req.top_k,
            filter_expr=req.filter_expr,
            offset=req.offset,
            output_fields=req.output_fields,
            collection_names=req.collection_names,
            anns_field=req.anns_field,
        )
    finally:
        # 恢复原始数据库
        if original_db is not None:
            service.client.using_database(original_db)

    items = [_to_item(r) for r in results]
    return SearchResponse(results=items, total=len(items))


@router.post("/hybrid", response_model=SearchResponse)
def hybrid_search(
    req: HybridSearchRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    # 切换到指定数据库
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)

    try:
        anns_fields = None
        if req.anns_fields is not None:
            anns_fields = [{"field": af.field, "weight": af.weight} for af in req.anns_fields]

        results = service.hybrid_search(
            query=req.query,
            top_k=req.top_k,
            filter_expr=req.filter_expr,
            vector_weight=req.vector_weight,
            bm25_weight=req.bm25_weight,
            reranker=req.reranker,
            rrf_k=req.rrf_k,
            output_fields=req.output_fields,
            collection_names=req.collection_names,
            anns_field=req.anns_field,
            anns_fields=anns_fields,
        )
    finally:
        # 恢复原始数据库
        if original_db is not None:
            service.client.using_database(original_db)

    items = [_to_item(r) for r in results]
    return SearchResponse(results=items, total=len(items))


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    # 切换到指定数据库
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)

    try:
        results = service.query(
            filter_expr=req.filter_expr,
            limit=req.limit,
            offset=req.offset,
            output_fields=req.output_fields,
            collection_names=req.collection_names,
        )
    finally:
        # 恢复原始数据库
        if original_db is not None:
            service.client.using_database(original_db)

    return QueryResponse(results=results, total=len(results))


@router.post("/count", response_model=CountResponse)
def count(
    req: CountRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    # 切换到指定数据库
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)

    try:
        n = service.count(filter_expr=req.filter_expr)
    finally:
        # 恢复原始数据库
        if original_db is not None:
            service.client.using_database(original_db)

    return CountResponse(count=n)


@router.post("/delete/chunk-ids", response_model=DeleteResponse)
def delete_by_chunk_ids(
    req: DeleteByChunkIdsRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)
    try:
        n = service.delete_by_chunk_ids(req.chunk_ids, collection_name=req.collection_name)
    finally:
        if original_db is not None:
            service.client.using_database(original_db)
    return DeleteResponse(deleted_count=n)


@router.post("/delete/source", response_model=DeleteResponse)
def delete_by_source(
    req: DeleteBySourceRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)
    try:
        n = service.delete_by_source(req.source_file, collection_name=req.collection_name)
    finally:
        if original_db is not None:
            service.client.using_database(original_db)
    return DeleteResponse(deleted_count=n)


@router.post("/delete/field", response_model=DeleteResponse)
def delete_by_field(
    req: DeleteByFieldRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    original_db = None
    if req.database:
        original_db = service.client.database
        service.client.using_database(req.database)
    try:
        n = service.delete_by_field(
            field_name=req.field_name,
            field_value=req.field_value,
            collection_name=req.collection_name,
        )
    finally:
        if original_db is not None:
            service.client.using_database(original_db)
    return DeleteResponse(deleted_count=n)


@router.post("/truncate/collections", response_model=MessageResponse)
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


@router.post("/truncate/database", response_model=MessageResponse)
def truncate_database(
    req: TruncateDatabaseRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    original_db = service.client.database
    service.client.using_database(req.database)
    try:
        collections = service.client.list_collections()
        total_deleted = 0
        for col_name in collections:
            total_deleted += service.truncate_collection(collection_name=col_name)
    finally:
        service.client.using_database(original_db)
    return MessageResponse(
        message=f"Truncated database '{req.database}': {len(collections)} collections, {total_deleted} rows deleted"
    )
