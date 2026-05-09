from fastapi import APIRouter, Depends

from api.deps import get_milvus_service
from api.schemas import (
    CountRequest,
    CountResponse,
    DeleteByChunkIdsRequest,
    DeleteBySourceRequest,
    DeleteResponse,
    HybridSearchRequest,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from milvus.service import MilvusService, SearchResult

router = APIRouter()


def _to_item(r: SearchResult) -> SearchResultItem:
    return SearchResultItem(
        chunk_id=r.chunk_id,
        content=r.content,
        score=r.score,
        source_file=r.source_file,
        file_name=r.file_name,
        file_type=r.file_type,
        chunk_index=r.chunk_index,
        metadata=r.metadata,
    )


@router.post("/vector", response_model=SearchResponse)
def vector_search(
    req: SearchRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    results = service.search(
        query=req.query,
        top_k=req.top_k,
        filter_expr=req.filter_expr,
        offset=req.offset,
        output_fields=req.output_fields,
    )
    items = [_to_item(r) for r in results]
    return SearchResponse(results=items, total=len(items))


@router.post("/hybrid", response_model=SearchResponse)
def hybrid_search(
    req: HybridSearchRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    results = service.hybrid_search(
        query=req.query,
        top_k=req.top_k,
        filter_expr=req.filter_expr,
        vector_weight=req.vector_weight,
        bm25_weight=req.bm25_weight,
        reranker=req.reranker,
        rrf_k=req.rrf_k,
        output_fields=req.output_fields,
    )
    items = [_to_item(r) for r in results]
    return SearchResponse(results=items, total=len(items))


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    results = service.query(
        filter_expr=req.filter_expr,
        limit=req.limit,
        offset=req.offset,
        output_fields=req.output_fields,
    )
    return QueryResponse(results=results, total=len(results))


@router.post("/count", response_model=CountResponse)
def count(
    req: CountRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    n = service.count(filter_expr=req.filter_expr)
    return CountResponse(count=n)


@router.post("/delete/chunk-ids", response_model=DeleteResponse)
def delete_by_chunk_ids(
    req: DeleteByChunkIdsRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    n = service.delete_by_chunk_ids(req.chunk_ids)
    return DeleteResponse(deleted_count=n)


@router.post("/delete/source", response_model=DeleteResponse)
def delete_by_source(
    req: DeleteBySourceRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    n = service.delete_by_source(req.source_file)
    return DeleteResponse(deleted_count=n)
