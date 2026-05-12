from fastapi import APIRouter, Depends

from api.deps import get_milvus_service
from api.schemas import (
    CountRequest,
    CountResponse,
    HybridSearchRequest,
    QueryRequest,
    QueryResponse,
    SearchResponse,
    SearchResultItem,
)
from retrieval.service import MilvusService, SearchResult

router = APIRouter()


def _to_item(r: SearchResult) -> SearchResultItem:
    return SearchResultItem(score=r.score, fields=r.fields)


def _switch_db(service: MilvusService, database):
    if database and database != service.client.database:
        service.client.using_database(database)


@router.post("/hybrid", response_model=SearchResponse)
def hybrid_search(
    req: HybridSearchRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    _switch_db(service, req.database)

    anns_fields = None
    if req.anns_fields is not None:
        anns_fields = [{"field": af.field, "weight": af.weight} for af in req.anns_fields]

    bm25_fields = None
    if req.bm25_fields is not None:
        bm25_fields = [{"field": bf.field, "weight": bf.weight} for bf in req.bm25_fields]

    results = service.hybrid_search(
        query=req.query,
        top_k=req.top_k,
        filter_expr=req.filter_expr,
        reranker=req.reranker,
        rrf_k=req.rrf_k,
        output_fields=req.output_fields,
        collection_names=req.collection_names,
        anns_fields=anns_fields,
        bm25_fields=bm25_fields,
    )

    items = [_to_item(r) for r in results]
    return SearchResponse(results=items, total=len(items))


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    _switch_db(service, req.database)
    results = service.query(
        filter_expr=req.filter_expr,
        limit=req.limit,
        offset=req.offset,
        output_fields=req.output_fields,
        collection_names=req.collection_names,
    )
    return QueryResponse(results=results, total=len(results))


@router.post("/count", response_model=CountResponse)
def count(
    req: CountRequest,
    service: MilvusService = Depends(get_milvus_service),
):
    _switch_db(service, req.database)
    n = service.count(filter_expr=req.filter_expr)
    return CountResponse(count=n)
