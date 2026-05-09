from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Common ───


class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ─── Health ───


class HealthResponse(BaseModel):
    status: str
    milvus_connected: bool
    embedding_model_loaded: bool
    embedding_device: str = "unknown"


# ─── Ingestion ───


class IngestRequest(BaseModel):
    folder_path: str = Field(..., description="入库文件夹绝对路径")
    chunk_size: int = Field(500, gt=0, description="分块大小（字符数）")
    chunk_overlap: int = Field(50, ge=0, description="分块重叠字符数")
    drop_if_exists: bool = Field(False, description="是否先删除已有集合")
    enable_bm25: bool = Field(True, description="是否启用 BM25 全文检索")
    upsert_mode: bool = Field(True, description="True=upsert 覆盖, False=insert 新增")


class IngestResponse(BaseModel):
    files_scanned: int
    chunks_parsed: int
    chunks_inserted: int


# ─── Database Management ───


class CreateDatabaseRequest(BaseModel):
    db_name: str


class DropDatabaseRequest(BaseModel):
    db_name: str


class UsingDatabaseRequest(BaseModel):
    db_name: str


class ListDatabasesResponse(BaseModel):
    databases: List[str]


# ─── Collection Management ───


class CreateCollectionRequest(BaseModel):
    dim: Optional[int] = Field(None, description="向量维度，None 则自动从嵌入模型获取")
    drop_if_exists: bool = False
    enable_bm25: bool = True


class InitCollectionRequest(BaseModel):
    drop_if_exists: bool = False
    enable_bm25: bool = True


class CollectionExistsResponse(BaseModel):
    exists: bool


class DescribeCollectionResponse(BaseModel):
    name: str
    description: str
    fields: List[Dict[str, Any]]


class BM25SupportResponse(BaseModel):
    has_bm25_support: bool


# ─── Search ───


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, gt=0)
    filter_expr: Optional[str] = None
    offset: int = Field(0, ge=0)
    output_fields: Optional[List[str]] = None


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, gt=0)
    filter_expr: Optional[str] = None
    vector_weight: float = Field(0.7, gt=0)
    bm25_weight: float = Field(0.3, gt=0)
    reranker: str = Field("weighted", pattern=r"^(weighted|rrf)$")
    rrf_k: int = Field(60, gt=0)
    output_fields: Optional[List[str]] = None


class QueryRequest(BaseModel):
    filter_expr: str
    limit: int = Field(100, gt=0)
    offset: int = Field(0, ge=0)
    output_fields: Optional[List[str]] = None


class CountRequest(BaseModel):
    filter_expr: Optional[str] = None


class DeleteByChunkIdsRequest(BaseModel):
    chunk_ids: List[str] = Field(..., min_length=1)


class DeleteBySourceRequest(BaseModel):
    source_file: str


# ─── Search Results ───


class SearchResultItem(BaseModel):
    chunk_id: str
    content: str
    score: float
    source_file: str
    file_name: str
    file_type: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    total: int


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int


class CountResponse(BaseModel):
    count: int


class DeleteResponse(BaseModel):
    deleted_count: int
