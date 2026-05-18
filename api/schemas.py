from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from retrieval.profile import FieldSpec


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
    upsert_mode: bool = Field(True, description="True=upsert 覆盖, False=insert 新增")
    collection_name: Optional[str] = Field(None, description="目标集合名称，不提供则使用默认值")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


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


class VectorIndexParams(BaseModel):
    field_name: str = Field("embedding", description="向量字段名")
    index_type: str = Field("IVF_FLAT", description="索引类型：IVF_FLAT, HNSW, FLAT 等")
    metric_type: str = Field("COSINE", description="度量类型：COSINE, L2, IP 等")
    params: Dict[str, Any] = Field(
        default_factory=lambda: {"nlist": 128},
        description="索引参数，如 IVF_FLAT 的 {\"nlist\": 128}，HNSW 的 {\"M\": 16, \"efConstruction\": 256}",
    )


class CreateCollectionRequest(BaseModel):
    dim: Optional[int] = Field(None, description="向量维度，None 则自动从嵌入模型获取")
    drop_if_exists: bool = False
    collection_name: Optional[str] = Field(None, description="自定义集合名称，不提供则使用默认值")
    fields: Optional[List[FieldSpec]] = Field(None, description="自定义字段列表，不提供则使用默认 RAG 字段")
    vector_index: Optional[List[VectorIndexParams]] = Field(None, description="向量索引参数列表")
    description: Optional[str] = Field(None, description="集合 schema 描述")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class InitCollectionRequest(BaseModel):
    drop_if_exists: bool = False
    collection_name: Optional[str] = Field(None, description="自定义集合名称")
    fields: Optional[List[FieldSpec]] = Field(None, description="自定义字段列表")
    vector_index: Optional[List[VectorIndexParams]] = Field(None, description="向量索引参数列表")
    description: Optional[str] = Field(None, description="集合 schema 描述")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class CollectionExistsResponse(BaseModel):
    exists: bool


class DescribeCollectionResponse(BaseModel):
    name: str
    description: str
    fields: List[Dict[str, Any]]


class DropCollectionRequest(BaseModel):
    collection_name: Optional[str] = Field(None, description="集合名称，不提供则使用默认值")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class BM25SupportResponse(BaseModel):
    has_bm25_support: bool


# ─── Search ───


class AnnsFieldWeight(BaseModel):
    field: str = Field(..., description="向量字段名")
    weight: float = Field(..., ge=0, description="该字段的检索权重，0 表示不参与排序")


class BM25FieldWeight(BaseModel):
    field: str = Field(..., description="BM25 文本字段名（如 content、file_name）")
    weight: float = Field(..., ge=0, description="该字段的检索权重，0 表示不参与排序")


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, gt=0)
    filter_expr: Optional[str] = None
    reranker: str = Field("weighted", pattern=r"^(weighted|rrf)$")
    rrf_k: int = Field(60, gt=0)
    output_fields: Optional[List[str]] = None
    collection_names: Optional[List[str]] = Field(None, description="要检索的集合列表，不提供则使用默认集合")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")
    anns_fields: Optional[List[AnnsFieldWeight]] = Field(None, description="向量字段检索配置，每项指定字段名和权重；不提供则使用 profile 中所有 FLOAT_VECTOR 字段均分权重")
    bm25_fields: Optional[List[BM25FieldWeight]] = Field(None, description="BM25字段检索配置，每项指定字段名和权重；不提供则使用 profile 中所有 enable_bm25 字段均分权重")


class QueryRequest(BaseModel):
    filter_expr: str
    limit: int = Field(100, gt=0)
    offset: int = Field(0, ge=0)
    output_fields: Optional[List[str]] = None
    collection_names: Optional[List[str]] = Field(None, description="要查询的集合列表，不提供则使用默认集合")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class CountRequest(BaseModel):
    filter_expr: Optional[str] = None
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class DeleteByChunkIdsRequest(BaseModel):
    chunk_ids: List[str] = Field(..., min_length=1)
    collection_name: Optional[str] = Field(None, description="集合名称，不提供则使用默认值")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class DeleteBySourceRequest(BaseModel):
    source_file: str
    collection_name: Optional[str] = Field(None, description="集合名称，不提供则使用默认值")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class DeleteByFieldRequest(BaseModel):
    field_name: str = Field(..., description="字段名")
    field_value: Any = Field(..., description="匹配的值")
    collection_name: Optional[str] = Field(None, description="集合名称，不提供则使用默认值")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class TruncateCollectionsRequest(BaseModel):
    collection_names: List[str] = Field(..., min_length=1, description="要清空的集合名称列表")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class TruncateDatabaseRequest(BaseModel):
    database: str = Field(..., description="要清空的数据库")


# ─── Search Results ───


class SearchResultItem(BaseModel):
    score: float
    fields: Dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs):
        """扁平化：把 fields 内容提升到顶层，保持向后兼容"""
        d = super().model_dump(**kwargs)
        flat = {"score": d.pop("score")}
        flat.update(d.pop("fields", {}))
        return flat


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


# ─── Chat ───


class SourceCitation(BaseModel):
    index: int = Field(..., description="引用编号，从1开始")
    chunk_id: str = Field(..., description="文档块ID")
    source_file: str = Field(..., description="源文件路径")
    score: float = Field(..., description="检索相关度分数")
    content: str = Field(..., description="文档块内容摘要")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    conversation_id: Optional[str] = Field(None, description="对话ID，不提供则创建新对话")
    top_k: int = Field(5, gt=0, description="检索返回的文档数量")
    collection_names: Optional[List[str]] = Field(None, description="检索的集合列表")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    conversation_id: str


class ConversationResponse(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: float
    message_count: int
    messages: Optional[List[Dict[str, Any]]] = None
