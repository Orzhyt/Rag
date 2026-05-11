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


class FieldDefinition(BaseModel):
    name: str = Field(..., description="字段名")
    dtype: str = Field(
        ...,
        description="Milvus DataType 名称，如 VARCHAR, INT64, FLOAT_VECTOR, JSON, BOOL, SPARSE_FLOAT_VECTOR",
        pattern=r"^(BOOL|INT8|INT16|INT32|INT64|FLOAT|DOUBLE|VARCHAR|JSON|FLOAT_VECTOR|SPARSE_FLOAT_VECTOR|ARRAY)$",
    )
    is_primary: bool = Field(False, description="是否为主键")
    auto_id: bool = Field(False, description="是否自动生成主键")
    max_length: Optional[int] = Field(None, description="VARCHAR 最大长度")
    enable_analyzer: Optional[bool] = Field(None, description="启用文本分析器（BM25 需要）")
    enable_match: Optional[bool] = Field(None, description="启用文本匹配（BM25 需要）")
    dim: Optional[int] = Field(None, description="FLOAT_VECTOR 维度")
    element_type: Optional[str] = Field(None, description="ARRAY 元素类型")
    max_capacity: Optional[int] = Field(None, description="ARRAY 最大容量")
    embedding_source: Optional[str] = Field(None, description="FLOAT_VECTOR 字段的嵌入源字段名，如 content、title；不提供则默认嵌入 content")
    description: str = Field("", description="字段描述")


class VectorIndexParams(BaseModel):
    field_name: str = Field("embedding", description="向量字段名")
    index_type: str = Field("IVF_FLAT", description="索引类型：IVF_FLAT, HNSW, FLAT 等")
    metric_type: str = Field("COSINE", description="度量类型：COSINE, L2, IP 等")
    params: Dict[str, Any] = Field(
        default_factory=lambda: {"nlist": 128},
        description="索引参数，如 IVF_FLAT 的 {\"nlist\": 128}，HNSW 的 {\"M\": 16, \"efConstruction\": 256}",
    )


class BM25Config(BaseModel):
    text_field_name: str = Field("content", description="BM25 输入的 VARCHAR 字段名（须 enable_analyzer=True）")
    sparse_field_name: str = Field("content_sparse", description="自动创建的 SPARSE_FLOAT_VECTOR 输出字段名")
    function_name: str = Field("content_bm25", description="BM25 Function 名称")


class CreateCollectionRequest(BaseModel):
    dim: Optional[int] = Field(None, description="向量维度，None 则自动从嵌入模型获取")
    drop_if_exists: bool = False
    enable_bm25: bool = True
    collection_name: Optional[str] = Field(None, description="自定义集合名称，不提供则使用默认值")
    fields: Optional[List[FieldDefinition]] = Field(None, description="自定义字段列表，不提供则使用默认 RAG 字段")
    vector_index: Optional[List[VectorIndexParams]] = Field(None, description="向量索引参数列表")
    bm25_config: Optional[BM25Config] = Field(None, description="BM25 配置，仅在 enable_bm25=True 时生效")
    description: Optional[str] = Field(None, description="集合 schema 描述")
    embedding_field_name: Optional[str] = Field(None, description="嵌入向量字段名，用于创建索引")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")


class InitCollectionRequest(BaseModel):
    drop_if_exists: bool = False
    enable_bm25: bool = True
    collection_name: Optional[str] = Field(None, description="自定义集合名称")
    fields: Optional[List[FieldDefinition]] = Field(None, description="自定义字段列表")
    vector_index: Optional[List[VectorIndexParams]] = Field(None, description="向量索引参数列表")
    bm25_config: Optional[BM25Config] = Field(None, description="BM25 配置")
    description: Optional[str] = Field(None, description="集合 schema 描述")
    embedding_field_name: Optional[str] = Field(None, description="嵌入向量字段名")
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


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, gt=0)
    filter_expr: Optional[str] = None
    offset: int = Field(0, ge=0)
    output_fields: Optional[List[str]] = None
    collection_names: Optional[List[str]] = Field(None, description="要检索的集合列表，不提供则使用默认集合")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")
    anns_field: Optional[str] = Field(None, description="要检索的向量字段名，不提供则使用 profile 中的 embedding_field")


class AnnsFieldWeight(BaseModel):
    field: str = Field(..., description="向量字段名")
    weight: float = Field(..., gt=0, description="该字段的检索权重")


class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, gt=0)
    filter_expr: Optional[str] = None
    vector_weight: float = Field(0.7, gt=0)
    bm25_weight: float = Field(0.3, gt=0)
    reranker: str = Field("weighted", pattern=r"^(weighted|rrf)$")
    rrf_k: int = Field(60, gt=0)
    output_fields: Optional[List[str]] = None
    collection_names: Optional[List[str]] = Field(None, description="要检索的集合列表，不提供则使用默认集合")
    database: Optional[str] = Field(None, description="指定 Milvus 数据库，不提供则使用当前数据库")
    anns_field: Optional[str] = Field(None, description="要检索的向量字段名，不提供则使用 profile 中的 embedding_field")
    anns_fields: Optional[List[AnnsFieldWeight]] = Field(None, description="多向量字段检索配置，每项指定字段名和权重；提供时忽略 anns_field 和 vector_weight")


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
