"""Collection profile — schema 和入库逻辑的唯一真相来源。

修改 DEFAULT_RAG_PROFILE.fields 即可同时影响:
  1. Milvus collection schema（哪些字段、什么类型）
  2. 入库数据映射（从 ParsedChunk 提取哪些字段）
  3. 搜索输出字段（哪些字段返回给调用方）
"""

from dataclasses import dataclass, fields as dc_fields, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pymilvus import DataType, FieldSchema


# ============================================================================
# DataType 映射
# ============================================================================

DTYPE_MAP: Dict[str, DataType] = {
    "BOOL": DataType.BOOL,
    "INT8": DataType.INT8,
    "INT16": DataType.INT16,
    "INT32": DataType.INT32,
    "INT64": DataType.INT64,
    "FLOAT": DataType.FLOAT,
    "DOUBLE": DataType.DOUBLE,
    "VARCHAR": DataType.VARCHAR,
    "JSON": DataType.JSON,
    "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    "SPARSE_FLOAT_VECTOR": DataType.SPARSE_FLOAT_VECTOR,
    "ARRAY": DataType.ARRAY,
}


# ============================================================================
# 字段声明
# ============================================================================

class FieldSpec(BaseModel):
    """单个 collection 字段的声明（同时用于 API 请求和 profile 定义）。"""
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str = Field(
        pattern=r"^(BOOL|INT8|INT16|INT32|INT64|FLOAT|DOUBLE|VARCHAR|JSON|FLOAT_VECTOR|SPARSE_FLOAT_VECTOR|ARRAY)$",
    )
    is_primary: bool = False
    auto_id: bool = False
    max_length: Optional[int] = None
    dim: Optional[int] = None
    enable_analyzer: Optional[bool] = None
    enable_match: Optional[bool] = None
    enable_bm25: bool = False
    enable_embedding: bool = False
    in_output: bool = True
    element_type: Optional[str] = None
    max_capacity: Optional[int] = None
    embedding_source: Optional[str] = None
    description: str = ""


@dataclass
class CollectionProfile:
    """一个 Milvus collection 的完整声明。"""
    name: str
    fields: List[FieldSpec]
    description: str = "RAG knowledge base chunks"

    @property
    def primary_key_field(self) -> FieldSpec:
        return next(f for f in self.fields if f.is_primary)

    @property
    def output_fields(self) -> List[str]:
        return [
            f.name for f in self.fields
            if f.in_output and f.dtype not in ("FLOAT_VECTOR", "SPARSE_FLOAT_VECTOR")
        ]

    @property
    def bm25_fields(self) -> List[FieldSpec]:
        """所有 enable_bm25=True 的 VARCHAR 字段。"""
        return [f for f in self.fields if f.dtype == "VARCHAR" and f.enable_bm25]

    @property
    def embedding_fields(self) -> List[FieldSpec]:
        """所有 enable_embedding=True 的 VARCHAR 字段。"""
        return [f for f in self.fields if f.dtype == "VARCHAR" and f.enable_embedding]

    def get_field(self, name: str) -> Optional[FieldSpec]:
        return next((f for f in self.fields if f.name == name), None)

    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def to_field_schemas(self, dim: Optional[int] = None) -> List[FieldSchema]:
        """将 profile 中所有字段转为 pymilvus FieldSchema 列表。"""
        return [to_field_schema(f, dim=dim) for f in self.fields]


# ============================================================================
# 默认 RAG Profile，建表语句
# 新增bm25字段开启enable_bm25=True，自动生成 X_sparse
# 新增embedding字段开启enable_embedding=True，自动生成 X_embedding
# ============================================================================

DEFAULT_RAG_PROFILE = CollectionProfile(
    name="rag_chunks",
    fields=[
        FieldSpec(name="chunk_id",    dtype="VARCHAR",       is_primary=True, max_length=256,  in_output=True),
        FieldSpec(name="content",     dtype="VARCHAR",       max_length=16384,enable_bm25=True,enable_embedding=True,in_output=True),
        FieldSpec(name="source_file", dtype="VARCHAR",       max_length=512,  in_output=True),
        FieldSpec(name="file_name",   dtype="VARCHAR",       max_length=256,  in_output=True),
        FieldSpec(name="file_type",   dtype="VARCHAR",       max_length=50,   in_output=True),
        FieldSpec(name="chunk_index", dtype="INT64",         in_output=True),
        FieldSpec(name="total_chunks", dtype="INT64",        in_output=False),
        FieldSpec(name="page_number", dtype="INT64",         in_output=False),
        FieldSpec(name="sheet_name",  dtype="VARCHAR",       max_length=256,  in_output=False),
        FieldSpec(name="title",       dtype="VARCHAR",       max_length=512,  in_output=False),
        FieldSpec(name="file_size",   dtype="INT64",         in_output=False),
        FieldSpec(name="created_at",  dtype="VARCHAR",       max_length=64,   in_output=False),
        FieldSpec(name="modified_at", dtype="VARCHAR",       max_length=64,   in_output=False),
        FieldSpec(name="metadata",    dtype="JSON",          in_output=True),
    ],
)

# ============================================================================
# 文本装载类
# ============================================================================
@dataclass
class ParsedChunk:
    """解析后的文本块，可直接入库 Milvus。"""
    chunk_id: str
    content: str
    source_file: str
    file_name: str
    file_type: str
    chunk_index: int
    total_chunks: int
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    title: Optional[str] = None
    file_size: int = 0
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# FieldSpec → pymilvus FieldSchema 转换
# ============================================================================

def to_field_schema(fspec: FieldSpec, dim: Optional[int] = None) -> FieldSchema:
    """将 FieldSpec 转为 pymilvus FieldSchema。"""
    dtype_enum = DTYPE_MAP[fspec.dtype]
    kwargs: Dict[str, Any] = {
        "name": fspec.name,
        "dtype": dtype_enum,
        "description": fspec.description,
        "is_primary": fspec.is_primary,
    }
    if fspec.auto_id:
        kwargs["auto_id"] = True

    if dtype_enum == DataType.VARCHAR:
        kwargs["max_length"] = fspec.max_length if fspec.max_length is not None else 256
        # enable_bm25 自动开启 enable_analyzer 和 enable_match
        if fspec.enable_bm25:
            kwargs["enable_analyzer"] = True
            kwargs["enable_match"] = True
        else:
            if fspec.enable_analyzer is not None:
                kwargs["enable_analyzer"] = fspec.enable_analyzer
            if fspec.enable_match is not None:
                kwargs["enable_match"] = fspec.enable_match
    elif dtype_enum == DataType.FLOAT_VECTOR:
        effective_dim = fspec.dim if fspec.dim is not None else dim
        if effective_dim is None:
            raise ValueError(
                f"Field '{fspec.name}' is FLOAT_VECTOR but no dim provided. "
                "Set dim on the field or provide dim in the request."
            )
        kwargs["dim"] = effective_dim
    elif dtype_enum == DataType.ARRAY:
        if fspec.element_type is None:
            raise ValueError(f"Field '{fspec.name}' is ARRAY but element_type not provided")
        kwargs["element_type"] = DTYPE_MAP[fspec.element_type]
        if fspec.max_capacity is not None:
            kwargs["max_capacity"] = fspec.max_capacity

    return FieldSchema(**kwargs)


# ============================================================================
# 入库数据映射
# ============================================================================

def build_insert_rows(
    chunks: List[ParsedChunk],
    profile: CollectionProfile,
    embedder,                    # milvus.embedder.EmbeddingModel
    max_content_length: int = 16384,
) -> List[Dict[str, Any]]:
    """将 ParsedChunk 列表转为可插入 Milvus 的行列表。

    向量字段会自动从对应的源文本字段编码生成。
    约定: 源字段 X 开启 enable_embedding=True 后，自动生成 X_embedding 向量字段。
    """
    # 收集向量字段信息: (field_name, source_field_name)
    chunk_field_names = {f.name for f in dc_fields(ParsedChunk)}
    vec_fields: List[tuple] = []
    for f in profile.fields:
        if f.dtype == "VARCHAR" and f.enable_embedding:
            vec_name = f"{f.name}_embedding"
            source = f.name
            vec_fields.append((vec_name, source))
        elif f.dtype == "FLOAT_VECTOR":
            # 兼容手动声明的 FLOAT_VECTOR 字段
            if f.embedding_source and f.embedding_source in chunk_field_names:
                source = f.embedding_source
            else:
                source = f.name.replace("_embedding", "").rstrip("_") or "content"
                if source not in chunk_field_names:
                    source = "content"
            vec_fields.append((f.name, source))

    # 第一遍：构建非向量字段
    rows: List[Dict[str, Any]] = []
    for chunk in chunks:
        row: Dict[str, Any] = {}
        for f in profile.fields:
            if f.dtype in ("FLOAT_VECTOR", "SPARSE_FLOAT_VECTOR"):
                continue
            if f.is_primary:
                row[f.name] = chunk.chunk_id
            elif f.dtype == "JSON":
                row[f.name] = chunk.metadata if chunk.metadata else {}
            elif hasattr(chunk, f.name):
                value = getattr(chunk, f.name, None)
                if value is None:
                    value = 0 if f.dtype in ("INT64", "INT32", "INT16", "INT8", "FLOAT", "DOUBLE") else ""
                row[f.name] = value
            else:
                row[f.name] = ""
        rows.append(row)

    # 第二遍：批量编码向量字段
    for vec_field, source_attr in vec_fields:
        texts = []
        for row in rows:
            text = row.get(source_attr, "")
            if text is None:
                text = ""
            if isinstance(text, str) and len(text) > max_content_length:
                text = text[:max_content_length]
            texts.append(text)

        embeddings = embedder.encode(texts)
        for row, emb in zip(rows, embeddings):
            row[vec_field] = emb

    return rows
