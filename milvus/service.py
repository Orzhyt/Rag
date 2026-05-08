import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_pipeline.parser import ParsedChunk
from common.logger import setup_logger

from .client import MAX_CONTENT_LENGTH, MilvusClient
from .embedder import EmbeddingModel

logger = setup_logger("milvus.service")

# 单次插入最大行数
MAX_INSERT_BATCH_SIZE = 1000
# 向量化批大小
EMBED_BATCH_SIZE = 32


@dataclass
class SearchResult:
    """向量搜索结果"""
    chunk_id: str
    content: str
    score: float
    source_file: str
    file_name: str
    file_type: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_hit(cls, hit) -> "SearchResult":
        entity = hit.entity
        return cls(
            chunk_id=entity.get("chunk_id") or "",
            content=entity.get("content") or "",
            score=hit.score,
            source_file=entity.get("source_file") or "",
            file_name=entity.get("file_name") or "",
            file_type=entity.get("file_type") or "",
            chunk_index=entity.get("chunk_index") or 0,
            metadata=entity.get("metadata") or {},
        )


class MilvusService:
    """Milvus 数据服务：入库、增量更新、查询、删除"""

    def __init__(self, client: MilvusClient = None, embedder: EmbeddingModel = None):
        self.client = client or MilvusClient()
        self.embedder = embedder or EmbeddingModel()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def init_collection(self, drop_if_exists: bool = False):
        """初始化集合：连接 Milvus、创建集合（若不存在）、加载到内存"""
        self.client.connect()
        if not self.client.collection_exists() or drop_if_exists:
            self.client.create_collection(dim=self.embedder.dim, drop_if_exists=drop_if_exists)
        self.client.load_collection()

    # ------------------------------------------------------------------
    # 入库
    # ------------------------------------------------------------------

    def insert(self, chunks: List[ParsedChunk]) -> int:
        """批量入库新数据（chunk_id 重复会报错，请用 upsert 做增量更新）"""
        if not chunks:
            return 0

        total = 0
        for i in range(0, len(chunks), MAX_INSERT_BATCH_SIZE):
            batch = chunks[i:i + MAX_INSERT_BATCH_SIZE]
            rows = self._build_rows(batch)
            if rows:
                col = self.client.get_collection()
                col.insert(rows)
                total += len(rows)

        if total > 0:
            self._flush()
            logger.info("Inserted %d chunks", total)
        return total

    # ------------------------------------------------------------------
    # 增量更新
    # ------------------------------------------------------------------

    def upsert(self, chunks: List[ParsedChunk]) -> int:
        """增量更新：按 chunk_id 覆盖已有数据，不存在则插入"""
        if not chunks:
            return 0

        total = 0
        for i in range(0, len(chunks), MAX_INSERT_BATCH_SIZE):
            batch = chunks[i:i + MAX_INSERT_BATCH_SIZE]
            rows = self._build_rows(batch)
            if rows:
                col = self.client.get_collection()
                col.upsert(rows)
                total += len(rows)

        if total > 0:
            self._flush()
            logger.info("Upserted %d chunks", total)
        return total

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        offset: int = 0,
        output_fields: List[str] = None,
    ) -> List[SearchResult]:
        """向量相似度搜索

        Args:
            query: 查询文本
            top_k: 返回结果数
            filter_expr: Milvus 标量过滤表达式，如 'file_type == "pdf"'
            offset: 分页偏移
            output_fields: 要返回的标量字段，默认返回所有
        """
        query_vec = self.embedder.encode([query])[0]

        if output_fields is None:
            output_fields = [
                "chunk_id", "content", "source_file", "file_name",
                "file_type", "chunk_index", "metadata",
            ]

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        col = self.client.get_collection()

        results = col.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            offset=offset,
            expr=filter_expr,
            output_fields=output_fields,
        )

        return [SearchResult.from_hit(hit) for hit in results[0]]

    def query(
        self,
        filter_expr: str,
        limit: int = 100,
        offset: int = 0,
        output_fields: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """标量查询（无向量搜索）

        Args:
            filter_expr: Milvus 过滤表达式，如 'file_type == "pdf"'
            limit: 返回数量上限
            offset: 分页偏移
            output_fields: 要返回的字段，默认返回所有标量字段
        """
        if output_fields is None:
            output_fields = ["*"]

        col = self.client.get_collection()
        return col.query(
            expr=filter_expr,
            offset=offset,
            limit=limit,
            output_fields=output_fields,
        )

    def count(self, filter_expr: Optional[str] = None) -> int:
        """统计 chunk 数量（不含已软删除的实体）"""
        col = self.client.get_collection()
        expr = filter_expr or 'chunk_id != ""'
        # Milvus query 的 offset+limit 上限为 16384，需分页累加
        page_size = 16384
        total = 0
        offset = 0
        while True:
            batch = col.query(expr=expr, limit=page_size, offset=offset, output_fields=["chunk_id"])
            total += len(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return total

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def delete_by_chunk_ids(self, chunk_ids: List[str]) -> int:
        """按 chunk_id 批量删除"""
        if not chunk_ids:
            return 0
        col = self.client.get_collection()
        ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
        expr = f'chunk_id in [{ids_str}]'
        result = col.delete(expr)
        self._flush()
        deleted = result.delete_count if hasattr(result, "delete_count") else 0
        logger.info("Deleted %d chunks by chunk_ids", deleted)
        return deleted

    def delete_by_source(self, source_file: str) -> int:
        """按源文件路径删除其所有 chunk"""
        col = self.client.get_collection()
        expr = f'source_file == "{source_file}"'
        result = col.delete(expr)
        self._flush()
        deleted = result.delete_count if hasattr(result, "delete_count") else 0
        logger.info("Deleted %d chunks for source: %s", deleted, source_file)
        return deleted

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_rows(self, chunks: List[ParsedChunk]) -> List[Dict[str, Any]]:
        """将 ParsedChunk 列表转为 Milvus 插入行，包含向量"""
        contents = []
        for c in chunks:
            # 截断过长内容
            if len(c.content) > MAX_CONTENT_LENGTH:
                logger.warning(
                    "Chunk %s content truncated: %d -> %d",
                    c.chunk_id, len(c.content), MAX_CONTENT_LENGTH,
                )
            contents.append(c.content[:MAX_CONTENT_LENGTH])

        embeddings = self.embedder.encode(contents, batch_size=EMBED_BATCH_SIZE)

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            row = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content[:MAX_CONTENT_LENGTH],
                "embedding": embedding,
                "source_file": chunk.source_file,
                "file_name": chunk.file_name,
                "file_type": chunk.file_type,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "page_number": chunk.page_number or 0,
                "sheet_name": chunk.sheet_name or "",
                "title": chunk.title or "",
                "file_size": chunk.file_size,
                "created_at": chunk.created_at or "",
                "modified_at": chunk.modified_at or "",
                "metadata": chunk.metadata if chunk.metadata else {},
            }
            rows.append(row)
        return rows

    def _flush(self):
        """刷写集合数据"""
        self.client.get_collection().flush()
