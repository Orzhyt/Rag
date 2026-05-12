from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

from retrieval.profile import ParsedChunk
from common.logger import setup_logger

from milvus.client import  MilvusClient
from milvus.embedder import EmbeddingModel
from retrieval.profile import CollectionProfile, DEFAULT_RAG_PROFILE, build_insert_rows

logger = setup_logger("milvus.service")

# 单次插入最大行数
MAX_INSERT_BATCH_SIZE = 1000


@dataclass
class SearchResult:
    """泛型搜索结果，字段存储在 dict 中"""
    fields: Dict[str, Any]
    score: float

    @classmethod
    def from_hit(cls, hit, profile: CollectionProfile, collection_name: str = "") -> "SearchResult":
        entity = hit.entity
        fields = {}
        for fspec in profile.fields:
            if fspec.in_output and fspec.dtype not in ("FLOAT_VECTOR", "SPARSE_FLOAT_VECTOR"):
                value = entity.get(fspec.name)
                fields[fspec.name] = value
        if collection_name:
            fields["collection_name"] = collection_name
        return cls(fields=fields, score=hit.score)

    @property
    def chunk_id(self):
        return self.fields.get("chunk_id", "")

    @property
    def content(self):
        return self.fields.get("content", "")

    @property
    def source_file(self):
        return self.fields.get("source_file", "")

    @property
    def file_name(self):
        return self.fields.get("file_name", "")

    @property
    def file_type(self):
        return self.fields.get("file_type", "")

    @property
    def chunk_index(self):
        return self.fields.get("chunk_index", 0)

    @property
    def metadata(self):
        return self.fields.get("metadata", {})


class MilvusService:
    """Milvus 数据服务：入库、增量更新、查询、删除"""

    def __init__(self, client: MilvusClient = None, embedder: EmbeddingModel = None,
                 profile: CollectionProfile = None):
        self.client = client or MilvusClient()
        self.embedder = embedder or EmbeddingModel()
        self.profile = profile or DEFAULT_RAG_PROFILE
        self._loaded_collections: set = set()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def init_collection(
        self,
        drop_if_exists: bool = False,
        collection_name: Optional[str] = None,
        fields: Optional[List] = None,
        vector_index: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        self.client.connect()
        if collection_name is not None:
            self.client.collection_name = collection_name
        effective_name = self.client.collection_name
        if not self.client.has_collection(effective_name) or drop_if_exists:
            self.client.create_collection(
                dim=self.embedder.dim,
                drop_if_exists=drop_if_exists,
                collection_name=collection_name,
                fields=fields,
                vector_index=vector_index,
                description=description,
            )
        elif not self.client.has_bm25_support:
            logger.warning(
                "Collection '%s' exists but lacks BM25 sparse vector field. "
                "Hybrid search will not work. Re-create with drop_if_exists=True.",
                self.client.collection_name,
            )
        self.client.load_collection()

    # ------------------------------------------------------------------
    # 入库
    # ------------------------------------------------------------------

    def insert(self, chunks: List[ParsedChunk]) -> int:
        if not chunks:
            return 0

        total = 0
        failed = 0
        for i in range(0, len(chunks), MAX_INSERT_BATCH_SIZE):
            batch = chunks[i:i + MAX_INSERT_BATCH_SIZE]
            try:
                rows = build_insert_rows(batch, self.profile, self.embedder)
                if rows:
                    self.client.insert(rows)
                    total += len(rows)
            except Exception as e:
                failed += len(batch)
                logger.error("Insert batch failed [%d:%d]: %s", i, i + len(batch), e)

        if total > 0:
            self._flush()
        logger.info("Inserted %d chunks, failed %d", total, failed)
        return total

    # ------------------------------------------------------------------
    # 增量更新
    # ------------------------------------------------------------------

    def upsert(self, chunks: List[ParsedChunk]) -> int:
        if not chunks:
            return 0

        total = 0
        failed = 0
        for i in range(0, len(chunks), MAX_INSERT_BATCH_SIZE):
            batch = chunks[i:i + MAX_INSERT_BATCH_SIZE]
            try:
                rows = build_insert_rows(batch, self.profile, self.embedder)
                if rows:
                    self.client.upsert(rows)
                    total += len(rows)
            except Exception as e:
                failed += len(batch)
                logger.error("Upsert batch failed [%d:%d]: %s", i, i + len(batch), e)

        if total > 0:
            self._flush()
        logger.info("Upserted %d chunks, failed %d", total, failed)
        return total

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def _ensure_loaded(self, collection_names: List[str]):
        """确保集合已加载到内存（每个集合只 load 一次）"""
        for col_name in collection_names:
            if col_name not in self._loaded_collections:
                try:
                    self.client._client.load_collection(col_name)
                    self._loaded_collections.add(col_name)
                except Exception:
                    pass

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        reranker: str = "weighted",
        rrf_k: int = 60,
        output_fields: List[str] = None,
        collection_names: Optional[List[str]] = None,
        anns_fields: Optional[List[Dict[str, Any]]] = None,
        bm25_fields: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SearchResult]:
        if collection_names is None:
            collection_names = [self.client.collection_name]
        self._ensure_loaded(collection_names)

        query_vec = self.embedder.encode([query])[0]

        if output_fields is None:
            output_fields = self.profile.output_fields

        # 构建所有 AnnSearchRequest 和对应权重
        reqs = []
        weights = []

        # 向量字段: None=从 profile 默认, []=不使用, list=显式指定
        if anns_fields is not None:
            for item in anns_fields:
                reqs.append(AnnSearchRequest(
                    data=[query_vec],
                    anns_field=item["field"],
                    param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                    limit=top_k,
                    expr=filter_expr,
                ))
                weights.append(item["weight"])
        else:
            vec_fields = [f for f in self.profile.fields if f.dtype == "FLOAT_VECTOR"]
            if vec_fields:
                per_vec_weight = 0.7 / len(vec_fields)
                for vf in vec_fields:
                    reqs.append(AnnSearchRequest(
                        data=[query_vec],
                        anns_field=vf.name,
                        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                        limit=top_k,
                        expr=filter_expr,
                    ))
                    weights.append(per_vec_weight)

        # BM25 字段: None=从 profile 默认, []=不使用, list=显式指定
        if bm25_fields is not None:
            for item in bm25_fields:
                sparse_req = AnnSearchRequest(
                    data=[query],
                    anns_field=f"{item['field']}_sparse",
                    param={"metric_type": "BM25"},
                    limit=top_k,
                    expr=filter_expr,
                )
                reqs.append(sparse_req)
                weights.append(item["weight"])
        else:
            bm25_field_list = self.profile.bm25_fields
            if bm25_field_list:
                per_bm25_weight = 0.3 / len(bm25_field_list)
                for bm25_f in bm25_field_list:
                    sparse_req = AnnSearchRequest(
                        data=[query],
                        anns_field=f"{bm25_f.name}_sparse",
                        param={"metric_type": "BM25"},
                        limit=top_k,
                        expr=filter_expr,
                    )
                    reqs.append(sparse_req)
                    weights.append(per_bm25_weight)

        if reranker == "rrf":
            ranker = RRFRanker(k=rrf_k)
        else:
            ranker = WeightedRanker(*weights)

        all_results = []
        for col_name in collection_names:
            try:
                results = self.client.hybrid_search(
                    reqs=reqs,
                    rerank=ranker,
                    limit=top_k,
                    output_fields=output_fields,
                    collection_name=col_name,
                )
            except Exception as e:
                if "not found" in str(e).lower() or "sparse" in str(e).lower():
                    raise RuntimeError(
                        "Hybrid search requires BM25 support. "
                        "Re-create the collection with BM25 fields enabled."
                    ) from e
                raise
            all_results.extend(
                SearchResult.from_hit(hit, profile=self.profile, collection_name=col_name)
                for hit in results[0]
            )

        return all_results

    def query(
        self,
        filter_expr: str,
        limit: int = 100,
        offset: int = 0,
        output_fields: List[str] = None,
        collection_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if collection_names is None:
            collection_names = [self.client.collection_name]
        self._ensure_loaded(collection_names)

        if output_fields is None:
            output_fields = ["*"]

        all_results = []
        for col_name in collection_names:
            batch = self.client.query(
                expr=filter_expr,
                offset=offset,
                limit=limit,
                output_fields=output_fields,
                collection_name=col_name,
            )
            for row in batch:
                row["collection_name"] = col_name
            all_results.extend(batch)

        return all_results

    def count(self, filter_expr: Optional[str] = None) -> int:
        pk = self.profile.primary_key_field.name
        expr = filter_expr or f'{pk} != ""'
        page_size = 16384
        total = 0
        offset = 0
        while True:
            batch = self.client.query(expr=expr, limit=page_size, offset=offset, output_fields=[pk])
            total += len(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return total

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def delete_by_chunk_ids(self, chunk_ids: List[str], collection_name: Optional[str] = None) -> int:
        if not chunk_ids:
            return 0
        pk = self.profile.primary_key_field.name
        ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
        expr = f'{pk} in [{ids_str}]'
        result = self.client.delete(expr, collection_name=collection_name)
        self._flush()
        deleted = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info("Deleted %d chunks by chunk_ids", deleted)
        return deleted

    def delete_by_source(self, source_file: str, collection_name: Optional[str] = None) -> int:
        expr = f'source_file == "{source_file}"'
        result = self.client.delete(expr, collection_name=collection_name)
        self._flush()
        deleted = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info("Deleted %d chunks for source: %s", deleted, source_file)
        return deleted

    def delete_by_field(self, field_name: str, field_value: Any, collection_name: Optional[str] = None) -> int:
        """按字段名匹配值删除"""
        if isinstance(field_value, str):
            expr = f'{field_name} == "{field_value}"'
        else:
            expr = f'{field_name} == {field_value}'
        result = self.client.delete(expr, collection_name=collection_name)
        self._flush()
        deleted = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info("Deleted %d rows by %s == %s", deleted, field_name, field_value)
        return deleted

    def truncate_collection(self, collection_name: Optional[str] = None) -> int:
        """清空集合数据（保留 schema）"""
        return self.client.truncate_collection(collection_name=collection_name)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _flush(self):
        self.client.flush()
