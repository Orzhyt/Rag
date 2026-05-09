import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient as PyMilvusClient,
)
from pymilvus.milvus_client.index import IndexParams

from common.logger import setup_logger

load_dotenv()
logger = setup_logger("milvus.client")

# content 字段最大长度
MAX_CONTENT_LENGTH = 16384


class MilvusClient:
    """Milvus 连接管理与集合操作（基于 PyMilvus MilvusClient 新 API）"""

    def __init__(
        self,
        host: str = None,
        port: str = None,
        user: str = None,
        password: str = None,
        database: str = None,
        collection_name: str = "rag_chunks",
    ):
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.user = user or os.getenv("MILVUS_USER", "root")
        self.password = password or os.getenv("MILVUS_PASSWORD", "")
        self.database = database or os.getenv("MILVUS_DATABASE", "default")
        self.collection_name = collection_name
        self._client: PyMilvusClient = None
        self._connected = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self):
        """建立 Milvus 连接"""
        if self._connected:
            return

        uri = f"http://{self.host}:{self.port}"
        kwargs = {"uri": uri}
        if self.user:
            kwargs["user"] = self.user
        if self.password:
            kwargs["password"] = self.password
        if self.database and self.database != "default":
            kwargs["db_name"] = self.database

        logger.info(
            "Connecting to Milvus: %s:%s, database=%s",
            self.host, self.port, self.database,
        )
        self._client = PyMilvusClient(**kwargs)
        self._connected = True
        logger.info("Milvus connected successfully")

    def disconnect(self):
        """断开 Milvus 连接"""
        if self._connected:
            self._client.close()
            self._client = None
            self._connected = False
            logger.info("Milvus disconnected")

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # 数据库管理
    # ------------------------------------------------------------------

    def create_database(self, db_name: str):
        """创建数据库（已存在则跳过）"""
        existing = self._client.list_databases()
        if db_name in existing:
            logger.info("Database '%s' already exists, skipping", db_name)
            return
        self._client.create_database(db_name)
        logger.info("Database '%s' created", db_name)

    def drop_database(self, db_name: str):
        """删除数据库"""
        self._client.drop_database(db_name)
        logger.info("Database '%s' dropped", db_name)

    def list_databases(self) -> List[str]:
        """列举所有数据库"""
        return self._client.list_databases()

    def using_database(self, db_name: str):
        """切换当前数据库"""
        self._client.use_database(db_name)
        self.database = db_name
        logger.info("Switched to database '%s'", db_name)

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        return self._client.has_collection(self.collection_name)

    def has_collection(self, name: str) -> bool:
        """检查指定名称的集合是否存在"""
        return self._client.has_collection(name)

    def describe_collection(self, name: str = None) -> Dict[str, Any]:
        """获取集合的 schema 描述"""
        col_name = name or self.collection_name
        if not self.has_collection(col_name):
            raise RuntimeError(f"Collection {col_name} does not exist")
        info = self._client.describe_collection(col_name)
        fields_info = []
        for f in info.get("fields", []):
            field_type = f.get("type")
            info_item = {
                "name": f["name"],
                "type": field_type.name if hasattr(field_type, "name") else str(field_type),
                "is_primary": f.get("is_primary", False),
            }
            params = f.get("params", {})
            if field_type == DataType.VARCHAR:
                info_item["max_length"] = params.get("max_length")
            elif field_type == DataType.FLOAT_VECTOR:
                info_item["dim"] = params.get("dim")
            elif field_type == DataType.SPARSE_FLOAT_VECTOR:
                info_item["type_detail"] = "sparse_float_vector"
            fields_info.append(info_item)
        return {
            "name": col_name,
            "description": info.get("description", ""),
            "fields": fields_info,
        }

    @property
    def has_bm25_support(self) -> bool:
        """检查当前集合是否支持 BM25 全文检索"""
        if not self.collection_exists():
            return False
        info = self._client.describe_collection(self.collection_name)
        field_names = [f["name"] for f in info.get("fields", [])]
        return "content_sparse" in field_names

    def create_collection(
        self,
        dim: int,
        drop_if_exists: bool = False,
        enable_bm25: bool = True,
    ):
        """创建 Rag 集合

        Args:
            dim: 向量维度
            drop_if_exists: 是否先删除已存在的同名集合
            enable_bm25: 是否启用 BM25 全文检索（需 Milvus 2.5+）
        """
        if self.collection_exists():
            if drop_if_exists:
                logger.info("Dropping existing collection: %s", self.collection_name)
                self._client.drop_collection(self.collection_name)
            else:
                logger.info("Collection %s already exists, reusing", self.collection_name)
                return

        # content 字段：启用 BM25 时需要 analyzer
        content_field = FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=MAX_CONTENT_LENGTH,
            **({"enable_analyzer": True, "enable_match": True} if enable_bm25 else {}),
        )

        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            content_field,
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="total_chunks", dtype=DataType.INT64),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="sheet_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_size", dtype=DataType.INT64),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="modified_at", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        # BM25：添加 sparse vector 字段和 Function
        functions = []
        if enable_bm25:
            fields.append(
                FieldSchema(name="content_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR)
            )
            functions.append(
                Function(
                    name="content_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names="content",
                    output_field_names="content_sparse",
                )
            )

        schema = CollectionSchema(
            fields,
            functions=functions if functions else None,
            description="RAG knowledge base chunks",
        )
        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )

        # 创建 IVF_FLAT 索引以支持向量搜索
        index_params = IndexParams()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            nlist=128,
        )
        self._client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )

        # 创建 sparse index 以支持 BM25 全文检索
        if enable_bm25:
            sparse_index_params = IndexParams()
            sparse_index_params.add_index(
                field_name="content_sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
            self._client.create_index(
                collection_name=self.collection_name,
                index_params=sparse_index_params,
            )

        logger.info(
            "Collection %s created with dim=%d, index=IVF_FLAT/COSINE, bm25=%s",
            self.collection_name, dim, enable_bm25,
        )

    def drop_collection(self):
        """删除集合"""
        if self.collection_exists():
            self._client.drop_collection(self.collection_name)
            logger.info("Collection %s dropped", self.collection_name)

    def load_collection(self):
        """将集合加载到内存（搜索前必须调用）"""
        self._client.load_collection(self.collection_name)
        logger.info("Collection %s loaded into memory", self.collection_name)

    def release_collection(self):
        """释放集合内存"""
        self._client.release_collection(self.collection_name)
        logger.info("Collection %s released from memory", self.collection_name)

    # ------------------------------------------------------------------
    # 数据操作
    # ------------------------------------------------------------------

    def insert(self, data: List[Dict[str, Any]]):
        """插入数据"""
        return self._client.insert(collection_name=self.collection_name, data=data)

    def upsert(self, data: List[Dict[str, Any]]):
        """增量更新数据"""
        return self._client.upsert(collection_name=self.collection_name, data=data)

    def search(
        self,
        data: List[List[float]],
        anns_field: str,
        search_params: Dict[str, Any],
        limit: int,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        offset: int = 0,
    ):
        """向量搜索"""
        return self._client.search(
            collection_name=self.collection_name,
            data=data,
            anns_field=anns_field,
            search_params=search_params,
            limit=limit,
            filter=expr or "",
            output_fields=output_fields,
            offset=offset,
        )

    def hybrid_search(
        self,
        reqs: List,
        rerank,
        limit: int,
        output_fields: Optional[List[str]] = None,
    ):
        """混合搜索"""
        return self._client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
        )

    def query(
        self,
        expr: str,
        limit: int = 100,
        offset: int = 0,
        output_fields: Optional[List[str]] = None,
    ):
        """标量查询"""
        return self._client.query(
            collection_name=self.collection_name,
            filter=expr,
            limit=limit,
            offset=offset,
            output_fields=output_fields,
        )

    def delete(self, expr: str):
        """按表达式删除"""
        return self._client.delete(
            collection_name=self.collection_name,
            filter=expr,
        )

    def flush(self):
        """刷写集合数据"""
        self._client.flush(self.collection_name)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
