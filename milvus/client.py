import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    connections,
    db,
    utility,
)

from common.logger import setup_logger

load_dotenv()
logger = setup_logger("milvus.client")

# content 字段最大长度
MAX_CONTENT_LENGTH = 16384


class MilvusClient:
    """Milvus 连接管理与集合操作"""

    def __init__(
        self,
        host: str = None,
        port: str = None,
        user: str = None,
        password: str = None,
        database: str = None,
        collection_name: str = "rag_chunks",
        alias: str = "default",
    ):
        self.host = host or os.getenv("MILVUS_HOST", "localhost")
        self.port = port or os.getenv("MILVUS_PORT", "19530")
        self.user = user or os.getenv("MILVUS_USER", "root")
        self.password = password or os.getenv("MILVUS_PASSWORD", "")
        self.database = database or os.getenv("MILVUS_DATABASE", "default")
        self.collection_name = collection_name
        self.alias = alias
        self._connected = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self):
        """建立 Milvus 连接"""
        if self._connected:
            return

        conn_kwargs = {
            "alias": self.alias,
            "host": self.host,
            "port": self.port,
        }
        if self.user:
            conn_kwargs["user"] = self.user
        if self.password:
            conn_kwargs["password"] = self.password
        if self.database and self.database != "default":
            conn_kwargs["db_name"] = self.database

        logger.info(
            "Connecting to Milvus: %s:%s, database=%s",
            self.host, self.port, self.database,
        )
        connections.connect(**conn_kwargs)

        self._connected = True
        logger.info("Milvus connected successfully")

    def disconnect(self):
        """断开 Milvus 连接"""
        if self._connected:
            connections.disconnect(self.alias)
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
        existing = db.list_database(using=self.alias)
        if db_name in existing:
            logger.info("Database '%s' already exists, skipping", db_name)
            return
        db.create_database(db_name, using=self.alias)
        logger.info("Database '%s' created", db_name)

    def drop_database(self, db_name: str):
        """删除数据库"""
        db.drop_database(db_name, using=self.alias)
        logger.info("Database '%s' dropped", db_name)

    def list_databases(self) -> List[str]:
        """列举所有数据库"""
        return db.list_database(using=self.alias)

    def using_database(self, db_name: str):
        """切换当前数据库"""
        db.using_database(db_name, using=self.alias)
        self.database = db_name
        logger.info("Switched to database '%s'", db_name)

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        return utility.has_collection(self.collection_name, using=self.alias)

    def has_collection(self, name: str) -> bool:
        """检查指定名称的集合是否存在"""
        return utility.has_collection(name, using=self.alias)

    def describe_collection(self, name: str = None) -> Dict[str, any]:
        """获取集合的 schema 描述

        Args:
            name: 集合名称，默认使用 self.collection_name
        """
        col_name = name or self.collection_name
        if not self.has_collection(col_name):
            raise RuntimeError(f"Collection {col_name} does not exist")
        col = Collection(name=col_name, using=self.alias)
        schema = col.schema
        fields_info = []
        for f in schema.fields:
            info = {
                "name": f.name,
                "type": f.dtype.name,
                "is_primary": f.is_primary,
            }
            if f.dtype == DataType.VARCHAR:
                info["max_length"] = f.params.get("max_length")
            elif f.dtype == DataType.FLOAT_VECTOR:
                info["dim"] = f.params.get("dim")
            elif f.dtype == DataType.SPARSE_FLOAT_VECTOR:
                info["type_detail"] = "sparse_float_vector"
            fields_info.append(info)
        return {
            "name": col_name,
            "description": schema.description,
            "fields": fields_info,
        }

    @property
    def has_bm25_support(self) -> bool:
        """检查当前集合是否支持 BM25 全文检索"""
        if not self.collection_exists():
            return False
        col = Collection(name=self.collection_name, using=self.alias)
        field_names = [f.name for f in col.schema.fields]
        return "content_sparse" in field_names

    def create_collection(
        self,
        dim: int,
        drop_if_exists: bool = False,
        enable_bm25: bool = True,
    ) -> Collection:
        """创建 Rag 集合

        Args:
            dim: 向量维度
            drop_if_exists: 是否先删除已存在的同名集合
            enable_bm25: 是否启用 BM25 全文检索（需 Milvus 2.5+）
        """
        if self.collection_exists():
            if drop_if_exists:
                logger.info("Dropping existing collection: %s", self.collection_name)
                utility.drop_collection(self.collection_name, using=self.alias)
            else:
                logger.info("Collection %s already exists, reusing", self.collection_name)
                return self.get_collection()

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
        col = Collection(name=self.collection_name, schema=schema, using=self.alias)

        # 创建 IVF_FLAT 索引以支持向量搜索
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        col.create_index(field_name="embedding", index_params=index_params)

        # 创建 sparse index 以支持 BM25 全文检索
        if enable_bm25:
            sparse_index_params = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
            }
            col.create_index(field_name="content_sparse", index_params=sparse_index_params)

        logger.info(
            "Collection %s created with dim=%d, index=IVF_FLAT/COSINE, bm25=%s",
            self.collection_name, dim, enable_bm25,
        )
        return col

    def get_collection(self) -> Collection:
        """获取集合对象"""
        if not self.collection_exists():
            raise RuntimeError(f"Collection {self.collection_name} does not exist")
        return Collection(name=self.collection_name, using=self.alias)

    def drop_collection(self):
        """删除集合"""
        if self.collection_exists():
            utility.drop_collection(self.collection_name, using=self.alias)
            logger.info("Collection %s dropped", self.collection_name)

    def load_collection(self):
        """将集合加载到内存（搜索前必须调用）"""
        col = self.get_collection()
        col.load()
        logger.info("Collection %s loaded into memory", self.collection_name)

    def release_collection(self):
        """释放集合内存"""
        col = self.get_collection()
        col.release()
        logger.info("Collection %s released from memory", self.collection_name)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
