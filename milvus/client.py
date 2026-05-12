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
from retrieval.profile import DEFAULT_RAG_PROFILE, DTYPE_MAP, FieldSpec, to_field_schema

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

        # 先用 default 数据库建立连接，再切换到目标数据库
        target_db = self.database
        logger.info(
            "Connecting to Milvus: %s:%s, database=%s",
            self.host, self.port, target_db,
        )
        self._client = PyMilvusClient(**kwargs)
        self._connected = True

        # 连接后切换到目标数据库（如不存在则回退到 default）
        if target_db and target_db != "default":
            try:
                existing = self._client.list_databases()
                if target_db in existing:
                    self._client.use_database(target_db)
                    logger.info("Switched to database '%s'", target_db)
                else:
                    logger.warning(
                        "Database '%s' does not exist, falling back to 'default'. Available: %s",
                        target_db, existing,
                    )
                    self.database = "default"
            except Exception as e:
                logger.warning("Failed to switch to database '%s': %s, using 'default'", target_db, e)
                self.database = "default"

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
        existing = self._client.list_databases()
        if db_name not in existing:
            raise ValueError(f"Database '{db_name}' does not exist. Available: {existing}")
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
        """检查当前集合是否支持 BM25 全文检索（检测任意 SPARSE_FLOAT_VECTOR 字段）"""
        if not self.collection_exists():
            return False
        info = self._client.describe_collection(self.collection_name)
        for f in info.get("fields", []):
            if f.get("type") == DataType.SPARSE_FLOAT_VECTOR:
                return True
        return False

    def create_collection(
        self,
        dim: int,
        drop_if_exists: bool = False,
        collection_name: Optional[str] = None,
        fields: Optional[List[FieldSpec]] = None,
        vector_index: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ):
        """创建集合

        Args:
            dim: 向量维度
            drop_if_exists: 是否先删除已存在的同名集合
            collection_name: 自定义集合名称，None 则使用 self.collection_name
            fields: 自定义字段列表（List[FieldSpec]），None 则使用默认 RAG 字段
            vector_index: 向量索引参数 dict，键: field_name, index_type, metric_type, params
            description: 集合 schema 描述
        """
        col_name = collection_name or self.collection_name

        if self._client.has_collection(col_name):
            if drop_if_exists:
                logger.info("Dropping existing collection: %s", col_name)
                self._client.drop_collection(col_name)
            else:
                logger.info("Collection %s already exists, reusing", col_name)
                return

        # 构建字段列表
        use_default_fields = fields is None
        if use_default_fields:
            field_specs = DEFAULT_RAG_PROFILE.fields
        else:
            field_specs = list(fields)

        # FieldSpec → FieldSchema
        field_list = [to_field_schema(f, dim=dim) for f in field_specs]

        # BM25 处理：从 FieldSpec 的 enable_bm25 自动推导
        functions = []
        bm25_sparse_fields = []  # 收集 (sparse_field_name, text_field_name)
        for fspec in field_specs:
            if fspec.dtype == "VARCHAR" and fspec.enable_bm25:
                sparse_name = f"{fspec.name}_sparse"
                func_name = f"{fspec.name}_bm25"
                field_list.append(
                    FieldSchema(name=sparse_name, dtype=DataType.SPARSE_FLOAT_VECTOR)
                )
                functions.append(
                    Function(
                        name=func_name,
                        function_type=FunctionType.BM25,
                        input_field_names=fspec.name,
                        output_field_names=sparse_name,
                    )
                )
                bm25_sparse_fields.append(sparse_name)

        schema = CollectionSchema(
            field_list,
            functions=functions if functions else None,
            description=description or "RAG knowledge base chunks",
        )
        self._client.create_collection(collection_name=col_name, schema=schema)

        # 创建向量索引（支持单个 dict 或 list[dict]）
        if vector_index is None:
            # 默认：为所有 FLOAT_VECTOR 字段建 IVF_FLAT 索引
            index_list = [
                {"field_name": f.name, "index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
                for f in field_list if f.dtype == DataType.FLOAT_VECTOR
            ]
        elif isinstance(vector_index, dict):
            index_list = [vector_index]
        else:
            index_list = vector_index

        index_params = IndexParams()
        for vi in index_list:
            index_params.add_index(
                field_name=vi.get("field_name", "embedding"),
                index_type=vi.get("index_type", "IVF_FLAT"),
                metric_type=vi.get("metric_type", "COSINE"),
                **vi.get("params", {"nlist": 128}),
            )
        self._client.create_index(collection_name=col_name, index_params=index_params)

        # BM25 sparse 索引
        for sparse_name in bm25_sparse_fields:
            sparse_index_params = IndexParams()
            sparse_index_params.add_index(
                field_name=sparse_name,
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
            self._client.create_index(collection_name=col_name, index_params=sparse_index_params)

        logger.info(
            "Collection %s created with dim=%d, vector_indexes=%d, bm25_fields=%d",
            col_name, dim, len(index_list), len(bm25_sparse_fields),
        )

    def drop_collection(self, collection_name: Optional[str] = None, database: Optional[str] = None):
        """删除集合

        Args:
            collection_name: 集合名称，None 则使用 self.collection_name
            database: 指定数据库，None 则使用当前数据库
        """
        col_name = collection_name or self.collection_name
        original_db = None
        if database:
            original_db = self.database
            self.using_database(database)
        try:
            if self.has_collection(col_name):
                self._client.drop_collection(col_name)
                logger.info("Collection %s dropped (database=%s)", col_name, database or self.database)
            else:
                logger.info("Collection %s does not exist, skip dropping", col_name)
        finally:
            if original_db is not None:
                self.using_database(original_db)

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
        collection_name: Optional[str] = None,
    ):
        """向量搜索"""
        return self._client.search(
            collection_name=collection_name or self.collection_name,
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
        collection_name: Optional[str] = None,
    ):
        """混合搜索"""
        return self._client.hybrid_search(
            collection_name=collection_name or self.collection_name,
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
        collection_name: Optional[str] = None,
    ):
        """标量查询"""
        return self._client.query(
            collection_name=collection_name or self.collection_name,
            filter=expr,
            limit=limit,
            offset=offset,
            output_fields=output_fields,
        )

    def delete(self, expr: str, collection_name: Optional[str] = None):
        """按表达式删除"""
        return self._client.delete(
            collection_name=collection_name or self.collection_name,
            filter=expr,
        )

    def list_collections(self) -> List[str]:
        """列举当前数据库下所有集合"""
        return self._client.list_collections()

    def truncate_collection(self, collection_name: Optional[str] = None):
        """清空集合数据（保留 schema）"""
        col_name = collection_name or self.collection_name
        if not self.has_collection(col_name):
            raise ValueError(f"Collection '{col_name}' does not exist")
        # 确保集合已加载
        try:
            self._client.load_collection(col_name)
        except Exception:
            pass
        # 获取主键字段及其类型
        info = self._client.describe_collection(col_name)
        pk_field = None
        pk_type = None
        for f in info.get("fields", []):
            if f.get("is_primary", False):
                pk_field = f["name"]
                pk_type = f.get("type", "")
                break
        if pk_field is None:
            raise ValueError(f"Cannot find primary key field in collection '{col_name}'")
        # 根据主键类型生成删除过滤表达式
        if pk_type == DataType.INT64:
            filter_expr = f"{pk_field} >= 0"
        else:
            filter_expr = f'{pk_field} != ""'
        # 按主键删除全部数据
        result = self._client.delete(
            collection_name=col_name,
            filter=filter_expr,
        )
        self._client.flush(col_name)
        deleted = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info("Collection '%s' truncated, deleted %d rows", col_name, deleted)
        return deleted

    def flush(self):
        """刷写集合数据"""
        self._client.flush(self.collection_name)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
