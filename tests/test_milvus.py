import pytest
from pymilvus import DataType, FieldSchema
from data_pipeline.parser import ParsedChunk
from milvus import MilvusClient, MilvusService
from milvus.client import build_field_schema, build_default_rag_fields
from api.schemas import FieldDefinition


# ============================================================================
# 连接测试
# ============================================================================


class TestConnection:
    def test_connect_and_disconnect(self):
        client = MilvusClient(
            host="localhost", port="19530",
            user="root", password="",
        )
        assert not client.connected
        client.connect()
        assert client.connected
        client.disconnect()
        assert not client.connected

    def test_context_manager(self):
        with MilvusClient(
            host="localhost", port="19530",
            user="root", password="",
        ) as client:
            assert client.connected


# ============================================================================
# Database 管理
# ============================================================================


class TestDatabase:
    def test_list_databases(self, milvus_client):
        dbs = milvus_client.list_databases()
        assert isinstance(dbs, list)
        assert "default" in dbs

    def test_create_and_drop_database(self, milvus_client):
        test_db = "test_db_temp"
        # 创建
        milvus_client.create_database(test_db)
        dbs = milvus_client.list_databases()
        assert test_db in dbs

        # 重复创建不报错
        milvus_client.create_database(test_db)

        # 删除
        milvus_client.drop_database(test_db)
        dbs = milvus_client.list_databases()
        assert test_db not in dbs

    def test_using_database(self, milvus_client):
        test_db = "test_db_switch"
        milvus_client.create_database(test_db)
        milvus_client.using_database(test_db)
        assert milvus_client.database == test_db

        # 切回 default
        milvus_client.using_database("default")
        assert milvus_client.database == "default"

        milvus_client.drop_database(test_db)


# ============================================================================
# Collection / Schema 管理
# ============================================================================


class TestCollection:
    def test_collection_exists_false(self, milvus_client):
        assert not milvus_client.has_collection("non_existent_collection")

    def test_create_and_describe_collection(self, milvus_client, embedder):
        col = milvus_client.create_collection(dim=embedder.dim, drop_if_exists=True)
        assert milvus_client.collection_exists()

        desc = milvus_client.describe_collection()
        assert desc["name"] == milvus_client.collection_name
        assert len(desc["fields"]) > 0

        # 检查关键字段
        field_names = [f["name"] for f in desc["fields"]]
        assert "chunk_id" in field_names
        assert "embedding" in field_names
        assert "content" in field_names

        # 检查向量字段维度
        embedding_field = next(f for f in desc["fields"] if f["name"] == "embedding")
        assert embedding_field["dim"] == embedder.dim

    def test_has_collection(self, milvus_client, embedder):
        milvus_client.create_collection(dim=embedder.dim, drop_if_exists=True)
        assert milvus_client.has_collection(milvus_client.collection_name)

    def test_drop_collection(self, milvus_client, embedder):
        milvus_client.create_collection(dim=embedder.dim, drop_if_exists=True)
        assert milvus_client.collection_exists()
        milvus_client.drop_collection()
        assert not milvus_client.collection_exists()

    def test_load_and_release_collection(self, milvus_client, embedder):
        milvus_client.create_collection(dim=embedder.dim, drop_if_exists=True)
        milvus_client.load_collection()
        milvus_client.release_collection()


# ============================================================================
# 数据 CRUD
# ============================================================================


def _make_chunk(chunk_id, content, source="/test/test.pdf", file_name="test.pdf"):
    return ParsedChunk(
        chunk_id=chunk_id,
        content=content,
        source_file=source,
        file_name=file_name,
        file_type="pdf",
        chunk_index=0,
        total_chunks=1,
        page_number=1,
        sheet_name="",
        title="",
        file_size=1024,
        created_at="2026-05-08",
        modified_at="2026-05-08",
        metadata={},
    )


class TestInsert:
    def test_insert_and_count(self, milvus_service):
        chunks = [
            _make_chunk("ins_001", "人工智能是计算机科学的一个分支。"),
            _make_chunk("ins_002", "Python是数据科学领域常用的编程语言。"),
        ]
        count = milvus_service.insert(chunks)
        assert count == 2
        assert milvus_service.count() == 2

    def test_insert_empty(self, milvus_service):
        assert milvus_service.insert([]) == 0


class TestSearch:
    def test_search_returns_results(self, milvus_service):
        chunks = [
            _make_chunk("sea_001", "人工智能是计算机科学的一个分支，致力于创建智能系统。"),
            _make_chunk("sea_002", "Python是一种广泛使用的编程语言。"),
            _make_chunk("sea_003", "向量数据库用于存储和检索高维向量。"),
        ]
        milvus_service.insert(chunks)

        results = milvus_service.search("什么是人工智能", top_k=3)
        assert len(results) > 0
        # AI 相关内容应排在最前
        assert results[0].chunk_id == "sea_001"
        assert results[0].score > 0

    def test_search_with_filter(self, milvus_service):
        chunks = [
            _make_chunk("fil_001", "人工智能相关内容", source="/test/ai.pdf", file_name="ai.pdf"),
            _make_chunk("fil_002", "Python相关内容", source="/test/py.pdf", file_name="py.pdf"),
        ]
        milvus_service.insert(chunks)

        results = milvus_service.search("技术", top_k=10, filter_expr='file_name == "ai.pdf"')
        assert all(r.file_name == "ai.pdf" for r in results)


class TestQuery:
    def test_scalar_query(self, milvus_service):
        chunks = [
            _make_chunk("qry_001", "测试内容1"),
            _make_chunk("qry_002", "测试内容2"),
        ]
        milvus_service.insert(chunks)

        results = milvus_service.query('file_type == "pdf"', limit=10)
        assert len(results) >= 2


class TestUpsert:
    def test_upsert_updates_existing(self, milvus_service):
        chunk = _make_chunk("ups_001", "原始内容")
        milvus_service.insert([chunk])

        updated = _make_chunk("ups_001", "更新后的内容")
        upserted = milvus_service.upsert([updated])
        assert upserted == 1

        results = milvus_service.query('chunk_id == "ups_001"', limit=1, output_fields=["content"])
        assert len(results) == 1
        assert results[0]["content"] == "更新后的内容"


class TestDelete:
    def test_delete_by_chunk_ids(self, milvus_service):
        chunks = [
            _make_chunk("del_001", "内容1"),
            _make_chunk("del_002", "内容2"),
        ]
        milvus_service.insert(chunks)
        assert milvus_service.count() == 2

        deleted = milvus_service.delete_by_chunk_ids(["del_001"])
        assert deleted == 1
        assert milvus_service.count() == 1

    def test_delete_by_source(self, milvus_service):
        chunks = [
            _make_chunk("dsr_001", "内容1", source="/test/a.pdf"),
            _make_chunk("dsr_002", "内容2", source="/test/b.pdf"),
        ]
        milvus_service.insert(chunks)
        assert milvus_service.count() == 2

        deleted = milvus_service.delete_by_source("/test/a.pdf")
        assert deleted == 1
        assert milvus_service.count() == 1

    def test_delete_empty_ids(self, milvus_service):
        assert milvus_service.delete_by_chunk_ids([]) == 0


# ============================================================================
# Hybrid Search (向量 + BM25 + 标量过滤)
# ============================================================================


class TestHybridSearch:
    def test_hybrid_search_returns_results(self, milvus_service):
        chunks = [
            _make_chunk("hyb_001", "人工智能是计算机科学的一个分支，致力于创建智能系统。"),
            _make_chunk("hyb_002", "Python是一种广泛使用的编程语言，适用于数据科学。"),
            _make_chunk("hyb_003", "向量数据库用于存储和检索高维向量，支持相似度搜索。"),
        ]
        milvus_service.insert(chunks)

        results = milvus_service.hybrid_search("什么是人工智能", top_k=3)
        assert len(results) > 0
        assert results[0].score > 0

    def test_hybrid_search_with_filter(self, milvus_service):
        chunks = [
            _make_chunk("hfl_001", "人工智能相关内容", source="/test/ai.pdf", file_name="ai.pdf"),
            _make_chunk("hfl_002", "Python相关内容", source="/test/py.pdf", file_name="py.pdf"),
        ]
        milvus_service.insert(chunks)

        results = milvus_service.hybrid_search(
            "技术", top_k=10, filter_expr='file_name == "ai.pdf"'
        )
        assert all(r.file_name == "ai.pdf" for r in results)

    def test_hybrid_search_weighted_ranker(self, milvus_service):
        chunks = [
            _make_chunk("hw_001", "深度学习是机器学习的一个子领域。"),
            _make_chunk("hw_002", "自然语言处理使用深度学习技术。"),
        ]
        milvus_service.insert(chunks)

        results_vec = milvus_service.hybrid_search(
            "深度学习", top_k=2, vector_weight=0.9, bm25_weight=0.1
        )
        assert len(results_vec) > 0

        results_bm25 = milvus_service.hybrid_search(
            "深度学习", top_k=2, vector_weight=0.1, bm25_weight=0.9
        )
        assert len(results_bm25) > 0

    def test_hybrid_search_rrf_ranker(self, milvus_service):
        chunks = [
            _make_chunk("hrf_001", "分布式系统设计原则包括一致性和可用性。"),
            _make_chunk("hrf_002", "微服务架构是一种分布式系统架构风格。"),
        ]
        milvus_service.insert(chunks)

        results = milvus_service.hybrid_search(
            "分布式系统", top_k=2, reranker="rrf", rrf_k=60
        )
        assert len(results) > 0
        assert results[0].score > 0

    def test_hybrid_search_exact_keyword_match(self, milvus_service):
        """BM25 应提升精确关键词匹配的排名"""
        chunks = [
            _make_chunk("kw_001", "Milvus是一个开源的向量数据库。"),
            _make_chunk("kw_002", "数据库管理系统负责数据的存储和检索。"),
            _make_chunk("kw_003", "开源软件促进了技术社区的发展。"),
        ]
        milvus_service.insert(chunks)

        results = milvus_service.hybrid_search("向量数据库", top_k=3)
        assert len(results) > 0
        assert results[0].chunk_id == "kw_001"


class TestCollectionBM25:
    def test_create_collection_with_bm25(self, milvus_client, embedder):
        milvus_client.create_collection(
            dim=embedder.dim, drop_if_exists=True, enable_bm25=True
        )
        desc = milvus_client.describe_collection()
        field_names = [f["name"] for f in desc["fields"]]
        assert "content_sparse" in field_names

    def test_create_collection_without_bm25(self, milvus_client, embedder):
        milvus_client.create_collection(
            dim=embedder.dim, drop_if_exists=True, enable_bm25=False
        )
        desc = milvus_client.describe_collection()
        field_names = [f["name"] for f in desc["fields"]]
        assert "content_sparse" not in field_names


# ============================================================================
# 自定义 Collection 创建
# ============================================================================


class TestBuildFieldSchema:
    """build_field_schema 单元测试（不需要 Milvus 连接）"""

    def test_varchar_field(self):
        fd = FieldDefinition(name="title", dtype="VARCHAR", max_length=512)
        fs = build_field_schema(fd)
        assert fs.name == "title"
        assert fs.dtype == DataType.VARCHAR
        assert fs.params.get("max_length") == 512

    def test_varchar_default_max_length(self):
        fd = FieldDefinition(name="name", dtype="VARCHAR")
        fs = build_field_schema(fd)
        assert fs.params.get("max_length") == 256

    def test_float_vector_with_dim(self):
        fd = FieldDefinition(name="vec", dtype="FLOAT_VECTOR", dim=768)
        fs = build_field_schema(fd)
        assert fs.dtype == DataType.FLOAT_VECTOR
        assert fs.params.get("dim") == 768

    def test_float_vector_with_request_dim(self):
        fd = FieldDefinition(name="vec", dtype="FLOAT_VECTOR")
        fs = build_field_schema(fd, dim=1024)
        assert fs.params.get("dim") == 1024

    def test_float_vector_missing_dim_raises(self):
        fd = FieldDefinition(name="vec", dtype="FLOAT_VECTOR")
        with pytest.raises(ValueError, match="no dim provided"):
            build_field_schema(fd)

    def test_int64_field(self):
        fd = FieldDefinition(name="count", dtype="INT64")
        fs = build_field_schema(fd)
        assert fs.dtype == DataType.INT64

    def test_json_field(self):
        fd = FieldDefinition(name="meta", dtype="JSON")
        fs = build_field_schema(fd)
        assert fs.dtype == DataType.JSON

    def test_bool_field(self):
        fd = FieldDefinition(name="active", dtype="BOOL")
        fs = build_field_schema(fd)
        assert fs.dtype == DataType.BOOL

    def test_primary_key(self):
        fd = FieldDefinition(name="id", dtype="VARCHAR", max_length=128, is_primary=True)
        fs = build_field_schema(fd)
        assert fs.is_primary

    def test_bm25_varchar_field(self):
        fd = FieldDefinition(
            name="content", dtype="VARCHAR", max_length=8192,
            enable_analyzer=True, enable_match=True,
        )
        fs = build_field_schema(fd)
        assert fs.params.get("enable_analyzer") is True
        assert fs.params.get("enable_match") is True

    def test_array_field_missing_element_type(self):
        fd = FieldDefinition(name="tags", dtype="ARRAY")
        with pytest.raises(ValueError, match="element_type not provided"):
            build_field_schema(fd)

    def test_array_field(self):
        fd = FieldDefinition(name="tags", dtype="ARRAY", element_type="VARCHAR", max_capacity=10)
        fs = build_field_schema(fd)
        assert fs.dtype == DataType.ARRAY


class TestBuildDefaultRagFields:
    """build_default_rag_fields 单元测试"""

    def test_field_count(self):
        fields = build_default_rag_fields(dim=768, enable_bm25=False)
        assert len(fields) == 15

    def test_key_fields_present(self):
        fields = build_default_rag_fields(dim=768, enable_bm25=True)
        names = [f.name for f in fields]
        assert "chunk_id" in names
        assert "content" in names
        assert "embedding" in names
        assert "metadata" in names

    def test_primary_key(self):
        fields = build_default_rag_fields(dim=768, enable_bm25=False)
        pk = next(f for f in fields if f.is_primary)
        assert pk.name == "chunk_id"

    def test_embedding_dim(self):
        fields = build_default_rag_fields(dim=1024, enable_bm25=False)
        emb = next(f for f in fields if f.name == "embedding")
        assert emb.params.get("dim") == 1024

    def test_content_bm25_enabled(self):
        fields = build_default_rag_fields(dim=768, enable_bm25=True)
        content = next(f for f in fields if f.name == "content")
        assert content.params.get("enable_analyzer") is True
        assert content.params.get("enable_match") is True

    def test_content_bm25_disabled(self):
        fields = build_default_rag_fields(dim=768, enable_bm25=False)
        content = next(f for f in fields if f.name == "content")
        assert content.params.get("enable_analyzer") is None
        assert content.params.get("enable_match") is None


class TestCustomCollection:
    """自定义 Collection 创建集成测试（需要 Milvus）"""

    def test_custom_collection_name(self, milvus_client, embedder):
        custom_name = "test_custom_col_name"
        milvus_client.create_collection(
            dim=embedder.dim,
            drop_if_exists=True,
            collection_name=custom_name,
        )
        assert milvus_client.has_collection(custom_name)
        desc = milvus_client.describe_collection(custom_name)
        assert desc["name"] == custom_name
        # 清理
        milvus_client._client.drop_collection(custom_name)

    def test_custom_description(self, milvus_client, embedder):
        custom_name = "test_custom_desc"
        milvus_client.create_collection(
            dim=embedder.dim,
            drop_if_exists=True,
            collection_name=custom_name,
            description="自定义描述测试",
        )
        desc = milvus_client.describe_collection(custom_name)
        assert desc["description"] == "自定义描述测试"
        milvus_client._client.drop_collection(custom_name)

    def test_custom_fields_with_bm25(self, milvus_client, embedder):
        custom_name = "test_custom_fields_bm25"
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=8192, enable_analyzer=True, enable_match=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=embedder.dim),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(name="tags", dtype=DataType.JSON),
        ]
        milvus_client.create_collection(
            dim=embedder.dim,
            drop_if_exists=True,
            collection_name=custom_name,
            fields=fields,
            enable_bm25=True,
            bm25_config={
                "text_field_name": "desc",
                "sparse_field_name": "desc_sparse",
                "function_name": "desc_bm25",
            },
            embedding_field_name="vec",
            vector_index={
                "field_name": "vec",
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128},
            },
        )
        assert milvus_client.has_collection(custom_name)
        desc = milvus_client.describe_collection(custom_name)
        field_names = [f["name"] for f in desc["fields"]]
        assert "id" in field_names
        assert "desc" in field_names
        assert "vec" in field_names
        assert "desc_sparse" in field_names
        assert "category" in field_names
        assert "price" in field_names
        assert "tags" in field_names
        milvus_client._client.drop_collection(custom_name)

    def test_custom_fields_without_bm25(self, milvus_client, embedder):
        custom_name = "test_custom_fields_no_bm25"
        fields = [
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedder.dim),
        ]
        milvus_client.create_collection(
            dim=embedder.dim,
            drop_if_exists=True,
            collection_name=custom_name,
            fields=fields,
            enable_bm25=False,
        )
        desc = milvus_client.describe_collection(custom_name)
        field_names = [f["name"] for f in desc["fields"]]
        assert "doc_id" in field_names
        assert "text" in field_names
        assert "embedding" in field_names
        # 不应有 sparse 字段
        assert not any("sparse" in n for n in field_names)
        milvus_client._client.drop_collection(custom_name)

    def test_custom_vector_index_hnsw(self, milvus_client, embedder):
        custom_name = "test_custom_hnsw"
        milvus_client.create_collection(
            dim=embedder.dim,
            drop_if_exists=True,
            collection_name=custom_name,
            vector_index={
                "field_name": "embedding",
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 256},
            },
        )
        assert milvus_client.has_collection(custom_name)
        desc = milvus_client.describe_collection(custom_name)
        assert desc["name"] == custom_name
        milvus_client._client.drop_collection(custom_name)

    def test_bm25_text_field_not_found_raises(self, milvus_client, embedder):
        custom_name = "test_bm25_missing_text"
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=embedder.dim),
        ]
        with pytest.raises(ValueError, match="not found"):
            milvus_client.create_collection(
                dim=embedder.dim,
                drop_if_exists=True,
                collection_name=custom_name,
                fields=fields,
                enable_bm25=True,
                bm25_config={"text_field_name": "nonexistent"},
            )

    def test_bm25_duplicate_sparse_field_raises(self, milvus_client, embedder):
        custom_name = "test_bm25_dup_sparse"
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192, enable_analyzer=True, enable_match=True),
            FieldSchema(name="content_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=embedder.dim),
        ]
        with pytest.raises(ValueError, match="already exists"):
            milvus_client.create_collection(
                dim=embedder.dim,
                drop_if_exists=True,
                collection_name=custom_name,
                fields=fields,
                enable_bm25=True,
            )

    def test_backward_compatible_default(self, milvus_client, embedder):
        """不传新参数时行为与原来一致"""
        milvus_client.create_collection(dim=embedder.dim, drop_if_exists=True)
        desc = milvus_client.describe_collection()
        field_names = [f["name"] for f in desc["fields"]]
        assert "chunk_id" in field_names
        assert "content" in field_names
        assert "embedding" in field_names
        assert "content_sparse" in field_names
        assert "metadata" in field_names
