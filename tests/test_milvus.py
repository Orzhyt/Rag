import pytest
from data_pipeline.parser import ParsedChunk
from milvus import MilvusClient, MilvusService


# ============================================================================
# 连接测试
# ============================================================================


class TestConnection:
    def test_connect_and_disconnect(self):
        client = MilvusClient(
            host="localhost", port="19530",
            user="cjc", password="l$Y2TOC4Y%FJsPJd",
        )
        assert not client.connected
        client.connect()
        assert client.connected
        client.disconnect()
        assert not client.connected

    def test_context_manager(self):
        with MilvusClient(
            host="localhost", port="19530",
            user="cjc", password="l$Y2TOC4Y%FJsPJd",
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
