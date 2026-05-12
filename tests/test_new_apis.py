"""新增接口测试：delete_by_field, truncate_collection, truncate_database,
drop_collection with database, 以及已有接口的 database 参数支持。"""

import pytest
from data_pipeline.parser import ParsedChunk
from milvus import MilvusClient
from retrieval.service import MilvusService


TEST_DB = "test_api_db"
TEST_COL = "test_api_col"
TEST_COL_2 = "test_api_col_2"


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
        created_at="2026-05-11",
        modified_at="2026-05-11",
        metadata={},
    )


@pytest.fixture
def test_db_client(milvus_client):
    """创建临时测试数据库，测试后清理"""
    milvus_client.create_database(TEST_DB)
    yield milvus_client
    # 清理：切到测试数据库，删除所有集合，再删除数据库
    try:
        milvus_client.using_database(TEST_DB)
        for col in milvus_client.list_collections():
            milvus_client.drop_collection(collection_name=col)
        milvus_client.using_database("default")
        milvus_client.drop_database(TEST_DB)
    except Exception:
        pass


@pytest.fixture
def test_db_service(test_db_client, embedder):
    """在测试数据库下创建 MilvusService 并初始化集合"""
    test_db_client.using_database(TEST_DB)
    service = MilvusService(client=test_db_client, embedder=embedder)
    service.init_collection(drop_if_exists=True, enable_bm25=True)
    yield service
    try:
        test_db_client.release_collection()
    except Exception:
        pass


# ============================================================================
# 1. delete_by_field — 按字段名匹配值删除
# ============================================================================


class TestDeleteByField:
    def test_delete_by_varchar_field(self, milvus_service):
        chunks = [
            _make_chunk("df_001", "AI内容", file_name="ai.pdf"),
            _make_chunk("df_002", "Python内容", file_name="python.pdf"),
        ]
        milvus_service.insert(chunks)
        assert milvus_service.count() == 2

        deleted = milvus_service.delete_by_field("file_name", "ai.pdf")
        assert deleted == 1
        assert milvus_service.count() == 1

    def test_delete_by_int_field(self, milvus_service):
        chunks = [
            _make_chunk("di_001", "内容1"),
            _make_chunk("di_002", "内容2"),
        ]
        # 两个 chunk 的 page_number 都是 1
        milvus_service.insert(chunks)
        assert milvus_service.count() == 2

        deleted = milvus_service.delete_by_field("page_number", 1)
        assert deleted == 2
        assert milvus_service.count() == 0

    def test_delete_by_field_with_collection_name(self, milvus_service):
        col_name = milvus_service.client.collection_name
        chunks = [_make_chunk("dcn_001", "测试内容")]
        milvus_service.insert(chunks)
        assert milvus_service.count() == 1

        deleted = milvus_service.delete_by_field("chunk_id", "dcn_001", collection_name=col_name)
        assert deleted == 1
        assert milvus_service.count() == 0

    def test_delete_by_field_in_database(self, test_db_service):
        chunks = [
            _make_chunk("dfd_001", "DB内容1", file_name="a.pdf"),
            _make_chunk("dfd_002", "DB内容2", file_name="b.pdf"),
        ]
        test_db_service.insert(chunks)
        assert test_db_service.count() == 2

        deleted = test_db_service.delete_by_field("file_name", "a.pdf")
        assert deleted == 1
        assert test_db_service.count() == 1


# ============================================================================
# 2. truncate_collection — 清空集合数据
# ============================================================================


class TestTruncateCollection:
    def test_truncate_collection(self, milvus_service):
        chunks = [
            _make_chunk("tr_001", "内容1"),
            _make_chunk("tr_002", "内容2"),
        ]
        milvus_service.insert(chunks)
        assert milvus_service.count() == 2

        deleted = milvus_service.truncate_collection()
        assert deleted == 2
        assert milvus_service.count() == 0

    def test_truncate_preserves_schema(self, milvus_service):
        """清空后集合仍存在，schema 不变"""
        chunks = [_make_chunk("trs_001", "内容")]
        milvus_service.insert(chunks)

        milvus_service.truncate_collection()
        assert milvus_service.client.has_collection(milvus_service.client.collection_name)
        desc = milvus_service.client.describe_collection()
        assert len(desc["fields"]) > 0

    def test_truncate_with_collection_name(self, milvus_service):
        col_name = milvus_service.client.collection_name
        chunks = [_make_chunk("trn_001", "内容")]
        milvus_service.insert(chunks)
        assert milvus_service.count() == 1

        deleted = milvus_service.truncate_collection(collection_name=col_name)
        assert deleted == 1
        assert milvus_service.count() == 0

    def test_truncate_in_database(self, test_db_service):
        chunks = [_make_chunk("trd_001", "DB内容")]
        test_db_service.insert(chunks)
        assert test_db_service.count() == 1

        deleted = test_db_service.truncate_collection()
        assert deleted == 1
        assert test_db_service.count() == 0


# ============================================================================
# 3. list_collections — 列举集合
# ============================================================================


class TestListCollections:
    def test_list_collections(self, milvus_client):
        cols = milvus_client.list_collections()
        assert isinstance(cols, list)

    def test_list_collections_in_database(self, test_db_client, embedder):
        test_db_client.using_database(TEST_DB)
        # 初始为空或只有已有集合
        cols_before = test_db_client.list_collections()

        test_db_client.create_collection(dim=embedder.dim, drop_if_exists=True, collection_name=TEST_COL)
        cols_after = test_db_client.list_collections()
        assert TEST_COL in cols_after

        # 清理
        test_db_client.drop_collection(collection_name=TEST_COL)


# ============================================================================
# 4. drop_collection with database — 跨库删除集合
# ============================================================================


class TestDropCollectionWithDatabase:
    def test_drop_collection_in_database(self, test_db_client, embedder):
        test_db_client.using_database(TEST_DB)
        test_db_client.create_collection(dim=embedder.dim, drop_if_exists=True, collection_name=TEST_COL)
        assert test_db_client.has_collection(TEST_COL)

        # 切回 default，用 database 参数跨库删除
        test_db_client.using_database("default")
        test_db_client.drop_collection(collection_name=TEST_COL, database=TEST_DB)

        # 验证已删除
        test_db_client.using_database(TEST_DB)
        assert not test_db_client.has_collection(TEST_COL)


# ============================================================================
# 5. delete_by_chunk_ids / delete_by_source with collection_name
# ============================================================================


class TestDeleteWithCollectionName:
    def test_delete_by_chunk_ids_with_collection_name(self, milvus_service):
        col_name = milvus_service.client.collection_name
        chunks = [
            _make_chunk("dci_001", "内容1"),
            _make_chunk("dci_002", "内容2"),
        ]
        milvus_service.insert(chunks)
        assert milvus_service.count() == 2

        deleted = milvus_service.delete_by_chunk_ids(["dci_001"], collection_name=col_name)
        assert deleted == 1
        assert milvus_service.count() == 1

    def test_delete_by_source_with_collection_name(self, milvus_service):
        col_name = milvus_service.client.collection_name
        chunks = [
            _make_chunk("dsc_001", "内容1", source="/test/a.pdf"),
            _make_chunk("dsc_002", "内容2", source="/test/b.pdf"),
        ]
        milvus_service.insert(chunks)
        assert milvus_service.count() == 2

        deleted = milvus_service.delete_by_source("/test/a.pdf", collection_name=col_name)
        assert deleted == 1
        assert milvus_service.count() == 1


# ============================================================================
# 6. truncate_database — 清空数据库下所有集合
# ============================================================================


class TestTruncateDatabase:
    def test_truncate_database(self, test_db_client, embedder):
        test_db_client.using_database(TEST_DB)

        # 创建两个集合并插入数据
        test_db_client.create_collection(dim=embedder.dim, drop_if_exists=True, collection_name=TEST_COL)
        test_db_client.create_collection(dim=embedder.dim, drop_if_exists=True, collection_name=TEST_COL_2)

        service = MilvusService(client=test_db_client, embedder=embedder)
        # 插入到第一个集合
        service.client.collection_name = TEST_COL
        service.init_collection(enable_bm25=True)
        service.insert([_make_chunk("tdb_001", "DB内容1")])

        # 插入到第二个集合
        service.client.collection_name = TEST_COL_2
        service.init_collection(enable_bm25=True)
        service.insert([_make_chunk("tdb_002", "DB内容2")])

        # 通过 client 层清空整个数据库
        test_db_client.using_database(TEST_DB)
        collections = test_db_client.list_collections()
        total_deleted = 0
        for col in collections:
            total_deleted += test_db_client.truncate_collection(collection_name=col)

        assert total_deleted >= 2

        # 验证集合仍存在但数据为空
        assert test_db_client.has_collection(TEST_COL)
        assert test_db_client.has_collection(TEST_COL_2)
