"""Integration tests for Huawei Cloud MaaS embedding and rerank APIs."""

import math
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

MAAS_API_KEY = os.getenv("MAAS_API_KEY")
skip_no_key = pytest.mark.skipif(not MAAS_API_KEY, reason="MAAS_API_KEY not set")


@skip_no_key
class TestMaaSEmbedding:
    def test_encode_single(self):
        from retrieval.embedder import Embedder
        embedder = Embedder(provider="maas")
        result = embedder.encode(["Hello world"])
        assert len(result) == 1
        assert len(result[0]) == embedder.dim
        assert all(isinstance(v, float) for v in result[0])

    def test_encode_batch(self):
        from retrieval.embedder import Embedder
        embedder = Embedder(provider="maas")
        texts = ["第一段文本", "第二段文本", "第三段文本"]
        result = embedder.encode(texts)
        assert len(result) == 3
        assert all(len(r) == embedder.dim for r in result)

    def test_dim_property(self):
        from retrieval.embedder import Embedder
        embedder = Embedder(provider="maas")
        assert embedder.dim == 1024

    def test_normalized(self):
        """验证向量已归一化（余弦相似度兼容）"""
        from retrieval.embedder import Embedder
        embedder = Embedder(provider="maas")
        result = embedder.encode(["测试文本"])
        norm = math.sqrt(sum(v * v for v in result[0]))
        assert abs(norm - 1.0) < 1e-4

    def test_chinese_text(self):
        from retrieval.embedder import Embedder
        embedder = Embedder(provider="maas")
        result = embedder.encode(["这是一只小猫", "这是一只小狗"])
        assert len(result) == 2
        assert len(result[0]) == embedder.dim
        assert len(result[1]) == embedder.dim


@skip_no_key
class TestMaaSReranker:
    def test_rerank_basic(self):
        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.rerank(
            query="什么是RAG？",
            documents=[
                "RAG是检索增强生成的缩写。",
                "今天天气晴朗。",
                "RAG结合了检索和语言模型。",
            ],
        )
        assert len(results) == 3
        assert results[0].relevance_score >= results[-1].relevance_score
        assert all(0 <= r.index < 3 for r in results)

    def test_rerank_top_n(self):
        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.rerank(
            query="如何冲泡咖啡？",
            documents=[
                "咖啡豆的产地主要分布在赤道附近。",
                "法压壶的步骤：研磨咖啡豆，加入热水，压下压杆。",
                "意式浓缩咖啡需要高压机器萃取。",
                "挑选咖啡豆时注意烘焙日期。",
                "手冲咖啡的关键是控制水流速度和水温。",
            ],
            top_n=2,
        )
        assert len(results) == 2

    def test_rerank_scores_valid(self):
        from retrieval.reranker import Reranker
        reranker = Reranker()
        results = reranker.rerank(
            query="测试查询",
            documents=["文档A", "文档B"],
        )
        for r in results:
            assert isinstance(r.relevance_score, float)
            assert isinstance(r.index, int)
