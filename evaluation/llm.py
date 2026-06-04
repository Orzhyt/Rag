import os
from typing import Any, List

from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory

from retrieval.embedder import Embedder


class LocalEmbedding(BaseRagasEmbedding):
    """包装现有 EmbeddingModel，适配 ragas 0.4.x 的 BaseRagasEmbedding 接口"""

    def __init__(self, embedder: Embedder, cache: Any = None):
        super().__init__(cache=cache)
        self.embedder = embedder

    def embed_text(self, text: str, **kwargs) -> List[float]:
        return self.embedder.encode([text])[0]

    async def aembed_text(self, text: str, **kwargs) -> List[float]:
        return self.embed_text(text)

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self.embedder.encode(texts)

    async def aembed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self.embed_texts(texts)

    # ragas AnswerRelevancy 内部调用 embed_query() / embed_documents()，
    # 语义分别与 embed_text / embed_texts 相同
    def embed_query(self, text: str, **kwargs) -> List[float]:
        return self.embed_text(text, **kwargs)

    async def aembed_query(self, text: str, **kwargs) -> List[float]:
        return self.embed_text(text, **kwargs)

    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self.embed_texts(texts, **kwargs)

    async def aembed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self.embed_texts(texts, **kwargs)


def get_ragas_llm():
    """构建 ragas 评估用的 LLM（华为云 MaaS，使用 JUDGE_MODEL）"""
    api_url = os.getenv("MAAS_API_URL", "https://api.modelarts-maas.com/v2/chat/completions")
    base_url = api_url.rsplit("/chat/completions", 1)[0]

    client = AsyncOpenAI(
        api_key=os.getenv("MAAS_API_KEY"),
        base_url=base_url,
    )
    return llm_factory(
        model=os.getenv("JUDGE_MODEL", "DeepSeek-V3.2"),
        client=client,
        temperature=0,
        max_tokens=16384,
    )


def get_ragas_embeddings(embedder: Embedder) -> LocalEmbedding:
    """构建 ragas 评估用的 Embeddings，复用已加载的 EmbeddingModel"""
    return LocalEmbedding(embedder)
