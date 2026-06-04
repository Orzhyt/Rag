import os
from typing import List, Optional, Tuple

from openai import OpenAI

from retrieval.service import MilvusService


RAG_PROMPT_TEMPLATE = """基于以下参考资料回答问题，如果资料中没有相关信息，请说明无法回答。

参考资料：
{contexts}

问题：{query}
"""


class RAGPipeline:
    """端到端 RAG 流程封装，供 ragas 评估调用"""

    def __init__(self, milvus_service: MilvusService):
        self.retriever = milvus_service
        api_url = os.getenv("MAAS_API_URL", "https://api.modelarts-maas.com/v2/chat/completions")
        base_url = api_url.rsplit("/chat/completions", 1)[0]
        self.llm_client = OpenAI(
            api_key=os.getenv("MAAS_API_KEY"),
            base_url=base_url,
        )
        self.model = os.getenv("MAAS_MODEL", "glm-5.1")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        collection_names: Optional[List[str]] = None,
    ) -> List[str]:
        """检索相关文档片段，返回 content 列表

        mode="hybrid" 时启用 ANN + BM25 混合检索（BM25 权重 0.5）；
        mode="vector" 时仅使用 ANN 向量检索。
        """
        anns_fields = [{"field": "content", "weight": 1.0}]
        if mode == "hybrid":
            bm25_fields = [{"field": "content", "weight": 0.5}]
        else:
            bm25_fields = []
        results = self.retriever.hybrid_search(
            query=query,
            top_k=top_k,
            collection_names=collection_names,
            anns_fields=anns_fields,
            bm25_fields=bm25_fields,
        )
        return [r.content for r in results]

    def generate(self, query: str, contexts: List[str]) -> str:
        """基于检索结果生成答案"""
        context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = RAG_PROMPT_TEMPLATE.format(contexts=context_text, query=query)

        resp = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content

    def invoke(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        collection_names: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """检索 + 生成，返回 (answer, contexts)"""
        contexts = self.retrieve(query, top_k=top_k, mode=mode, collection_names=collection_names)
        answer = self.generate(query, contexts)
        return answer, contexts
