import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

from llm.client import LLMClient
from llm.prompts import RAG_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT_NO_CONTEXT
from retrieval.service import MilvusService, SearchResult
from common.logger import setup_logger

logger = setup_logger("llm.service")


@dataclass
class ChatMessage:
    role: str
    content: str
    sources: List[Dict] = field(default_factory=list)


@dataclass
class Conversation:
    id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    title: Optional[str] = None


class RAGChatService:
    """RAG-powered chat service: retrieve + generate with streaming."""

    def __init__(self, llm_client: LLMClient, milvus_service: MilvusService):
        self.llm = llm_client
        self.retriever = milvus_service
        self._conversations: Dict[str, Conversation] = {}

    def create_conversation(self) -> Conversation:
        conv = Conversation(id=str(uuid.uuid4()))
        self._conversations[conv.id] = conv
        return conv

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)

    def list_conversations(self) -> List[Conversation]:
        return sorted(
            self._conversations.values(), key=lambda c: c.created_at, reverse=True
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        return self._conversations.pop(conversation_id, None) is not None

    def _retrieve(
        self,
        query: str,
        top_k: int = 5,
        collection_names: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        try:
            return self.retriever.hybrid_search(
                query=query,
                top_k=top_k,
                collection_names=collection_names,
            )
        except Exception as e:
            logger.warning("Hybrid search failed, falling back to vector-only search: %s", e)
            return self.retriever.hybrid_search(
                query=query,
                top_k=top_k,
                collection_names=collection_names,
                bm25_fields=[],
            )

    def _format_sources(self, results: List[SearchResult]) -> List[Dict]:
        sources = []
        for i, r in enumerate(results):
            sources.append({
                "index": i + 1,
                "chunk_id": r.chunk_id,
                "source_file": r.source_file,
                "score": round(r.score, 4),
                "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
            })
        return sources

    def _format_contexts(self, results: List[SearchResult]) -> str:
        return "\n\n".join(f"[{i + 1}] {r.content}" for i, r in enumerate(results))

    def _build_messages(
        self,
        conversation: Conversation,
        query: str,
        contexts: str,
    ) -> List[Dict[str, str]]:
        system_content = (
            RAG_SYSTEM_PROMPT.format(contexts=contexts)
            if contexts
            else RAG_SYSTEM_PROMPT_NO_CONTEXT
        )
        messages = [{"role": "system", "content": system_content}]
        for msg in conversation.messages:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": query})
        return messages

    async def chat(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        collection_names: Optional[List[str]] = None,
    ) -> Dict:
        if conversation_id and conversation_id in self._conversations:
            conv = self._conversations[conversation_id]
        else:
            conv = self.create_conversation()

        results = await asyncio.to_thread(
            self._retrieve, query, top_k, collection_names
        )
        sources = self._format_sources(results)
        contexts = self._format_contexts(results)

        conv.messages.append(ChatMessage(role="user", content=query))

        messages = self._build_messages(conv, query, contexts)
        answer = await self.llm.chat_completion(messages)

        conv.messages.append(ChatMessage(role="assistant", content=answer, sources=sources))

        return {"answer": answer, "sources": sources, "conversation_id": conv.id}

    async def chat_stream(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
        collection_names: Optional[List[str]] = None,
    ) -> AsyncIterator[Dict]:
        if conversation_id and conversation_id in self._conversations:
            conv = self._conversations[conversation_id]
        else:
            conv = self.create_conversation()

        results = await asyncio.to_thread(
            self._retrieve, query, top_k, collection_names
        )
        sources = self._format_sources(results)
        contexts = self._format_contexts(results)

        conv.messages.append(ChatMessage(role="user", content=query))

        yield {"event": "sources", "data": {"sources": sources, "conversation_id": conv.id}}

        messages = self._build_messages(conv, query, contexts)
        full_answer = ""
        async for delta in self.llm.chat_completion_stream(messages):
            full_answer += delta
            yield {"event": "delta", "data": {"content": delta}}

        conv.messages.append(
            ChatMessage(role="assistant", content=full_answer, sources=sources)
        )

        yield {"event": "done", "data": {"conversation_id": conv.id}}
