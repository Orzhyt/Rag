import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.deps import get_llm_service, get_milvus_service, get_milvus_client
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    SourceCitation,
)
from llm.service import RAGChatService
from milvus.client import MilvusClient
from common.logger import setup_logger

logger = setup_logger("api.chat")
router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    service: RAGChatService = Depends(get_llm_service),
    client: MilvusClient = Depends(get_milvus_client),
):
    if req.database and req.database != client.database:
        client.using_database(req.database)
    result = await service.chat(
        query=req.query,
        conversation_id=req.conversation_id,
        top_k=req.top_k,
        collection_names=req.collection_names,
    )
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        conversation_id=result["conversation_id"],
    )


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    service: RAGChatService = Depends(get_llm_service),
    client: MilvusClient = Depends(get_milvus_client),
):
    if req.database and req.database != client.database:
        client.using_database(req.database)

    async def event_generator():
        async for event in service.chat_stream(
            query=req.query,
            conversation_id=req.conversation_id,
            top_k=req.top_k,
            collection_names=req.collection_names,
        ):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    service: RAGChatService = Depends(get_llm_service),
):
    convs = service.list_conversations()
    return [
        ConversationResponse(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            message_count=len(c.messages),
        )
        for c in convs
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    service: RAGChatService = Depends(get_llm_service),
):
    conv = service.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = [
        {"role": m.role, "content": m.content, "sources": m.sources}
        for m in conv.messages
    ]
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        message_count=len(conv.messages),
        messages=messages,
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    service: RAGChatService = Depends(get_llm_service),
):
    deleted = service.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}
