import os
from typing import AsyncIterator, List, Dict, Optional

from openai import AsyncOpenAI

from common.logger import setup_logger

logger = setup_logger("llm.client")


class LLMClient:
    """OpenAI-compatible LLM client with streaming support."""

    def __init__(self):
        api_url = os.getenv(
            "MAAS_API_URL",
            "https://api.modelarts-maas.com/v2/chat/completions",
        )
        base_url = api_url.rsplit("/chat/completions", 1)[0]
        self.client = AsyncOpenAI(
            api_key=os.getenv("MAAS_API_KEY"),
            base_url=base_url,
        )
        self.model = os.getenv("MAAS_MODEL", "glm-5.1")
        logger.info("LLM client initialized, model=%s, base_url=%s", self.model, base_url)

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
