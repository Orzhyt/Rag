import apiClient from './client';
import type { ChatRequest, ChatResponse, Conversation, SourceCitation } from './types';

export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
  const { data } = await apiClient.post<ChatResponse>('/chat', req);
  return data;
}

export async function listConversations(): Promise<Conversation[]> {
  const { data } = await apiClient.get<Conversation[]>('/chat/conversations');
  return data;
}

export async function getConversation(id: string): Promise<Conversation> {
  const { data } = await apiClient.get<Conversation>(`/chat/conversations/${id}`);
  return data;
}

export async function deleteConversation(id: string): Promise<void> {
  await apiClient.delete(`/chat/conversations/${id}`);
}

export function streamChat(
  req: ChatRequest,
  onSources: (data: { sources: SourceCitation[]; conversation_id: string }) => void,
  onDelta: (content: string) => void,
  onDone: (data: { conversation_id: string }) => void,
  onError: (err: Error) => void,
): AbortController {
  const controller = new AbortController();

  fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      }
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentEvent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7);
          } else if (line.startsWith('data: ')) {
            try {
              const payload = JSON.parse(line.slice(6));
              if (currentEvent === 'sources') onSources(payload);
              else if (currentEvent === 'delta') onDelta(payload.content);
              else if (currentEvent === 'done') onDone(payload);
            } catch {
              // ignore parse errors for partial data
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError(err);
    });

  return controller;
}
