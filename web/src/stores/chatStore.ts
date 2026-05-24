import { create } from 'zustand';
import type { ChatMessage, SourceCitation } from '../api/types';

const STORAGE_KEY = 'rag-chat-state';

function uuid(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

interface ConversationState {
  title?: string;
  messages: ChatMessage[];
}

interface PersistedState {
  conversations: Record<string, ConversationState>;
  activeConversationId: string | null;
}

function loadPersisted(): PersistedState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return { conversations: {}, activeConversationId: null };
}

function persist(state: PersistedState) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch { /* ignore */ }
}

interface ChatState {
  conversations: Record<string, ConversationState>;
  activeConversationId: string | null;
  isStreaming: boolean;
  streamingContent: string;
  streamingSources: SourceCitation[];

  createConversation: () => string;
  setActiveConversation: (id: string) => void;
  addUserMessage: (conversationId: string, content: string) => void;
  addAssistantMessage: (conversationId: string, content: string, sources: SourceCitation[]) => void;
  appendStreamingContent: (delta: string) => void;
  setStreamingSources: (sources: SourceCitation[]) => void;
  startStreaming: () => void;
  finishStreaming: (conversationId: string, localId?: string) => void;
  deleteConversation: (id: string) => void;
  clearAll: () => void;
}

const initial = loadPersisted();

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: initial.conversations,
  activeConversationId: initial.activeConversationId,
  isStreaming: false,
  streamingContent: '',
  streamingSources: [],

  createConversation: () => {
    const id = uuid();
    set((state) => {
      const next = {
        conversations: { ...state.conversations, [id]: { messages: [] } },
        activeConversationId: id,
      };
      persist(next);
      return next;
    });
    return id;
  },

  setActiveConversation: (id) => {
    set({ activeConversationId: id });
    persist({ conversations: get().conversations, activeConversationId: id });
  },

  addUserMessage: (conversationId, content) => {
    set((state) => {
      const next = {
        conversations: {
          ...state.conversations,
          [conversationId]: {
            ...state.conversations[conversationId],
            messages: [
              ...state.conversations[conversationId].messages,
              { role: 'user', content },
            ],
          },
        },
      };
      persist({ ...next, activeConversationId: state.activeConversationId });
      return next;
    });
  },

  addAssistantMessage: (conversationId, content, sources) => {
    set((state) => {
      const next = {
        conversations: {
          ...state.conversations,
          [conversationId]: {
            ...state.conversations[conversationId],
            messages: [
              ...state.conversations[conversationId].messages,
              { role: 'assistant', content, sources },
            ],
          },
        },
      };
      persist({ ...next, activeConversationId: state.activeConversationId });
      return next;
    });
  },

  appendStreamingContent: (delta) => {
    set((state) => ({ streamingContent: state.streamingContent + delta }));
  },

  setStreamingSources: (sources) => {
    set({ streamingSources: sources });
  },

  startStreaming: () => set({ isStreaming: true, streamingContent: '', streamingSources: [] }),

  finishStreaming: (conversationId, localId) => {
    const { streamingContent, streamingSources, conversations, activeConversationId } = get();

    if (localId && localId !== conversationId) {
      const localConv = conversations[localId];
      const { [localId]: _, ...rest } = conversations;
      const next = {
        isStreaming: false,
        streamingContent: '',
        streamingSources: [],
        activeConversationId: conversationId,
        conversations: {
          ...rest,
          [conversationId]: {
            ...localConv,
            messages: [
              ...localConv.messages,
              { role: 'assistant', content: streamingContent, sources: streamingSources },
            ],
          },
        },
      };
      persist({ conversations: next.conversations, activeConversationId: next.activeConversationId });
      set(next);
    } else {
      set((state) => {
        const next = {
          isStreaming: false,
          streamingContent: '',
          streamingSources: [],
          conversations: {
            ...state.conversations,
            [conversationId]: {
              ...state.conversations[conversationId],
              messages: [
                ...state.conversations[conversationId].messages,
                { role: 'assistant', content: streamingContent, sources: streamingSources },
              ],
            },
          },
        };
        persist({ conversations: next.conversations, activeConversationId: state.activeConversationId });
        return next;
      });
    }
  },

  deleteConversation: (id) => {
    set((state) => {
      const { [id]: _, ...rest } = state.conversations;
      const next = {
        conversations: rest,
        activeConversationId: state.activeConversationId === id ? null : state.activeConversationId,
      };
      persist(next);
      return next;
    });
  },

  clearAll: () => {
    const next = { conversations: {}, activeConversationId: null };
    persist(next);
    set(next);
  },
}));
