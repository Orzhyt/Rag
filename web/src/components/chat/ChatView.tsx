import { useRef, useEffect, useCallback, useState } from 'react';
import { Typography, Empty, Select, Space, Spin, message } from 'antd';
import ChatInput from './ChatInput';
import ChatMessage from './ChatMessage';
import ConversationList from './ConversationList';
import { useChatStore } from '../../stores/chatStore';
import { useAppStore } from '../../stores/appStore';
import { streamChat } from '../../api/chat';
import { listDatabases } from '../../api/databases';
import { listCollections } from '../../api/collections';

const { Title } = Typography;

export default function ChatView() {
  const {
    conversations,
    activeConversationId,
    isStreaming,
    streamingContent,
    streamingSources,
    createConversation,
    setActiveConversation,
    addUserMessage,
    startStreaming,
    appendStreamingContent,
    setStreamingSources,
    finishStreaming,
  } = useChatStore();

  const {
    selectedDatabase,
    selectedCollections,
    setSelectedDatabase,
    setSelectedCollections,
  } = useAppStore();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [databases, setDatabases] = useState<string[]>([]);
  const [collections, setCollections] = useState<string[]>([]);
  const [dbLoading, setDbLoading] = useState(false);
  const [colLoading, setColLoading] = useState(false);

  const activeConv = activeConversationId ? conversations[activeConversationId] : null;

  // Fetch databases on mount
  useEffect(() => {
    setDbLoading(true);
    listDatabases()
      .then((dbs) => {
        setDatabases(dbs);
        if (dbs.length > 0 && !selectedDatabase) {
          setSelectedDatabase(dbs[0]);
        }
      })
      .catch(() => message.error('获取数据库列表失败'))
      .finally(() => setDbLoading(false));
  }, []);

  // Fetch collections when database changes
  useEffect(() => {
    if (!selectedDatabase) {
      setCollections([]);
      return;
    }
    setColLoading(true);
    listCollections(selectedDatabase)
      .then((cols) => {
        setCollections(cols);
        // If cached selections are stale, reset to all
        const valid = selectedCollections.filter((c) => cols.includes(c));
        if (valid.length === 0 && cols.length > 0) {
          setSelectedCollections(cols);
        } else if (valid.length !== selectedCollections.length) {
          setSelectedCollections(valid.length > 0 ? valid : cols);
        }
      })
      .catch(() => message.error('获取集合列表失败'))
      .finally(() => setColLoading(false));
  }, [selectedDatabase]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeConv?.messages, streamingContent]);

  const handleSend = useCallback(
    (query: string, topK: number) => {
      const isNewConv = !activeConversationId;
      let convId = activeConversationId;
      if (!convId) {
        convId = createConversation();
      }

      addUserMessage(convId, query);
      startStreaming();

      abortRef.current = streamChat(
        {
          query,
          conversation_id: isNewConv ? undefined : convId,
          top_k: topK,
          database: selectedDatabase ?? undefined,
          collection_names: selectedCollections.length > 0 ? selectedCollections : undefined,
        },
        (data) => {
          // Don't switch activeConversationId during streaming for new conversations
          // — the local ID is the key in the store. Migration happens in finishStreaming.
          if (data.conversation_id ===convId) {
            setActiveConversation(data.conversation_id);
          }
          setStreamingSources(data.sources);
        },
        (delta) => {
          appendStreamingContent(delta);
        },
        (data) => {
          finishStreaming(data.conversation_id, convId!);
        },
        (err) => {
          console.error('Stream error:', err);
          finishStreaming(convId!, convId!);
        },
      );
    },
    [
      activeConversationId, createConversation, addUserMessage,
      startStreaming, appendStreamingContent, setStreamingSources,
      finishStreaming, setActiveConversation,
      selectedDatabase, selectedCollections,
    ],
  );

  const handleSelectConversation = useCallback(
    (id: string) => {
      if (isStreaming) return;
      setActiveConversation(id);
    },
    [isStreaming, setActiveConversation],
  );

  const handleDatabaseChange = (db: string) => {
    setSelectedDatabase(db);
    setSelectedCollections([]); // reset collections, will be refetched
  };

  const selectorBar = (
    <div style={{
      padding: '8px 24px',
      borderBottom: '1px solid #f0f0f0',
      background: '#fafafa',
      display: 'flex',
      alignItems: 'center',
      gap: 16,
    }}>
      <Space size={8}>
        <span style={{ fontSize: 13, color: '#666' }}>数据库</span>
        <Select
          value={selectedDatabase}
          onChange={handleDatabaseChange}
          loading={dbLoading}
          style={{ minWidth: 140 }}
          placeholder="选择数据库"
          options={databases.map((db) => ({ label: db, value: db }))}
        />
      </Space>
      <Space size={8}>
        <span style={{ fontSize: 13, color: '#666' }}>集合</span>
        {colLoading ? (
          <Spin size="small" />
        ) : (
          <Select
            mode="multiple"
            value={selectedCollections}
            onChange={setSelectedCollections}
            style={{ minWidth: 240 }}
            placeholder="选择集合"
            options={collections.map((c) => ({ label: c, value: c }))}
            maxTagCount="responsive"
            allowClear
          />
        )}
      </Space>
    </div>
  );

  return (
    <div style={{ display: 'flex', height: '100%' }}>
      {/* Conversation sidebar */}
      <div
        style={{
          width: 220,
          borderRight: '1px solid #f0f0f0',
          background: '#fafafa',
          overflow: 'hidden',
        }}
      >
        <ConversationList onSelect={handleSelectConversation} />
      </div>

      {/* Chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#fff' }}>
        {selectorBar}
        {activeConv ? (
          <>
            <div className="chat-messages-container">
              {activeConv.messages.map((msg, idx) => (
                <ChatMessage key={idx} message={msg} />
              ))}
              {isStreaming && (
                <ChatMessage
                  message={{
                    role: 'assistant',
                    content: streamingContent,
                    sources: streamingSources,
                  }}
                  isStreaming
                />
              )}
              <div ref={messagesEndRef} />
            </div>
            <ChatInput onSend={handleSend} disabled={isStreaming} />
          </>
        ) : (
          <>
            <div className="chat-messages-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <Title level={3} style={{ color: '#999' }}>
                RAG 知识库问答
              </Title>
              <Empty description="开始新对话" />
            </div>
            <ChatInput onSend={handleSend} disabled={isStreaming} />
          </>
        )}
      </div>
    </div>
  );
}
