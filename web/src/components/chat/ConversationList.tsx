import { Button, List, Popconfirm, Typography, Space } from 'antd';
import { PlusOutlined, DeleteOutlined, MessageOutlined } from '@ant-design/icons';
import { useChatStore } from '../../stores/chatStore';

const { Text } = Typography;

interface Props {
  onSelect: (id: string) => void;
}

export default function ConversationList({ onSelect }: Props) {
  const { conversations, activeConversationId, createConversation, deleteConversation } = useChatStore();

  const handleNew = () => {
    const id = createConversation();
    onSelect(id);
  };

  const sorted = Object.entries(conversations)
    .map(([id, conv]) => ({ id, ...conv }))
    .sort((a, b) => {
      const aTime = a.messages.length ? 0 : 1;
      const bTime = b.messages.length ? 0 : 1;
      return aTime - bTime;
    });

  return (
    <div style={{ padding: '12px 8px', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Button
        type="dashed"
        icon={<PlusOutlined />}
        block
        onClick={handleNew}
        style={{ marginBottom: 12 }}
      >
        新对话
      </Button>
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <List
          size="small"
          dataSource={sorted}
          renderItem={(item) => {
            const firstMsg = item.messages.find((m) => m.role === 'user');
            const isActive = item.id === activeConversationId;
            return (
              <List.Item
                onClick={() => onSelect(item.id)}
                style={{
                  cursor: 'pointer',
                  background: isActive ? '#e6f4ff' : 'transparent',
                  borderRadius: 6,
                  padding: '6px 8px',
                  marginBottom: 2,
                }}
                actions={[
                  <Popconfirm
                    key="delete"
                    title="删除此对话？"
                    onConfirm={(e) => {
                      e?.stopPropagation();
                      deleteConversation(item.id);
                    }}
                    onCancel={(e) => e?.stopPropagation()}
                  >
                    <DeleteOutlined
                      style={{ color: '#999', fontSize: 12 }}
                      onClick={(e) => e.stopPropagation()}
                    />
                  </Popconfirm>,
                ]}
              >
                <Space size={4} style={{ overflow: 'hidden', flex: 1 }}>
                  <MessageOutlined style={{ color: '#999', fontSize: 12 }} />
                  <Text ellipsis style={{ fontSize: 13, maxWidth: 140 }}>
                    {firstMsg?.content || '新对话'}
                  </Text>
                </Space>
              </List.Item>
            );
          }}
        />
      </div>
    </div>
  );
}
