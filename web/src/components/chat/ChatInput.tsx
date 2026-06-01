import { useState } from 'react';
import { Input, Button, Space, Tooltip } from 'antd';
import { SendOutlined } from '@ant-design/icons';

const { TextArea } = Input;

interface Props {
  onSend: (query: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState('');

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ padding: '12px 24px', borderTop: '1px solid #f0f0f0', background: '#fff', width: '100%' }}>
      <Space.Compact style={{ width: '100%', display: 'flex' }}>
        <TextArea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入问题，按 Enter 发送..."
          autoSize={{ minRows: 1, maxRows: 4 }}
          disabled={disabled}
          style={{ borderRadius: '8px 0 0 8px', flex: 1 }}
        />
        <Tooltip title="发送">
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            style={{ height: 'auto', borderRadius: '0 8px 8px 0' }}
          />
        </Tooltip>
      </Space.Compact>
    </div>
  );
}
