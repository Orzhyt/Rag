import { useState } from 'react';
import { Input, Button, Space, Tooltip, Slider } from 'antd';
import { SendOutlined, SettingOutlined } from '@ant-design/icons';

const { TextArea } = Input;

interface Props {
  onSend: (query: string, topK: number) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState('');
  const [topK, setTopK] = useState(5);
  const [showSettings, setShowSettings] = useState(false);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, topK);
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
      {showSettings && (
        <div style={{ marginBottom: 8, padding: '8px 12px', background: '#fafafa', borderRadius: 8 }}>
          <Space>
            <span style={{ fontSize: 13 }}>检索数量:</span>
            <Slider
              min={1}
              max={20}
              value={topK}
              onChange={setTopK}
              style={{ width: 120 }}
            />
            <span style={{ fontSize: 13, fontWeight: 500 }}>{topK}</span>
          </Space>
        </div>
      )}
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
        <Tooltip title="检索设置">
          <Button
            icon={<SettingOutlined />}
            onClick={() => setShowSettings(!showSettings)}
            style={{ height: 'auto', borderRadius: 0 }}
          />
        </Tooltip>
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={handleSend}
          disabled={disabled || !value.trim()}
          style={{ height: 'auto', borderRadius: '0 8px 8px 0' }}
        />
      </Space.Compact>
    </div>
  );
}
