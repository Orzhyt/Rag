import { Collapse, Tag, Typography, Space, Tooltip } from 'antd';
import type { SourceCitation as SourceCitationType } from '../../api/types';

const { Text, Paragraph } = Typography;

interface Props {
  sources: SourceCitationType[];
}

export default function SourceCitation({ sources }: Props) {
  if (!sources.length) return null;

  const items = sources.map((s) => ({
    key: s.index,
    label: (
      <Space size={4} style={{ flex: 1, minWidth: 0 }}>
        <Tag color="blue" style={{ margin: 0 }}>[{s.index}]</Tag>
        <Tooltip title={s.source_file} placement="topLeft">
          <Text ellipsis style={{ flex: 1, minWidth: 0, fontSize: 13 }}>
            {s.source_file}
          </Text>
        </Tooltip>
        <Text type="secondary" style={{ fontSize: 12, flexShrink: 0 }}>
          {s.score.toFixed(4)}
        </Text>
      </Space>
    ),
    children: (
      <Paragraph style={{ fontSize: 13, margin: 0, color: '#555' }}>
        {s.content}
      </Paragraph>
    ),
  }));

  return (
    <div className="source-citation" style={{ marginTop: 8 }}>
      <Collapse
        items={items}
        size="small"
        bordered={false}
        style={{ background: 'transparent' }}
      />
    </div>
  );
}
