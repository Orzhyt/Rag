import { Card, Tag, Typography, Collapse } from 'antd';
import type { SearchHit } from '../../api/types';

const { Text } = Typography;

interface Props {
  hit: SearchHit;
  index: number;
}

export default function SearchResultCard({ hit, index }: Props) {
  const sourceFile = (hit.source_file || hit.file_name || '') as string;
  const { score, ...rest } = hit;

  return (
    <Card
      size="small"
      style={{ marginBottom: 8 }}
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Tag color="blue">#{index + 1}</Tag>
          <Text ellipsis style={{ flex: 1, minWidth: 0, fontSize: 13 }}>
            {sourceFile}
          </Text>
        </div>
      }
      extra={<Text type="secondary" style={{ fontSize: 12 }}>{hit.score.toFixed(4)}</Text>}
    >
      <Collapse
        size="small"
        bordered={false}
        defaultActiveKey={['json']}
        style={{ background: 'transparent' }}
        items={[{
          key: 'json',
          label: <Text style={{ fontSize: 12, color: '#999' }}>完整数据</Text>,
          children: (
            <pre style={{ fontSize: 12, margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', background: '#fafafa', padding: 8, borderRadius: 4 }}>
              {JSON.stringify(rest, null, 2)}
            </pre>
          ),
        }]}
      />
    </Card>
  );
}
