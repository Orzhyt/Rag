import { Card, Tag, Typography, Progress } from 'antd';
import type { SearchHit } from '../../api/types';

const { Text, Paragraph } = Typography;

interface Props {
  hit: SearchHit;
  index: number;
}

export default function SearchResultCard({ hit, index }: Props) {
  const scorePercent = Math.min(Math.round(hit.score * 100), 100);
  const sourceFile = (hit.source_file || hit.file_name || '') as string;
  const content = (hit.content || '') as string;

  return (
    <Card
      size="small"
      style={{ marginBottom: 8 }}
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Tag color="blue">#{index + 1}</Tag>
          <Text ellipsis style={{ maxWidth: 300, fontSize: 13 }}>
            {sourceFile}
          </Text>
        </div>
      }
      extra={
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Progress
            percent={scorePercent}
            size="small"
            style={{ width: 60 }}
            format={() => `${hit.score.toFixed(4)}`}
          />
        </div>
      }
    >
      <Paragraph
        ellipsis={{ rows: 3, expandable: true, symbol: '展开' }}
        style={{ fontSize: 13, margin: 0 }}
      >
        {content}
      </Paragraph>
    </Card>
  );
}
