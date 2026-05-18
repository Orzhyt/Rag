import { useState } from 'react';
import { Input, Button, Radio, Space, Typography, Spin, Empty, Select, Slider } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import SearchResultCard from './SearchResultCard';
import { vectorSearch, hybridSearch } from '../../api/search';
import { listCollections } from '../../api/collections';
import type { SearchHit } from '../../api/types';

const { Title } = Typography;

export default function SearchView() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'hybrid' | 'vector'>('hybrid');
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState<SearchHit[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [collectionOptions, setCollectionOptions] = useState<{ label: string; value: string }[]>([]);
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);

  const loadCollections = async () => {
    try {
      const cols = await listCollections();
      setCollectionOptions(cols.map((c) => ({ label: c, value: c })));
    } catch {
      // ignore
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const req = {
        query: query.trim(),
        top_k: topK,
        collection_names: selectedCollections.length ? selectedCollections : undefined,
      };
      const resp = mode === 'hybrid' ? await hybridSearch(req) : await vectorSearch(req);
      setResults(resp.results);
      setTotal(resp.total);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 900, margin: '0 auto' }}>
      <Title level={4}>知识库检索</Title>

      <Space direction="vertical" style={{ width: '100%' }} size={12}>
        <Input.Search
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onSearch={handleSearch}
          placeholder="输入检索内容..."
          enterButton={<Button type="primary" icon={<SearchOutlined />}>检索</Button>}
          size="large"
          loading={loading}
        />

        <Space wrap>
          <Radio.Group value={mode} onChange={(e) => setMode(e.target.value)} size="small">
            <Radio.Button value="hybrid">混合检索</Radio.Button>
            <Radio.Button value="vector">向量检索</Radio.Button>
          </Radio.Group>

          <span style={{ fontSize: 13, color: '#999' }}>返回数量:</span>
          <Slider min={1} max={50} value={topK} onChange={setTopK} style={{ width: 100 }} />
          <span style={{ fontSize: 13 }}>{topK}</span>

          <Select
            mode="multiple"
            placeholder="选择集合"
            options={collectionOptions}
            value={selectedCollections}
            onChange={setSelectedCollections}
            onDropdownVisibleChange={(open) => open && loadCollections()}
            style={{ minWidth: 200 }}
            size="small"
          />
        </Space>
      </Space>

      <div style={{ marginTop: 16 }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Spin size="large" />
          </div>
        ) : results.length ? (
          <>
            <Typography.Text type="secondary" style={{ fontSize: 13 }}>
              共 {total} 条结果
            </Typography.Text>
            {results.map((hit, idx) => (
              <SearchResultCard key={idx} hit={hit} index={idx} />
            ))}
          </>
        ) : (
          <Empty description="暂无检索结果" style={{ marginTop: 40 }} />
        )}
      </div>
    </div>
  );
}
