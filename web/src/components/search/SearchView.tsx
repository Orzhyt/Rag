import { useState, useEffect } from 'react';
import { Input, Button, Space, Typography, Spin, Empty, Select, message } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import SearchResultCard from './SearchResultCard';
import { hybridSearch } from '../../api/search';
import { listDatabases } from '../../api/databases';
import { listCollections } from '../../api/collections';
import type { SearchHit } from '../../api/types';

const { Title, Text } = Typography;

export default function SearchView() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchHit[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  const [databases, setDatabases] = useState<string[]>([]);
  const [selectedDatabase, setSelectedDatabase] = useState<string | undefined>();
  const [collections, setCollections] = useState<string[]>([]);
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  const [dbLoading, setDbLoading] = useState(false);
  const [colLoading, setColLoading] = useState(false);

  useEffect(() => {
    setDbLoading(true);
    listDatabases()
      .then((dbs) => {
        setDatabases(dbs);
        if (dbs.length > 0 && !selectedDatabase) setSelectedDatabase(dbs[0]);
      })
      .catch(() => message.error('获取数据库列表失败'))
      .finally(() => setDbLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedDatabase) { setCollections([]); return; }
    setColLoading(true);
    listCollections(selectedDatabase)
      .then((cols) => { setCollections(cols); setSelectedCollections(cols); })
      .catch(() => message.error('获取集合列表失败'))
      .finally(() => setColLoading(false));
  }, [selectedDatabase]);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const req = {
        query: query.trim(),
        collection_names: selectedCollections.length ? selectedCollections : undefined,
        database: selectedDatabase,
      };
      const resp = await hybridSearch(req);
      setResults(resp.results);
      setTotal(resp.total);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: 24 }}>
      <Title level={4} style={{ marginBottom: 12 }}>知识库检索</Title>

      {/* Controls */}
      <Space direction="vertical" style={{ width: '100%' }} size={8}>
        <Input.Search
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onSearch={handleSearch}
          placeholder="输入检索内容..."
          enterButton={<Button type="primary" icon={<SearchOutlined />}>检索</Button>}
          size="large"
          loading={loading}
        />

        <Space wrap size={8}>
          <Text style={{ fontSize: 13, color: '#666' }}>数据库</Text>
          <Select
            value={selectedDatabase}
            onChange={(db) => { setSelectedDatabase(db); setSelectedCollections([]); }}
            loading={dbLoading}
            style={{ minWidth: 120 }}
            placeholder="选择数据库"
            options={databases.map((db) => ({ label: db, value: db }))}
            size="small"
          />

          <Text style={{ fontSize: 13, color: '#666' }}>集合</Text>
          <Select
            mode="multiple"
            placeholder="选择集合"
            options={collections.map((c) => ({ label: c, value: c }))}
            value={selectedCollections}
            onChange={setSelectedCollections}
            style={{ minWidth: 180 }}
            size="small"
            loading={colLoading}
          />
        </Space>
      </Space>

      {/* Results with scroll */}
      <div style={{ marginTop: 16, flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {loading ? (
          <div style={{ textAlign: 'center', padding: 40 }}><Spin size="large" /></div>
        ) : results.length ? (
          <>
            <Text type="secondary" style={{ fontSize: 13 }}>共 {total} 条结果</Text>
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
