import { useState, useEffect } from 'react';
import { Input, Button, Space, Typography, Spin, Empty, Select, Slider, message, InputNumber } from 'antd';
import { SearchOutlined, PlusOutlined, MinusCircleOutlined } from '@ant-design/icons';
import SearchResultCard from './SearchResultCard';
import { hybridSearch } from '../../api/search';
import { listDatabases } from '../../api/databases';
import { listCollections } from '../../api/collections';
import type { SearchHit, FieldWeight } from '../../api/types';

const { Title, Text } = Typography;

export default function SearchView() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState<SearchHit[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  const [databases, setDatabases] = useState<string[]>([]);
  const [selectedDatabase, setSelectedDatabase] = useState<string | undefined>();
  const [collections, setCollections] = useState<string[]>([]);
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  const [dbLoading, setDbLoading] = useState(false);
  const [colLoading, setColLoading] = useState(false);

  // Weight configs
  const [annsFields, setAnnsFields] = useState<FieldWeight[]>([{ field: 'content_embedding', weight: 1 }]);
  const [bm25Fields, setBm25Fields] = useState<FieldWeight[]>([]);

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
        top_k: topK,
        collection_names: selectedCollections.length ? selectedCollections : undefined,
        database: selectedDatabase,
        anns_fields: annsFields.filter((f) => f.field.trim() && f.weight > 0),
        bm25_fields: bm25Fields.filter((f) => f.field.trim() && f.weight > 0),
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

          <Text style={{ fontSize: 13, color: '#666' }}>topK</Text>
          <InputNumber min={1} max={30} value={topK} onChange={(v) => v && setTopK(v)} size="small" style={{ width: 60 }} />
        </Space>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <Text style={{ fontSize: 13, color: '#666', whiteSpace: 'nowrap' }}>向量字段</Text>
          {annsFields.map((f, i) => (
            <Space key={i} size={4}>
              <Input
                value={f.field}
                onChange={(e) => { const next = [...annsFields]; next[i] = { ...next[i], field: e.target.value }; setAnnsFields(next); }}
                size="small"
                style={{ width: 150 }}
              />
              <InputNumber
                value={f.weight}
                onChange={(v) => { if (v != null) { const next = [...annsFields]; next[i] = { ...next[i], weight: v }; setAnnsFields(next); } }}
                min={0} max={1} step={0.1}
                size="small"
                style={{ width: 60 }}
              />
              {annsFields.length > 1 && (
                <MinusCircleOutlined
                  style={{ color: '#999', cursor: 'pointer' }}
                  onClick={() => setAnnsFields(annsFields.filter((_, j) => j !== i))}
                />
              )}
            </Space>
          ))}
          <Button
            size="small"
            type="dashed"
            icon={<PlusOutlined />}
            onClick={() => setAnnsFields([...annsFields, { field: '', weight: 0.5 }])}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <Text style={{ fontSize: 13, color: '#666', whiteSpace: 'nowrap' }}>BM25字段</Text>
          {bm25Fields.map((f, i) => (
            <Space key={i} size={4}>
              <Input
                value={f.field}
                onChange={(e) => { const next = [...bm25Fields]; next[i] = { ...next[i], field: e.target.value }; setBm25Fields(next); }}
                size="small"
                style={{ width: 100 }}
              />
              <InputNumber
                value={f.weight}
                onChange={(v) => { if (v != null) { const next = [...bm25Fields]; next[i] = { ...next[i], weight: v }; setBm25Fields(next); } }}
                min={0} max={1} step={0.1}
                size="small"
                style={{ width: 60 }}
              />
              <MinusCircleOutlined
                style={{ color: '#999', cursor: 'pointer' }}
                onClick={() => setBm25Fields(bm25Fields.filter((_, j) => j !== i))}
              />
            </Space>
          ))}
          <Button
            size="small"
            type="dashed"
            icon={<PlusOutlined />}
            onClick={() => setBm25Fields([...bm25Fields, { field: '', weight: 0.3 }])}
          />
        </div>
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
