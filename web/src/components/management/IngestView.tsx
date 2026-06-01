import { useState, useEffect } from 'react';
import {
  Form, Input, Button, Typography, message, Space, Card, Statistic, Select, Spin,
} from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { ingestData } from '../../api/data';
import { listDatabases } from '../../api/databases';
import { listCollections } from '../../api/collections';
import type { IngestResponse } from '../../api/types';

const { Title } = Typography;

export default function IngestView() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [form] = Form.useForm();

  const [databases, setDatabases] = useState<string[]>([]);
  const [collections, setCollections] = useState<string[]>([]);
  const [selectedDatabase, setSelectedDatabase] = useState<string | undefined>();
  const [dbLoading, setDbLoading] = useState(false);
  const [colLoading, setColLoading] = useState(false);

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

  useEffect(() => {
    if (!selectedDatabase) {
      setCollections([]);
      return;
    }
    setColLoading(true);
    listCollections(selectedDatabase)
      .then(setCollections)
      .catch(() => message.error('获取集合列表失败'))
      .finally(() => setColLoading(false));
  }, [selectedDatabase]);

  const handleIngest = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);
      setResult(null);
      const resp = await ingestData({
        folder_path: values.folder_path,
        collection_name: values.collection_name || undefined,
        database: selectedDatabase,
      });
      setResult(resp);
      message.success(`入库成功：${resp.chunks_inserted} 条`);
    } catch (err: any) {
      if (err?.response?.data?.error) message.error(err.response.data.error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24, maxWidth: 700, margin: '0 auto' }}>
      <Title level={4}>数据入库</Title>

      <Card>
        <Form form={form} layout="vertical">
          <Space style={{ width: '100%' }} size={16}>
            <Form.Item label="数据库" style={{ width: 200 }}>
              <Select
                value={selectedDatabase}
                onChange={setSelectedDatabase}
                loading={dbLoading}
                style={{ width: '100%' }}
                placeholder="选择数据库"
                options={databases.map((db) => ({ label: db, value: db }))}
              />
            </Form.Item>
            <Form.Item name="collection_name" label="目标集合" style={{ width: 200 }}>
              {colLoading ? (
                <Spin size="small" />
              ) : (
                <Select
                  style={{ width: '100%' }}
                  placeholder="不填则使用默认集合"
                  options={collections.map((c) => ({ label: c, value: c }))}
                  allowClear
                />
              )}
            </Form.Item>
          </Space>

          <Form.Item
            name="folder_path"
            label="文件夹路径"
            rules={[{ required: true, message: '请输入文件夹路径' }]}
          >
            <Input placeholder="例如: /data/documents" />
          </Form.Item>

          <Form.Item>
            <Button
              type="primary"
              icon={<UploadOutlined />}
              onClick={handleIngest}
              loading={loading}
              size="large"
            >
              开始入库
            </Button>
          </Form.Item>
        </Form>
      </Card>

      {result && (
        <Card style={{ marginTop: 16 }}>
          <Space size={32}>
            <Statistic title="扫描文件" value={result.files_scanned} />
            <Statistic title="解析分块" value={result.chunks_parsed} />
            <Statistic title="插入条数" value={result.chunks_inserted} valueStyle={{ color: '#3f8600' }} />
          </Space>
        </Card>
      )}
    </div>
  );
}
