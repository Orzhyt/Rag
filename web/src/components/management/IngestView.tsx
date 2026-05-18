import { useState } from 'react';
import {
  Form, Input, InputNumber, Switch, Button, Typography, message, Space, Card, Statistic,
} from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { ingestData } from '../../api/data';
import type { IngestResponse } from '../../api/types';

const { Title } = Typography;

export default function IngestView() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [form] = Form.useForm();

  const handleIngest = async () => {
    try {
      const values = await form.validateFields();
      setLoading(true);
      setResult(null);
      const resp = await ingestData({
        folder_path: values.folder_path,
        chunk_size: values.chunk_size,
        chunk_overlap: values.chunk_overlap,
        upsert_mode: values.upsert_mode,
        collection_name: values.collection_name || undefined,
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
        <Form form={form} layout="vertical" initialValues={{
          chunk_size: 500,
          chunk_overlap: 50,
          upsert_mode: true,
        }}>
          <Form.Item
            name="folder_path"
            label="文件夹路径"
            rules={[{ required: true, message: '请输入文件夹路径' }]}
          >
            <Input placeholder="例如: /data/documents" />
          </Form.Item>

          <Space style={{ width: '100%' }} size={16}>
            <Form.Item name="chunk_size" label="分块大小" style={{ width: 200 }}>
              <InputNumber min={100} max={2000} />
            </Form.Item>
            <Form.Item name="chunk_overlap" label="分块重叠" style={{ width: 200 }}>
              <InputNumber min={0} max={500} />
            </Form.Item>
          </Space>

          <Form.Item name="upsert_mode" label="覆盖模式" valuePropName="checked">
            <Switch checkedChildren="Upsert" unCheckedChildren="Insert" />
          </Form.Item>

          <Form.Item name="collection_name" label="目标集合（可选）">
            <Input placeholder="不填则使用默认集合" />
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
