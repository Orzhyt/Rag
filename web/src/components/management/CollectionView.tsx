import { useState, useEffect } from 'react';
import {
  Table, Button, Modal, Form, InputNumber, Input, Typography, Tag, message,
  Popconfirm, Space, Descriptions,
} from 'antd';
import { PlusOutlined, DeleteOutlined, ReloadOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { listCollections, createCollection, dropCollection, describeCollection } from '../../api/collections';

const { Title } = Typography;

export default function CollectionView() {
  const [collections, setCollections] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [schemaModalOpen, setSchemaModalOpen] = useState(false);
  const [schemaData, setSchemaData] = useState<any>(null);
  const [form] = Form.useForm();

  const fetchCollections = async () => {
    setLoading(true);
    try {
      const cols = await listCollections();
      setCollections(cols);
    } catch (err: any) {
      message.error(err?.response?.data?.error || '获取集合列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCollections();
  }, []);

  const handleCreate = async () => {
    try {
      const values = await form.validateFields();
      await createCollection({
        collection_name: values.collection_name,
        dim: values.dim,
        description: values.description,
      });
      message.success('创建成功');
      setModalOpen(false);
      form.resetFields();
      fetchCollections();
    } catch (err: any) {
      if (err?.response?.data?.error) message.error(err.response.data.error);
    }
  };

  const handleDrop = async (name: string) => {
    try {
      await dropCollection({ collection_name: name });
      message.success('删除成功');
      fetchCollections();
    } catch (err: any) {
      message.error(err?.response?.data?.error || '删除失败');
    }
  };

  const handleViewSchema = async (name: string) => {
    try {
      const schema = await describeCollection(name);
      setSchemaData(schema);
      setSchemaModalOpen(true);
    } catch (err: any) {
      message.error(err?.response?.data?.error || '获取 Schema 失败');
    }
  };

  const dataSource = collections.map((name) => ({ key: name, name }));

  return (
    <div style={{ padding: 24, maxWidth: 800, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>集合管理</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchCollections} loading={loading}>
            刷新
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
            创建集合
          </Button>
        </Space>
      </div>

      <Table
        dataSource={dataSource}
        loading={loading}
        pagination={false}
        columns={[
          {
            title: '集合名称',
            dataIndex: 'name',
            render: (name: string) => <Tag color="green">{name}</Tag>,
          },
          {
            title: '操作',
            width: 200,
            render: (_, record) => (
              <Space>
                <Button
                  size="small"
                  icon={<InfoCircleOutlined />}
                  onClick={() => handleViewSchema(record.name)}
                >
                  Schema
                </Button>
                <Popconfirm
                  title={`确认删除集合 "${record.name}"？`}
                  onConfirm={() => handleDrop(record.name)}
                >
                  <Button danger size="small" icon={<DeleteOutlined />}>
                    删除
                  </Button>
                </Popconfirm>
              </Space>
            ),
          },
        ]}
      />

      <Modal
        title="创建集合"
        open={modalOpen}
        onOk={handleCreate}
        onCancel={() => setModalOpen(false)}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="collection_name" label="集合名称" rules={[{ required: true }]}>
            <Input placeholder="输入集合名称" />
          </Form.Item>
          <Form.Item name="dim" label="向量维度" initialValue={1024}>
            <InputNumber min={1} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={2} placeholder="可选描述" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="集合 Schema"
        open={schemaModalOpen}
        onCancel={() => setSchemaModalOpen(false)}
        footer={null}
        width={600}
      >
        {schemaData && (
          <Descriptions bordered size="small" column={1}>
            <Descriptions.Item label="名称">{schemaData.name}</Descriptions.Item>
            <Descriptions.Item label="描述">{schemaData.description}</Descriptions.Item>
            <Descriptions.Item label="字段">
              <Table
                size="small"
                pagination={false}
                dataSource={schemaData.fields.map((f: any, i: number) => ({ ...f, key: i }))}
                columns={[
                  { title: '名称', dataIndex: 'name' },
                  { title: '类型', dataIndex: 'type' },
                  { title: '主键', dataIndex: 'is_primary_key', render: (v: boolean) => v ? '✓' : '' },
                ]}
              />
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </div>
  );
}
