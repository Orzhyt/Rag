import { useState, useEffect } from 'react';
import {
  Typography, Button, Modal, Form, Input, InputNumber, Table, Tag, Space, message, Popconfirm,
} from 'antd';
import { PlusOutlined, DeleteOutlined, ReloadOutlined } from '@ant-design/icons';
import { listDatabases, createDatabase, dropDatabase } from '../../api/databases';
import { listCollections, createCollection, dropCollection } from '../../api/collections';

const { Title } = Typography;

export default function ManageView() {
  const [databases, setDatabases] = useState<string[]>([]);
  const [selectedDatabase, setSelectedDatabase] = useState<string | null>(null);
  const [collections, setCollections] = useState<string[]>([]);
  const [dbLoading, setDbLoading] = useState(false);
  const [colLoading, setColLoading] = useState(false);

  const [dbModalOpen, setDbModalOpen] = useState(false);
  const [colModalOpen, setColModalOpen] = useState(false);
  const [dbForm] = Form.useForm();
  const [colForm] = Form.useForm();

  const fetchDatabases = async () => {
    setDbLoading(true);
    try {
      const dbs = await listDatabases();
      setDatabases(dbs);
      if (dbs.length > 0 && !selectedDatabase) {
        setSelectedDatabase(dbs[0]);
      }
    } catch {
      message.error('获取数据库列表失败');
    } finally {
      setDbLoading(false);
    }
  };

  const fetchCollections = async (db: string) => {
    setColLoading(true);
    try {
      const cols = await listCollections(db);
      setCollections(cols);
    } catch {
      message.error('获取集合列表失败');
    } finally {
      setColLoading(false);
    }
  };

  useEffect(() => { fetchDatabases(); }, []);

  useEffect(() => {
    if (selectedDatabase) fetchCollections(selectedDatabase);
    else setCollections([]);
  }, [selectedDatabase]);

  const handleCreateDatabase = async () => {
    try {
      const { db_name } = await dbForm.validateFields();
      await createDatabase({ db_name });
      message.success(`数据库 "${db_name}" 创建成功`);
      setDbModalOpen(false);
      dbForm.resetFields();
      fetchDatabases();
    } catch (err: any) {
      if (err?.response?.data?.error) message.error(err.response.data.error);
    }
  };

  const handleDropDatabase = async (name: string) => {
    try {
      await dropDatabase({ db_name: name });
      message.success(`数据库 "${name}" 已删除`);
      if (selectedDatabase === name) {
        setSelectedDatabase(null);
        setCollections([]);
      }
      fetchDatabases();
    } catch (err: any) {
      message.error(err?.response?.data?.error || '删除失败');
    }
  };

  const handleCreateCollection = async () => {
    try {
      const values = await colForm.validateFields();
      await createCollection({
        collection_name: values.collection_name,
        dim: values.dim,
        description: values.description,
        database: selectedDatabase || undefined,
      });
      message.success('集合创建成功');
      setColModalOpen(false);
      colForm.resetFields();
      if (selectedDatabase) fetchCollections(selectedDatabase);
    } catch (err: any) {
      if (err?.response?.data?.error) message.error(err.response.data.error);
    }
  };

  const handleDropCollection = async (name: string) => {
    try {
      await dropCollection({
        collection_name: name,
        database: selectedDatabase || undefined,
      });
      message.success(`集合 "${name}" 已删除`);
      if (selectedDatabase) fetchCollections(selectedDatabase);
    } catch (err: any) {
      message.error(err?.response?.data?.error || '删除失败');
    }
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: 24 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16, alignItems: 'center' }}>
        <Title level={4} style={{ margin: 0 }}>向量库管理</Title>
      </div>

      {/* Two-column layout */}
      <div style={{ flex: 1, display: 'flex', gap: 24, minHeight: 0 }}>
        {/* Left: Databases */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, alignItems: 'center' }}>
            <Typography.Text strong style={{ fontSize: 15 }}>数据库</Typography.Text>
            <Space size={4}>
              <Button size="small" icon={<ReloadOutlined />} onClick={fetchDatabases} loading={dbLoading} />
              <Button size="small" type="primary" icon={<PlusOutlined />} onClick={() => setDbModalOpen(true)}>创建</Button>
            </Space>
          </div>
          <div style={{ flex: 1, overflowY: 'auto' }}>
            <Table
              dataSource={databases.map((name) => ({ key: name, name }))}
              loading={dbLoading}
              pagination={false}
              size="small"
              onRow={(record) => ({
                onClick: () => setSelectedDatabase(record.name),
                style: { cursor: 'pointer', background: record.name === selectedDatabase ? '#e6f4ff' : undefined },
              })}
              columns={[
                {
                  title: '名称',
                  dataIndex: 'name',
                  render: (name: string) => (
                    <Tag color={name === selectedDatabase ? 'blue' : 'default'}>
                      {name}
                    </Tag>
                  ),
                },
                {
                  title: '操作',
                  width: 60,
                  render: (_, record) => (
                    <Popconfirm
                      title={`确认删除数据库 "${record.name}"？`}
                      onConfirm={() => handleDropDatabase(record.name)}
                    >
                      <Button danger size="small" icon={<DeleteOutlined />} style={{ visibility: 'hidden' }} />
                    </Popconfirm>
                  ),
                },
              ]}
            />
          </div>
        </div>

        {/* Right: Collections */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, alignItems: 'center' }}>
            <Typography.Text strong style={{ fontSize: 15 }}>
              集合{selectedDatabase ? ` — ${selectedDatabase}` : ''}
            </Typography.Text>
            <Space size={4}>
              <Button
                size="small"
                icon={<ReloadOutlined />}
                onClick={() => selectedDatabase && fetchCollections(selectedDatabase)}
                loading={colLoading}
                disabled={!selectedDatabase}
              />
              <Button
                size="small"
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setColModalOpen(true)}
                disabled={!selectedDatabase}
              >
                创建
              </Button>
            </Space>
          </div>
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {!selectedDatabase ? (
              <Typography.Text type="secondary">请先点击选择一个数据库</Typography.Text>
            ) : (
              <Table
                dataSource={collections.map((name) => ({ key: name, name }))}
                loading={colLoading}
                pagination={false}
                size="small"
                columns={[
                  {
                    title: '名称',
                    dataIndex: 'name',
                    render: (name: string) => <Tag color="green">{name}</Tag>,
                  },
                  {
                    title: '操作',
                    width: 60,
                    render: (_, record) => (
                      <Popconfirm
                        title={`确认删除集合 "${record.name}"？`}
                        onConfirm={() => handleDropCollection(record.name)}
                      >
                        <Button danger size="small" icon={<DeleteOutlined />} />
                      </Popconfirm>
                    ),
                  },
                ]}
              />
            )}
          </div>
        </div>
      </div>

      {/* Create Database Modal */}
      <Modal
        title="创建数据库"
        open={dbModalOpen}
        onOk={handleCreateDatabase}
        onCancel={() => { setDbModalOpen(false); dbForm.resetFields(); }}
      >
        <Form form={dbForm} layout="vertical">
          <Form.Item name="db_name" label="数据库名称" rules={[{ required: true }]}>
            <Input placeholder="输入数据库名称" />
          </Form.Item>
        </Form>
      </Modal>

      {/* Create Collection Modal */}
      <Modal
        title="创建集合"
        open={colModalOpen}
        onOk={handleCreateCollection}
        onCancel={() => { setColModalOpen(false); colForm.resetFields(); }}
      >
        <Form form={colForm} layout="vertical">
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
    </div>
  );
}
