import { useState, useEffect } from 'react';
import { Table, Button, Modal, Input, Form, Typography, Tag, message, Popconfirm, Space } from 'antd';
import { PlusOutlined, DeleteOutlined, ReloadOutlined } from '@ant-design/icons';
import { listDatabases, createDatabase, dropDatabase } from '../../api/databases';

const { Title } = Typography;

export default function DatabaseView() {
  const [databases, setDatabases] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [form] = Form.useForm();

  const fetchDatabases = async () => {
    setLoading(true);
    try {
      const dbs = await listDatabases();
      setDatabases(dbs);
    } catch (err: any) {
      message.error(err?.response?.data?.error || '获取数据库列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatabases();
  }, []);

  const handleCreate = async () => {
    try {
      const values = await form.validateFields();
      await createDatabase({ db_name: values.db_name });
      message.success('创建成功');
      setModalOpen(false);
      form.resetFields();
      fetchDatabases();
    } catch (err: any) {
      if (err?.response?.data?.error) message.error(err.response.data.error);
    }
  };

  const handleDrop = async (dbName: string) => {
    try {
      await dropDatabase({ db_name: dbName });
      message.success('删除成功');
      fetchDatabases();
    } catch (err: any) {
      message.error(err?.response?.data?.error || '删除失败');
    }
  };

  const dataSource = databases.map((name) => ({ key: name, name }));

  return (
    <div style={{ padding: 24, maxWidth: 800, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4} style={{ margin: 0 }}>数据库管理</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchDatabases} loading={loading}>
            刷新
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
            创建数据库
          </Button>
        </Space>
      </div>

      <Table
        dataSource={dataSource}
        loading={loading}
        pagination={false}
        columns={[
          {
            title: '数据库名称',
            dataIndex: 'name',
            render: (name: string) => (
              <Space>
                <Tag color="blue">{name}</Tag>
              </Space>
            ),
          },
          {
            title: '操作',
            width: 100,
            render: (_, record) => (
              <Popconfirm
                title={`确认删除数据库 "${record.name}"？`}
                onConfirm={() => handleDrop(record.name)}
              >
                <Button danger size="small" icon={<DeleteOutlined />}>
                  删除
                </Button>
              </Popconfirm>
            ),
          },
        ]}
      />

      <Modal
        title="创建数据库"
        open={modalOpen}
        onOk={handleCreate}
        onCancel={() => setModalOpen(false)}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="db_name" label="数据库名称" rules={[{ required: true }]}>
            <Input placeholder="输入数据库名称" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
