import { Layout, Menu } from 'antd';
import {
  MessageOutlined,
  SearchOutlined,
  CloudUploadOutlined,
} from '@ant-design/icons';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';

const { Sider, Content } = Layout;

const menuItems = [
  { key: '/', icon: <MessageOutlined />, label: 'RAG 对话' },
  { key: '/search', icon: <SearchOutlined />, label: '知识库检索' },
  { key: '/ingest', icon: <CloudUploadOutlined />, label: '数据入库' },
];

export default function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Layout style={{ height: '100vh' }}>
      <Sider
        width={180}
        theme="light"
        style={{ borderRight: '1px solid #f0f0f0' }}
      >
        <div
          style={{
            height: 48,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderBottom: '1px solid #f0f0f0',
            fontWeight: 700,
            fontSize: 16,
            color: '#1677ff',
          }}
        >
          RAG System
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ borderRight: 0 }}
        />
      </Sider>
      <Layout>
        <Content style={{ height: '100%', overflow: 'hidden' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
