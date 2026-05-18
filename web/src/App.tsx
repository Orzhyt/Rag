import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import AppLayout from './components/layout/AppLayout';
import ChatView from './components/chat/ChatView';
import SearchView from './components/search/SearchView';
import ManageView from './components/management/ManageView';
import IngestView from './components/management/IngestView';
import './styles/global.css';

export default function App() {
  return (
    <ConfigProvider locale={zhCN}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<ChatView />} />
            <Route path="/search" element={<SearchView />} />
            <Route path="/manage" element={<ManageView />} />
            <Route path="/ingest" element={<IngestView />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ConfigProvider>
  );
}
