# RAG Project

A comprehensive Retrieval-Augmented Generation (RAG) project with modular architecture.

## Project Structure

```
rag_project/
├── data_pipeline/      # 数据流水线模块
├── milvus/             # Milvus向量数据库对接模块
├── retrieval/          # 检索模块
├── llm/                # LLM对话集成
├── api/                # 后端服务API模块
├── web/                # 前端（React + Vite）
├── evaluation/         # 评估模块
├── common/             # 公共工具模块
├── tests/              # 测试模块
├── .env                # 环境配置
├── .env.example        # 环境配置示例
├── pyproject.toml      # 项目配置
└── main.py             # 启动文件
```

## Quick Start

### 1. 环境配置
编辑 .env，填入 Milvus、Embedding 模型、LLM API 等配置

### 2. 后端启动

```bash
# 安装 Python 依赖
pip install -r ./requirements.txt

# 启动 API 服务（默认 0.0.0.0:8000）
python main.py
```

启动后可以访问 http://localhost:8000/docs 查看 API 文档。

### 3. 前端启动
（前后端分别运行，Vite 代理 API 请求到后端）：

```bash
cd web
npm install        # 首次安装依赖
npm run dev        # 启动开发服务器 → http://localhost:3000
```

## Modules

1. **data_pipeline**: 数据加载、清洗、分割、向量化流水线
   - 当前问题：
     1. 没有数据清洗
     2. 分割方式固定300字符，没有引入llamaindex等开源框架能力
     3. 数据字段没有保存层级关系
     4. 没有OCR和表格处理
     5. 没有做并发处理
     6. 没有重复检测（chunk_id或file_name等维度）
2. **milvus**: Milvus向量数据库连接、索引管理
   - 当前问题：
     1. 没有做领域、时间等分表设计
     2. 只做了一个向量字段
     3. 没有配合data模块做层级关系入库
3. **retrieval**: 语义检索、相似度匹配
   - 当前问题：
     1. 没有放重排序模块
     2. 没有测试混合检索和向量检索的效果哪个更好
4. **llm**: LLM对话集成
   - 当前问题：
     1. 不会的问题没有拒答
     2. 没有重写/扩展问题
     3. 没有相似度阈值
5. **api**: FastAPI后端服务
6. **evaluation**: 评估模块
7. **web**: 前端页面（React + Ant Design）