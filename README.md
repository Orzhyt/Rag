# RAG Project

A comprehensive Retrieval-Augmented Generation (RAG) project with modular architecture.

## Project Structure

```
rag_project/
├── data_pipeline/      # 数据流水线模块
├── milvus/             # Milvus向量数据库对接模块
├── retrieval/          # 检索模块
├── agent/              # 智能体对话模块
├── api/                # 后端服务API模块
├── common/             # 公共工具模块
├── tests/              # 测试模块
├── .env                # 环境配置
├── .env.example        # 环境配置示例
├── pyproject.toml      # 项目配置
└── main.py             # 入口文件
```

## Modules

1. **data_pipeline**: 数据加载、清洗、分割、向量化流水线
   - 当前问题：
     1. 没有数据清洗
     2. 分割方式固定500字符
     3. 数据字段没有保存层级关系
     4. 图片、表格无独立处理
     5. 没有做并发处理
2. **milvus**: Milvus向量数据库连接、索引管理
3. **retrieval**: 语义检索、相似度匹配
4. **agent**: 对话智能体、LLM集成
5. **api**: FastAPI后端服务

## Setup

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

3. Run the API:
   ```bash
   python main.py
   ```

## API Endpoints

- `POST /api/v1/ingest`: 数据入库
- `POST /api/v1/query`: 检索问答
- `GET /api/v1/health`: 健康检查