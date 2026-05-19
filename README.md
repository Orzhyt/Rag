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
   - 编辑 .env，填入 Milvus、Embedding 模型、LLM API 等配置
   - GPU 用户请根据cuda版本（nvidia-smi）卸载torch后改用: 
   - pip install torch --find-links https://mirrors.aliyun.com/pytorch-wheels/cu130
   - 昇腾 NPU 用户请额外安装（需要查询cann版本与torch版本对应表）: torch_npu==2.5.1

### 2. 后端启动

```bash
pip install -r requirements.txt
```

**后端服务启动，以9100端口为例：**

| 环境 | 命令 |
|------|------|
| Linux / Mac | `API_PORT=9100 python main.py` |
| Windows PowerShell | `$env:API_PORT=9100; python main.py` |

### 3. 前端启动

```bash
cd web
npm install
```

**前端服务启动以9200端口为例，后端以9100为例：**

| 环境 | 命令 |
|------|------|
| Linux / Mac | `PORT=9200 API_PORT=9100 npm run dev` |
| Windows PowerShell | `$env:PORT=9200; $env:API_PORT=9100; npm run dev` |

## 模块

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