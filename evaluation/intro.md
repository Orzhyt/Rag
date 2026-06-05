# 评估模块使用指南

## 快速开始

```bash
# 完整评估（7 个指标：4 ragas LLM-judge + 3 确定性检索指标）
python -m evaluation.runner evaluate \
  --dataset evaluation/test_dataset_optimized.jsonl \
  --database cjc \
  --collections test

# 前后对比
python -m evaluation.runner compare \
  evaluation/results/eval_before.json \
  evaluation/results/eval_after.json

# 直接用 compare 模块
python -m evaluation.compare \
  evaluation/results/eval_before.json \
  evaluation/results/eval_after.json
  
# 只跑检索指标（不调 LLM judge）
python -m evaluation.runner evaluate \
  --dataset evaluation/test_dataset_optimized.jsonl \
  --database cjc \
  --collections test \
  --metrics hit_rate,mrr,recall

# 只跑 ragas LLM-judge 指标
python -m evaluation.runner evaluate \
  --dataset evaluation/test_dataset_optimized.jsonl \
  --database cjc \
  --collections test \
  --metrics faithfulness,answer_relevancy,context_precision,context_recall
```

## CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 测试数据集 JSONL 路径 | （必填） |
| `--metrics` | 指标列表（逗号分隔） | `faithfulness,answer_relevancy,context_precision,context_recall,hit_rate,mrr,recall` |
| `--top-k` | 检索返回文档数 | 5 |
| `--search-mode` | 检索模式：`hybrid`（ANN+BM25）/ `vector`（纯向量） | hybrid |
| `--database` | Milvus 数据库名 | 环境变量 `MILVUS_DATABASE` |
| `--collections` | Milvus 集合名列表 | 默认集合 |
| `--similarity-threshold` | 检索指标命中阈值 | 0.7 |
| `--output-dir` | 结果输出目录 | evaluation/results |

---

## 指标说明

### ragas LLM-judge 指标

#### Faithfulness（忠实度）

两次 LLM 调用 + 一次计算：

1. LLM Call 1 — 把答案拆成原子语句：
   "Given a question and answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements..."
2. LLM Call 2 — 逐句判断能否从上下文推断出来（NLI 判定 0/1）：
   "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not..."
3. 计算：score = 可推断的语句数 / 总语句数

#### AnswerRelevancy（答案相关性）

LLM 调用 + Embedding 余弦相似度 + 计算：

1. LLM Call — 从答案反推生成问题（默认跑 3 次 strictness），同时判断答案是否"敷衍"（noncommittal）：
   "Generate a question for the given answer and Identify if answer is noncommittal..."
2. Embedding 计算：对原始问题和 LLM 生成的多个问题分别做 embedding，算余弦相似度
3. 计算：score = mean(余弦相似度)，如果所有生成结果都是敷衍型则 score = 0

#### ContextPrecision（上下文精确度）

逐条 LLM 调用 + Average Precision 计算：

1. LLM Call（对每个检索到的上下文片段各调一次）— 判断该上下文对回答是否有用（verdict 0/1）：
   "Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as 1 if useful and 0 if not..."
2. 计算：经典 IR 的 Average Precision 公式：
   AP = Σ(precision@k × verdict[k]) / Σ(verdicts)
   只在 verdict=1 的位置贡献，权重由该位置的 precision 决定 → 衡量相关文档是否排在前面

#### ContextRecall（上下文召回率）

一次 LLM 调用 + 计算：

1. LLM Call — 逐句判断标准答案的每句话能否归因于检索上下文（attributed 0/1）：
   "Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not..."
2. 计算：score = 可归因的句子数 / 总句子数

### 确定性检索指标（无需 LLM，秒级出结果）

通过 embedding 余弦相似度判断 retrieved chunk 是否命中 reference_context。

#### Hit Rate

检索结果中是否至少有一个 chunk 命中了参考上下文。值域 [0, 1]。

- 1 = 命中，0 = 未命中
- 判定条件：retrieved chunk 与 reference_context 的余弦相似度 ≥ threshold（默认 0.7）

#### MRR（Mean Reciprocal Rank）

第一个命中参考上下文的 chunk 的排名倒数。值域 [0, 1]。

- MRR = 1 表示正确 chunk 排在第 1 位
- MRR = 0.5 表示正确 chunk 排在第 2 位
- 衡量检索的**排序质量**

#### Recall

被命中的参考上下文数 / 总参考上下文数。值域 [0, 1]。

- 衡量检索的**覆盖度**
- 适用于 reference_contexts 有多条的场景

---

## 指标汇总

| 指标 | 类型 | 需要 LLM | 需要 Embedding | 衡量 | 计算方式 |
|------|------|---------|---------------|------|---------|
| Faithfulness | ragas | ✅ 2次 | ❌ | 生成质量 | faithful数/总句数 |
| AnswerRelevancy | ragas | ✅ 1×3次 | ✅ 余弦相似度 | 生成质量 | mean(cos_sim) |
| ContextPrecision | ragas | ✅ N次 | ❌ | 检索排序 | Average Precision |
| ContextRecall | ragas | ✅ 1次 | ❌ | 检索覆盖 | 可归因数/总句数 |
| Hit Rate | 确定性 | ❌ | ✅ | 检索命中 | 0/1 |
| MRR | 确定性 | ❌ | ✅ | 检索排序 | 1/rank |
| Recall | 确定性 | ❌ | ✅ | 检索覆盖 | 命中数/总数 |

---

## 结果文件格式

```json
{
  "summary": {
    "faithfulness": 0.70,
    "hit_rate": 0.84,
    "mrr": 0.62,
    "recall": 0.58
  },
  "group_summary": {
    "baseline":       {"faithfulness": 0.85, "hit_rate": 1.0, ...},
    "query_rewrite":  {"faithfulness": 0.50, "hit_rate": 0.33, ...},
    "__all__":        {"faithfulness": 0.70, "hit_rate": 0.84, ...}
  },
  "samples": [
    {
      "user_input": "...",
      "optimization_target": "bm25_hybrid",
      "difficulty": "simple_fact",
      "faithfulness": 0.67,
      "hit_rate": 1.0,
      "mrr": 0.5,
      "recall": 1.0,
      "response": "...",
      "retrieved_contexts": [...]
    }
  ]
}
```

- **summary**: 全局指标均值
- **group_summary**: 按 `optimization_target` 分组的指标均值
- **samples**: 逐条评估结果

---

## 指标低分根因诊断

每个指标低分时，可能涉及 **数据入库、检索、生成** 等几个层面的原因。

### hit_rate 低（检索没找到相关 chunk）

| 层面 | 根因 | 表现 | 优化动作 |
|------|------|------|---------|
| 数据 | 嵌入模型与领域不匹配 | 通用模型对专业术语（危化品编号、法规条文）编码不准 | 换领域微调嵌入模型；或增加 BM25 补偿 |
| 数据 | 数据未入库/遗漏 | 文档更新后未重新入库，或部分文档未入库 | 建立增量入库流程；入库前校验文件完整性 |
| 数据 | 嵌入模型不一致 | 入库用模型 A，检索用模型 B，向量空间不对齐 | 确保入库和检索使用同一个嵌入模型 |
| 检索 | 口语化 query 与正式表述鸿沟 | "95号文是干嘛的" 检索不到 "安监总管三〔2011〕95号" | 加查询改写模块（LLM 改写/扩展 query 后再检索） |
| 检索 | 编码/编号类 query | `CBCS-WH006` 在向量空间几乎不可区分 | 依赖 BM25 全文精确匹配；确保 BM25 权重足够大 |
| 检索 | 查询歧义/多意图 | 一个 query 含多个子问题，检索只能匹配其中一个 | 查询分解：拆成多个子 query 分别检索再合并 |

### mrr 低（相关 chunk 排名靠后）

| 层面 | 根因 | 表现 | 优化动作 |
|------|------|------|---------|
| 数据 | 分块切断关键信息 | 否定条件被切到相邻 chunk，检索到的不完整 | 增大 chunk_overlap；或用语义分块 |
| 数据 | 嵌入模型与领域不匹配 | 专业术语向量区分度低，相似 chunk 排序混乱 | 换领域微调嵌入模型 |
| 数据 | BM25 索引缺失 | 集合未启用 `enable_bm25`，关键词精确匹配不可用 | 入库时确保 `enable_bm25=True` |
| 检索 | 无 reranker | 同一文档多个语义相近 chunk，最重要的排不到前面 | 开启 reranker（`MAAS_RERANK_ENABLED=true`） |
| 检索 | BM25 权重为 0 | hybrid 模式退化为纯向量检索 | 确保 BM25 weight > 0（当前已修复为 0.5） |
| 检索 | 向量索引参数不当 | nprobe 太小导致 ANN 漏掉最近邻 | 调大 nprobe（当前 16，可试 32） |

### recall 低（部分参考上下文未被检索到）

| 层面 | 根因 | 表现 | 优化动作 |
|------|------|------|---------|
| 数据 | 数据未入库/遗漏 | 部分文档或段落未入库 | 建立增量入库流程；入库前校验完整性 |
| 检索 | top_k 太小 | 多条相关 chunk 只返回了部分 | 增大 top_k；或用递归检索（先粗后精） |
| 检索 | 跨文档检索不足 | 需要同时检索多个文档（如跨法规对比），但只返回单文档 chunk | 增大 top_k；多 collection 并行检索；查询路由 |

### faithfulness 低（答案无法从检索上下文推导）

| 层面 | 根因 | 表现 | 优化动作 |
|------|------|------|---------|
| 数据 | 文档解析乱码 | chunk 混入 `炆炆馆馆`、`Root Entry` 等噪声 | 用 COM 自动化或 olefile 替换 `_binary_extract`，重新入库 |
| 数据 | 分块切断关键信息 | 否定条件、跨句关联被切到相邻 chunk | 增大 chunk_overlap（50→100）；或用语义分块 |
| 数据 | 枚举/公式被截断 | 列表、公式在 chunk 边界处截断 | 调优 chunk_size；或用结构感知分块（按标题/列表项切分） |
| 检索 | 跨文档检索不足 | 多跳问题需要两个文档的信息，只检索到一个 | 增大 top_k；多 collection 并行检索 |
| 生成 | LLM 幻觉 | 答案包含检索结果未提及的细节 | 强化 prompt（"仅基于参考资料回答"）；降低 temperature；加自检 |
| 生成 | chunk 排序影响注意力 | 最相关的 chunk 排在后面，LLM 注意力被无关 chunk 分散 | reranker 将最相关 chunk 排到前面 |
| 生成 | 上下文窗口溢出 | top_k 个 chunk 总长度超过 LLM 上下文窗口被截断 | 控制 top_k × chunk_size；或用长上下文模型 |

### answer_relevancy 低（答案与问题相关性弱）

| 层面 | 根因 | 表现 | 优化动作 |
|------|------|------|---------|
| 检索 | 口语化 query 与正式表述鸿沟 | 检索到无关 chunk，LLM 被误导 | 加查询改写模块 |
| 生成 | prompt 对"无法回答"引导不足 | 超出范围的问题 LLM 仍编造答案 | prompt 加"如果资料中没有相关信息，请明确说明无法回答" |
| 生成 | 生成模型能力不足 | 小模型指令遵循差，容易跑题或遗漏 | 换用更强的生成模型；或用 CoT prompt |
| 生成 | 答案格式不稳定 | 问列表答段落，问是非答长文 | prompt 中指定输出格式 |

### context_precision 低（无关 chunk 排在前面）

| 层面 | 根因 | 表现 | 优化动作 |
|------|------|------|---------|
| 数据 | 文档解析乱码 | 含乱码的 chunk 被 judge 判为无用 | 修复解析器，重新入库 |
| 数据 | BM25 索引缺失 | 关键词精确匹配不可用，无法提升相关 chunk 排名 | 入库时确保 `enable_bm25=True` |
| 数据 | 重复/冗余 chunk | 同一内容多次入库，检索结果中出现重复 | 入库时去重（按 chunk_id 或 content hash） |
| 检索 | 无 reranker | 语义相近但重要性不同的 chunk 无法区分排序 | 开启 reranker |
| 检索 | BM25 权重为 0 | hybrid 退化为纯向量检索 | 确保 BM25 weight > 0 |
| 检索 | 查询歧义/多意图 | 部分子问题匹配到无关 chunk | 查询分解 |

### context_recall 低（标准答案无法归因于检索上下文）

| 层面 | 根因 | 表现 | 优化动作 |
|------|------|------|---------|
| 数据 | 分块切断关键信息 | chunk 边界切断了关键信息（否定条件、跨句关联） | 增大 chunk_overlap；父子分块 |
| 检索 | 口语化 query 匹配不上 | 口语化查询检索不到正式表述的 chunk | 加查询改写模块 |

---

### 诊断决策树

```
1. 看 hit_rate
   ├─ 低 → 检索层问题
   │     ├─ query 与文档表述鸿沟大？ → query_rewrite
   │     ├─ 编码/关键词匹配不上？    → bm25_hybrid 权重
   │     ├─ 嵌入模型领域不匹配？    → 换嵌入模型
   │     └─ 数据未入库/遗漏？       → 检查入库完整性
   └─ 高 → 检索命中OK，往下看

2. 看 mrr
   ├─ 低 → 排序问题
   │     ├─ 相似 chunk 无法区分？    → reranker
   │     ├─ BM25 权重不够？         → 调大 BM25 weight
   │     └─ ANN 索引精度不够？      → 调大 nprobe
   └─ 高 → 排序OK，往下看

3. 看 faithfulness
   ├─ 低 → 生成层问题
   │     ├─ chunk 内容有乱码？      → 修复解析器，重新入库
   │     ├─ chunk 信息被切断？      → 增大 chunk_overlap / chunk_size
   │     ├─ LLM 幻觉？             → 强化 prompt / 换模型
   │     └─ 无关 chunk 分散注意力？ → reranker 排到前面
   └─ 高 → 生成OK，系统健康
```

---

## 前后对比输出示例

```
============================================================
  评估结果前后对比
============================================================

【指标变化】
metric                      before     after     delta
------------------------------------------------------
faithfulness                0.7018    0.8500   +0.1482
hit_rate                    0.5789    0.8421   +0.2632
mrr                         0.4211    0.6842   +0.2631
recall                      0.3158    0.6316   +0.3158

  ✅ 改善: faithfulness, hit_rate, mrr, recall
  ❌ 退步: 无
  ➖ 持平: 无

【按优化目标分组变化】
  baseline             无明显变化
  bm25_hybrid          hit_rate ↑0.333, mrr ↑0.250
  query_rewrite        hit_rate ↑0.333
  ...
```
