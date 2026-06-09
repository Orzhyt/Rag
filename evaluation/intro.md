# 评估模块使用指南

## 快速开始

```bash
# 完整评估
python -m evaluation.runner evaluate \
  --dataset evaluation/test_dataset_optimized.jsonl \
  --database cjc \
  --collections test

# 前后对比
python -m evaluation.runner compare \
  evaluation/results/eval_before.json \
  evaluation/results/eval_after.json
```

## CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 测试数据集 JSONL 路径 | （必填） |
| `--metrics` | 指标列表（逗号分隔） | `faithfulness,answer_relevancy,context_precision,context_recall` |
| `--top-k` | 检索返回文档数 | 5 |
| `--database` | Milvus 数据库名 | 环境变量 `MILVUS_DATABASE` |
| `--collections` | Milvus 集合名列表 | 默认集合 |
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

---

## 指标汇总

| 指标 | 类型 | 需要 LLM | 需要 Embedding | 衡量 | 计算方式 |
|------|------|---------|---------------|------|---------|
| Faithfulness | ragas | ✅ 2次 | ❌ | 生成质量 | faithful数/总句数 |
| AnswerRelevancy | ragas | ✅ 1×3次 | ✅ 余弦相似度 | 生成质量 | mean(cos_sim) |
| ContextPrecision | ragas | ✅ N次 | ❌ | 检索排序 | Average Precision |
| ContextRecall | ragas | ✅ 1次 | ❌ | 检索覆盖 | 可归因数/总句数 |

---

## 结果文件格式

```json
{
  "summary": {
    "faithfulness": 0.70,
    "answer_relevancy": 0.85,
    "context_precision": 0.65,
    "context_recall": 0.72
  },
  "group_summary": {
    "baseline":       {"faithfulness": 0.85, "answer_relevancy": 0.90, ...},
    "query_rewrite":  {"faithfulness": 0.50, "answer_relevancy": 0.60, ...},
    "__all__":        {"faithfulness": 0.70, "answer_relevancy": 0.85, ...}
  },
  "samples": [
    {
      "user_input": "...",
      "optimization_target": "bm25_hybrid",
      "faithfulness": 0.67,
      "answer_relevancy": 0.80,
      "context_precision": 0.55,
      "context_recall": 0.70,
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
1. 看 faithfulness
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
answer_relevancy            0.7500    0.8800   +0.1300
context_precision           0.6200    0.7900   +0.1700
context_recall              0.5800    0.7200   +0.1400

  ✅ 改善: faithfulness, answer_relevancy, context_precision, context_recall
  ❌ 退步: 无
  ➖ 持平: 无

【按优化目标分组变化】
  baseline             无明显变化
  bm25_hybrid          faithfulness ↑0.200, context_precision ↑0.150
  query_rewrite        answer_relevancy ↑0.180
  ...
```
