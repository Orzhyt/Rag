# RAG 评估指标说明

## Faithfulness（忠实度）

两次 LLM 调用 + 一次计算：

1. LLM Call 1 — 把答案拆成原子语句：
   "Given a question and answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements..."
2. LLM Call 2 — 逐句判断能否从上下文推断出来（NLI 判定 0/1）：
   "Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not..."
3. 计算：score = 可推断的语句数 / 总语句数

## AnswerRelevancy（答案相关性）

LLM 调用 + Embedding 余弦相似度 + 计算：

1. LLM Call — 从答案反推生成问题（默认跑 3 次 strictness），同时判断答案是否"敷衍"（noncommittal）：
   "Generate a question for the given answer and Identify if answer is noncommittal..."
2. Embedding 计算：对原始问题和 LLM 生成的多个问题分别做 embedding，算余弦相似度
3. 计算：score = mean(余弦相似度)，如果所有生成结果都是敷衍型则 score = 0

## ContextPrecision（上下文精确度）

逐条 LLM 调用 + Average Precision 计算：

1. LLM Call（对每个检索到的上下文片段各调一次）— 判断该上下文对回答是否有用（verdict 0/1）：
   "Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as 1 if useful and 0 if not..."
2. 计算：经典 IR 的 Average Precision 公式：
   AP = Σ(precision@k × verdict[k]) / Σ(verdicts)
   只在 verdict=1 的位置贡献，权重由该位置的 precision 决定 → 衡量相关文档是否排在前面

## ContextRecall（上下文召回率）

一次 LLM 调用 + 计算：

1. LLM Call — 逐句判断标准答案的每句话能否归因于检索上下文（attributed 0/1）：
   "Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not..."
2. 计算：score = 可归因的句子数 / 总句子数

---

## 汇总

| 指标 | LLM 调用次数 | Prompt 用途 | Embedding | 最终计算公式 |
|------|-------------|-------------|-----------|-------------|
| Faithfulness | 2 | 拆句 + NLI判定 | 无 | faithful数/总句数 |
| AnswerRelevancy | 1×3 | 反推问题+敷衍检测 | 余弦相似度 | mean(cos_sim) |
| ContextPrecision | N(每条上下文1次) | 逐条判有用性 | 无 | Average Precision |
| ContextRecall | 1 | 逐句归因判定 | 无 | 可归因数/总句数 |
