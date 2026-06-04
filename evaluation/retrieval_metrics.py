"""确定性检索指标模块 — 无需 LLM 调用，秒级出结果。

通过 embedding 余弦相似度判断 retrieved chunk 是否命中 reference_context，
计算 Hit Rate、MRR、Recall 三个检索指标。

用法:
    from evaluation.retrieval_metrics import compute_retrieval_metrics_batch
    scores = compute_retrieval_metrics_batch(embedder, samples, threshold=0.7)
"""

import logging
from typing import Dict, List

from retrieval.embedder import Embedder

logger = logging.getLogger("evaluation.retrieval_metrics")


def compute_similarity_matrix(
    embedder: Embedder,
    retrieved_contexts: List[str],
    reference_contexts: List[str],
) -> List[List[float]]:
    """计算 retrieved × reference 的余弦相似度矩阵。

    利用 Embedder.encode() 返回的 L2 归一化向量，
    点积即余弦相似度，无需额外归一化。

    Returns:
        similarity_matrix[i][j] = cos_sim(retrieved[i], reference[j])
    """
    if not retrieved_contexts or not reference_contexts:
        return []

    all_texts = retrieved_contexts + reference_contexts
    all_vecs = embedder.encode(all_texts)

    n_ret = len(retrieved_contexts)
    ret_vecs = all_vecs[:n_ret]
    ref_vecs = all_vecs[n_ret:]

    matrix: List[List[float]] = []
    for rv in ret_vecs:
        row: List[float] = []
        for fv in ref_vecs:
            # 点积 = 余弦相似度（因为向量已 L2 归一化）
            sim = sum(a * b for a, b in zip(rv, fv))
            row.append(sim)
        matrix.append(row)

    return matrix


def compute_retrieval_metrics(
    embedder: Embedder,
    retrieved_contexts: List[str],
    reference_contexts: List[str],
    threshold: float = 0.7,
) -> Dict[str, float]:
    """计算单条查询的检索指标。

    Args:
        embedder: 嵌入模型，用于计算语义相似度
        retrieved_contexts: 检索返回的上下文列表（按排名顺序）
        reference_contexts: 黄金标准上下文列表
        threshold: 余弦相似度阈值，>= threshold 视为命中

    Returns:
        {"hit_rate": 0/1, "mrr": 1/rank, "recall": hit_count/total}
    """
    if not reference_contexts:
        return {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0}
    if not retrieved_contexts:
        return {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0}

    matrix = compute_similarity_matrix(embedder, retrieved_contexts, reference_contexts)
    if not matrix:
        return {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0}

    # 对每个 reference context，找出所有 retrieved 中与它的最大相似度
    n_ref = len(reference_contexts)
    max_sim_per_ref: List[float] = [0.0] * n_ref
    for i, row in enumerate(matrix):
        for j, sim in enumerate(row):
            if sim > max_sim_per_ref[j]:
                max_sim_per_ref[j] = sim

    # 哪些 reference contexts 被命中
    hit_refs = [j for j in range(n_ref) if max_sim_per_ref[j] >= threshold]
    hit_count = len(hit_refs)

    # Hit Rate: 至少一个 reference context 被命中
    hit_rate = 1.0 if hit_count > 0 else 0.0

    # MRR: 第一个命中任何 reference context 的 retrieved 排名的倒数
    mrr = 0.0
    for i, row in enumerate(matrix):
        if any(sim >= threshold for sim in row):
            mrr = 1.0 / (i + 1)
            break

    # Recall: 被命中的 reference contexts 占总 reference contexts 的比例
    recall = hit_count / n_ref

    return {"hit_rate": hit_rate, "mrr": mrr, "recall": recall}


def compute_retrieval_metrics_batch(
    embedder: Embedder,
    samples: List[Dict],
    threshold: float = 0.7,
) -> List[Dict[str, float]]:
    """批量计算检索指标。

    Args:
        embedder: 嵌入模型
        samples: 每个元素需包含 "retrieved_contexts" 和 "reference_contexts" 键
        threshold: 余弦相似度阈值

    Returns:
        每条样本的检索指标字典列表
    """
    results: List[Dict[str, float]] = []
    for i, sample in enumerate(samples):
        retrieved = sample.get("retrieved_contexts", [])
        reference = sample.get("reference_contexts", [])
        metrics = compute_retrieval_metrics(embedder, retrieved, reference, threshold)
        results.append(metrics)
        logger.debug(
            "Sample %d: hit_rate=%.2f, mrr=%.4f, recall=%.2f",
            i, metrics["hit_rate"], metrics["mrr"], metrics["recall"],
        )
    return results
