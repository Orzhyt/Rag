"""评估运行器 — 执行 RAG 评估并输出结果。

用法:
    # 运行评估
    python -m evaluation.runner --mode evaluate --dataset evaluation/test_dataset_optimized.jsonl

    # 指定指标和检索参数
    python -m evaluation.runner --mode evaluate --dataset evaluation/test_dataset.jsonl --top-k 5 --search-mode hybrid

    # 前后对比
    python -m evaluation.runner --mode compare evaluation/results/before.json evaluation/results/after.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.logger import setup_logger
from evaluation.compare import aggregate_by_target, compare_results, print_aggregation, print_comparison
from evaluation.dataset import build_eval_dataset, load_dataset, save_results
from evaluation.llm import get_ragas_embeddings, get_ragas_llm
from evaluation.rag_pipeline import RAGPipeline
from evaluation.retrieval_metrics import compute_retrieval_metrics_batch
from milvus.client import MilvusClient
from retrieval.embedder import Embedder
from retrieval.profile import DEFAULT_RAG_PROFILE
from retrieval.reranker import Reranker
from retrieval.service import MilvusService

logger = setup_logger("evaluation.runner")

# ragas LLM-judge 指标名 → 构建函数
_RAGAS_METRIC_MAP = {
    "faithfulness": lambda: __import__("ragas.metrics", fromlist=["Faithfulness"]).Faithfulness(),
    "answer_relevancy": lambda: __import__("ragas.metrics", fromlist=["AnswerRelevancy"]).AnswerRelevancy(),
    "context_precision": lambda: __import__("ragas.metrics", fromlist=["ContextPrecision"]).ContextPrecision(),
    "context_recall": lambda: __import__("ragas.metrics", fromlist=["ContextRecall"]).ContextRecall(),
}

# 确定性检索指标名
_RETRIEVAL_METRIC_NAMES = {"hit_rate", "mrr", "recall"}

# 默认全部指标
DEFAULT_METRICS = list(_RAGAS_METRIC_MAP.keys()) + list(_RETRIEVAL_METRIC_NAMES)


def init_services(
    database: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> tuple:
    """初始化评估所需的服务，返回 (MilvusService, EmbeddingModel)"""
    embedder = Embedder()
    _ = embedder.dim  # 触发模型加载

    client = MilvusClient(database=database, collection_name=collection_name or "rag_chunks")
    client.connect()

    reranker = None
    if os.getenv("MAAS_RERANK_ENABLED", "false").lower() in ("true", "1", "yes"):
        reranker = Reranker()

    service = MilvusService(
        client=client, embedder=embedder, profile=DEFAULT_RAG_PROFILE,
        reranker=reranker,
    )
    return service, embedder


def get_metrics(names: List[str], ragas_llm=None, ragas_embeddings=None) -> List:
    """构建 ragas 指标对象列表，并设置 llm/embeddings。

    仅处理 _RAGAS_METRIC_MAP 中的指标；检索指标在 run_evaluation 中单独计算。
    """
    metrics = []
    for name in names:
        if name not in _RAGAS_METRIC_MAP:
            continue
        metric = _RAGAS_METRIC_MAP[name]()
        if ragas_llm is not None and hasattr(metric, "llm"):
            metric.llm = ragas_llm
        if ragas_embeddings is not None and hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings
        metrics.append(metric)
    return metrics


def run_evaluation(
    dataset_path: str,
    metrics: Optional[List[str]] = None,
    top_k: int = 5,
    search_mode: str = "hybrid",
    output_dir: str = "evaluation/results",
    collection_names: Optional[List[str]] = None,
    database: Optional[str] = None,
    similarity_threshold: float = 0.7,
) -> str:
    """执行完整评估：ragas LLM-judge 指标 + 确定性检索指标。

    Args:
        dataset_path: 测试数据集 JSONL 路径
        metrics: 指标名列表，None 使用默认全部指标
        top_k: 检索返回的文档数
        search_mode: 检索模式 ("hybrid" / "vector")
        output_dir: 结果输出目录
        collection_names: 检索的 Milvus 集合名列表
        database: Milvus 数据库名，None 使用环境变量或默认
        similarity_threshold: 检索指标命中阈值

    Returns:
        结果文件路径
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    # 分离 ragas 指标和检索指标
    ragas_metric_names = [m for m in metrics if m in _RAGAS_METRIC_MAP]
    retrieval_metric_names = [m for m in metrics if m in _RETRIEVAL_METRIC_NAMES]

    # 1. 初始化服务
    logger.info("初始化服务 (database=%s)...", database or "default")
    milvus_service, embedder = init_services(database=database)

    # 2. 加载测试数据集
    logger.info("加载测试数据集: %s", dataset_path)
    records = load_dataset(dataset_path)
    logger.info("共 %d 条测试样本", len(records))

    # 3. 构建 RAG pipeline 并执行检索+生成
    pipeline = RAGPipeline(milvus_service)
    logger.info("执行 RAG 流程 (top_k=%d, mode=%s)...", top_k, search_mode)
    eval_dataset, debug_info = build_eval_dataset(
        records, pipeline, top_k=top_k, mode=search_mode,
        collection_names=collection_names,
    )

    # 4. 计算 ragas LLM-judge 指标
    all_results: List[Dict[str, Any]] = [{} for _ in records]

    if ragas_metric_names:
        logger.info("计算 ragas 指标: %s", ragas_metric_names)
        ragas_llm = get_ragas_llm()
        ragas_embeddings = get_ragas_embeddings(embedder)
        ragas_metrics = get_metrics(ragas_metric_names, ragas_llm, ragas_embeddings)

        from ragas import evaluate as ragas_evaluate
        eval_result = ragas_evaluate(
            dataset=eval_dataset,
            metrics=ragas_metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            show_progress=True,
        )

        # eval_result.scores 是 List[Dict[str, float]]，每条样本一个 dict
        for i, score_dict in enumerate(eval_result.scores):
            all_results[i].update(score_dict)

        logger.info("ragas 指标计算完成")
    else:
        logger.info("跳过 ragas 指标（未指定）")

    # 5. 计算确定性检索指标
    if retrieval_metric_names:
        logger.info("计算检索指标: %s (threshold=%.2f)", retrieval_metric_names, similarity_threshold)
        samples_with_contexts = []
        for i, rec in enumerate(records):
            retrieved = debug_info[i].get("retrieved_contexts", [])
            reference = rec.get("reference_contexts", [])
            samples_with_contexts.append({
                "retrieved_contexts": retrieved,
                "reference_contexts": reference,
            })

        retrieval_scores = compute_retrieval_metrics_batch(
            embedder, samples_with_contexts, threshold=similarity_threshold,
        )
        for i, rscores in enumerate(retrieval_scores):
            all_results[i].update(rscores)

        logger.info("检索指标计算完成")
    else:
        logger.info("跳过检索指标（未指定）")

    # 6. 合并元数据（从原始 records 中保留 optimization_target 等）
    _metadata_keys = ["optimization_target", "difficulty", "optimization_note", "source"]
    for i, rec in enumerate(records):
        all_results[i]["user_input"] = rec.get("user_input", "")
        for mk in _metadata_keys:
            if mk in rec:
                all_results[i][mk] = rec[mk]

    # 7. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_stem = Path(dataset_path).stem
    experiment_name = f"eval_{dataset_stem}_{timestamp}"

    all_metric_names = ragas_metric_names + retrieval_metric_names
    result_path = save_results(
        all_results, all_metric_names, output_dir, experiment_name,
        debug_info=debug_info,
    )
    logger.info("结果已保存到: %s", result_path)

    # 8. 打印摘要
    _print_summary(all_results, all_metric_names)

    return result_path


def _print_summary(results: List[Dict[str, Any]], metric_names: List[str]) -> None:
    """打印评估摘要和分组报告。"""
    print("\n" + "=" * 60)
    print("  评估结果摘要")
    print("=" * 60)

    # 总体均值
    print("\n【总体指标】")
    for mk in metric_names:
        values = [r[mk] for r in results if r.get(mk) is not None]
        if values:
            avg = sum(values) / len(values)
            print(f"  {mk:<24} {avg:.4f}  (n={len(values)})")
        else:
            print(f"  {mk:<24} N/A")

    # 分组报告
    agg = aggregate_by_target(results, metric_keys=metric_names)
    if len(agg) > 1:  # 有多个 target 或有 __all__
        print("\n【按优化目标分组】")
        print_aggregation(agg)


def main():
    parser = argparse.ArgumentParser(description="RAG 评估运行器")
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # evaluate 子命令
    eval_parser = subparsers.add_parser("evaluate", help="运行评估")
    eval_parser.add_argument("--dataset", required=True, help="测试数据集 JSONL 路径")
    eval_parser.add_argument(
        "--metrics", default=None,
        help=f"指标列表（逗号分隔），默认: {','.join(DEFAULT_METRICS)}",
    )
    eval_parser.add_argument("--top-k", type=int, default=5, help="检索返回文档数 (默认 5)")
    eval_parser.add_argument(
        "--search-mode", default="hybrid", choices=["hybrid", "vector"],
        help="检索模式 (默认 hybrid)",
    )
    eval_parser.add_argument("--output-dir", default="evaluation/results", help="结果输出目录")
    eval_parser.add_argument("--database", default=None, help="Milvus 数据库名 (默认使用环境变量)")
    eval_parser.add_argument("--collections", nargs="+", default=None, help="Milvus 集合名列表")
    eval_parser.add_argument(
        "--similarity-threshold", type=float, default=0.7,
        help="检索指标命中阈值 (默认 0.7)",
    )

    # compare 子命令
    cmp_parser = subparsers.add_parser("compare", help="前后对比")
    cmp_parser.add_argument("before", help="优化前评估结果 JSON 路径")
    cmp_parser.add_argument("after", help="优化后评估结果 JSON 路径")

    args = parser.parse_args()

    if args.mode == "evaluate":
        metrics = args.metrics.split(",") if args.metrics else None
        run_evaluation(
            dataset_path=args.dataset,
            metrics=metrics,
            top_k=args.top_k,
            search_mode=args.search_mode,
            output_dir=args.output_dir,
            collection_names=args.collections,
            database=args.database,
            similarity_threshold=args.similarity_threshold,
        )
    elif args.mode == "compare":
        comparison = compare_results(args.before, args.after)
        print_comparison(comparison)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
