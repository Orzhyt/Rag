"""评估运行器 — 执行 RAG 评估并输出结果。

用法:
    # 运行评估
    python -m evaluation.runner evaluate --dataset evaluation/test_dataset_optimized.jsonl

    # 指定指标和检索参数
    python -m evaluation.runner evaluate --dataset evaluation/test_dataset.jsonl --top-k 5

    # 前后对比
    python -m evaluation.runner compare evaluation/results/before.json evaluation/results/after.json
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.logger import setup_logger
from evaluation.compare import compare_results, print_comparison
from evaluation.dataset import build_eval_dataset, load_dataset, save_results
from evaluation.llm import get_ragas_embeddings, get_ragas_llm
from llm.client import LLMClient
from llm.service import RAGChatService
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

# 默认全部指标
DEFAULT_METRICS = list(_RAGAS_METRIC_MAP.keys())


def init_services(
    database: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> tuple:
    """初始化评估所需的服务，返回 (RAGChatService, EmbeddingModel)"""
    embedder = Embedder()
    _ = embedder.dim  # 触发模型加载

    client = MilvusClient(database=database, collection_name=collection_name or "rag_chunks")
    client.connect()

    reranker = None
    if os.getenv("MAAS_RERANK_ENABLED", "false").lower() in ("true", "1", "yes"):
        reranker = Reranker()

    milvus_service = MilvusService(
        client=client, embedder=embedder, profile=DEFAULT_RAG_PROFILE,
        reranker=reranker,
    )

    llm_client = LLMClient()
    rag_chat_service = RAGChatService(llm_client=llm_client, milvus_service=milvus_service)

    return rag_chat_service, embedder


def get_metrics(names: List[str], ragas_llm=None, ragas_embeddings=None) -> List:
    """构建 ragas 指标对象列表，并设置 llm/embeddings。"""
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
    output_dir: str = "evaluation/results",
    collection_names: Optional[List[str]] = None,
    database: Optional[str] = None,
) -> str:
    """执行完整评估：ragas LLM-judge 指标。

    Args:
        dataset_path: 测试数据集 JSONL 路径
        metrics: 指标名列表，None 使用默认全部指标
        top_k: 检索返回的文档数
        output_dir: 结果输出目录
        collection_names: 检索的 Milvus 集合名列表
        database: Milvus 数据库名，None 使用环境变量或默认

    Returns:
        结果文件路径
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    # 过滤有效的 ragas 指标
    ragas_metric_names = [m for m in metrics if m in _RAGAS_METRIC_MAP]

    # 1. 初始化服务
    logger.info("初始化服务 (database=%s)...", database or "default")
    rag_chat_service, embedder = init_services(database=database)

    # 2. 加载测试数据集
    logger.info("加载测试数据集: %s", dataset_path)
    records = load_dataset(dataset_path)
    logger.info("共 %d 条测试样本", len(records))

    # 3. 通过接口侧执行 RAG 流程
    logger.info("执行 RAG 流程 (top_k=%d)...", top_k)
    eval_dataset, debug_info = build_eval_dataset(
        records, rag_chat_service, top_k=top_k,
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

    # 5. 合并元数据（从原始 records 中保留 optimization_target 等）
    _metadata_keys = ["optimization_target", "optimization_note", "source"]
    for i, rec in enumerate(records):
        all_results[i]["user_input"] = rec.get("user_input", "")
        for mk in _metadata_keys:
            if mk in rec:
                all_results[i][mk] = rec[mk]

    # 6. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_stem = Path(dataset_path).stem
    experiment_name = f"eval_{dataset_stem}_{timestamp}"

    all_metric_names = ragas_metric_names
    result_path = save_results(
        all_results, all_metric_names, output_dir, experiment_name,
        debug_info=debug_info,
    )
    logger.info("结果已保存到: %s", result_path)

    # 7. 打印摘要
    _print_summary(all_results, all_metric_names)

    return result_path


def _print_summary(results: List[Dict[str, Any]], metric_names: List[str]) -> None:
    """打印评估摘要。"""
    print("\n" + "=" * 60)
    print("  评估结果摘要")
    print("=" * 60)

    for mk in metric_names:
        values = [r[mk] for r in results if r.get(mk) is not None and not math.isnan(r.get(mk))]
        if values:
            avg = sum(values) / len(values)
            print(f"  {mk:<24} {avg:.4f}  (n={len(values)})")
        else:
            print(f"  {mk:<24} N/A")


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
    eval_parser.add_argument("--output-dir", default="evaluation/results", help="结果输出目录")
    eval_parser.add_argument("--database", default=None, help="Milvus 数据库名 (默认使用环境变量)")
    eval_parser.add_argument("--collections", nargs="+", default=None, help="Milvus 集合名列表")
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
            output_dir=args.output_dir,
            collection_names=args.collections,
            database=args.database,
        )
    elif args.mode == "compare":
        comparison = compare_results(args.before, args.after)
        print_comparison(comparison)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
