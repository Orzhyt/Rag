import argparse
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from common.logger import setup_logger
from milvus.client import MilvusClient
from milvus.embedder import EmbeddingModel
from retrieval.profile import DEFAULT_RAG_PROFILE
from retrieval.service import MilvusService

from evaluation.dataset import build_eval_dataset, load_dataset, save_results
from evaluation.llm import get_ragas_embeddings, get_ragas_llm
from evaluation.rag_pipeline import RAGPipeline

logger = setup_logger("evaluation.runner")


def get_metrics(names: List[str], ragas_llm, ragas_embeddings):
    """根据名称列表构建 ragas 指标对象"""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    mapping = {
        "faithfulness": lambda: Faithfulness(llm=ragas_llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        "context_precision": lambda: ContextPrecision(llm=ragas_llm),
        "context_recall": lambda: ContextRecall(llm=ragas_llm),
    }
    metrics = []
    for name in names:
        factory = mapping.get(name)
        if factory is None:
            raise ValueError(f"未知指标: {name}，可选: {list(mapping.keys())}")
        metrics.append((name, factory()))
    return metrics


def init_services() -> tuple:
    """初始化 MilvusService 和 EmbeddingModel（独立于 FastAPI，供评估使用）"""
    logger.info("Loading embedding model...")
    embedder = EmbeddingModel()
    _ = embedder.dim
    logger.info("Embedding model loaded, dim=%d", embedder.dim)

    logger.info("Connecting to Milvus...")
    client = MilvusClient()
    client.connect()
    logger.info("Milvus connected")

    service = MilvusService(client=client, embedder=embedder, profile=DEFAULT_RAG_PROFILE)
    return service, embedder


async def _score_sample(metric, sample: Dict[str, Any]) -> float:
    """调用单个指标的 ascore 方法，返回分数"""
    result = await metric.ascore(**sample)
    return float(result.value) if hasattr(result, "value") else float(result)


async def run_evaluation(
    dataset_path: str,
    metrics: List[str],
    top_k: int = 5,
    search_mode: str = "hybrid",
    output_dir: str = "evaluation/results",
    collection_names: Optional[List[str]] = None,
):
    """执行 ragas 评估主流程"""
    # 1. 初始化服务
    milvus_service, embedder = init_services()
    pipeline = RAGPipeline(milvus_service)

    # 2. 加载测试数据集
    logger.info("Loading dataset from %s", dataset_path)
    records = load_dataset(dataset_path)
    logger.info("Loaded %d test queries", len(records))

    # 3. 构建 ragas EvaluationDataset（执行检索+生成）
    logger.info("Running RAG pipeline for each query (mode=%s, top_k=%d, collections=%s)...", search_mode, top_k, collection_names)
    eval_dataset = build_eval_dataset(records, pipeline, top_k=top_k, mode=search_mode, collection_names=collection_names)

    # 4. 配置 ragas LLM 和 Embeddings（复用已加载的 embedder）
    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings(embedder)

    # 5. 构建指标
    ragas_metrics = get_metrics(metrics, ragas_llm, ragas_embeddings)
    logger.info("Metrics: %s", metrics)

    # 6. 逐样本逐指标打分
    logger.info("Running ragas evaluation...")
    all_results: List[Dict[str, Any]] = []

    for i, sample in enumerate(eval_dataset.samples):
        row: Dict[str, Any] = {
            "user_input": sample.user_input,
            "response": sample.response,
            "retrieved_contexts": sample.retrieved_contexts,
            "reference": sample.reference,
            "reference_contexts": sample.reference_contexts,
        }

        for metric_name, metric_obj in ragas_metrics:
            # 按各指标 ascore 签名传参
            if metric_name == "faithfulness":
                args = dict(user_input=sample.user_input, response=sample.response, retrieved_contexts=sample.retrieved_contexts)
            elif metric_name == "answer_relevancy":
                args = dict(user_input=sample.user_input, response=sample.response)
            elif metric_name == "context_precision":
                args = dict(user_input=sample.user_input, reference=sample.reference, retrieved_contexts=sample.retrieved_contexts)
            elif metric_name == "context_recall":
                args = dict(user_input=sample.user_input, retrieved_contexts=sample.retrieved_contexts, reference=sample.reference)
            else:
                raise ValueError(f"Unknown metric: {metric_name}")

            try:
                result = await metric_obj.ascore(**args)
                score = float(result.value) if hasattr(result, "value") else float(result)
            except Exception as e:
                logger.error("Metric %s failed for sample %d: %s", metric_name, i, e)
                score = None

            row[metric_name] = score

        all_results.append(row)
        logger.info("Sample %d/%d scored", i + 1, len(eval_dataset.samples))

    # 7. 保存结果
    experiment_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = save_results(all_results, metrics, output_dir, experiment_name)
    logger.info("Results saved to %s", output_path)

    # 8. 打印汇总
    print("\n" + "=" * 50)
    print("RAG Evaluation Results")
    print("=" * 50)
    for metric_name in metrics:
        values = [r[metric_name] for r in all_results if r.get(metric_name) is not None]
        if values:
            avg = sum(values) / len(values)
            print(f"  {metric_name}: {avg:.4f} (avg over {len(values)} samples)")
        else:
            print(f"  {metric_name}: N/A")
    print("=" * 50)

    return all_results


async def run_generate(
    data_dir: Optional[str] = None,
    from_milvus: bool = False,
    collection_names: Optional[List[str]] = None,
    testset_size: int = 10,
    output_path: str = "evaluation/generated_testset.jsonl",
):
    """生成测试集主流程"""
    from evaluation.testset_generator import (
        generate_testset,
        generate_testset_from_chunks,
        load_chunks_from_milvus,
        load_documents_from_dir,
        save_testset_to_jsonl,
    )

    # 1. 加载文档或切片
    if from_milvus:
        milvus_service, embedder = init_services()
        documents = load_chunks_from_milvus(milvus_service, collection_names)
    elif data_dir:
        documents = load_documents_from_dir(data_dir)
        embedder = None
    else:
        raise ValueError("必须指定 --data-dir 或 --from-milvus")

    if not documents:
        logger.error("没有可用的文档/切片，无法生成测试集")
        return

    # 2. 配置 ragas LLM 和 Embeddings
    if embedder is None:
        logger.info("Loading embedding model for testset generation...")
        embedder = EmbeddingModel()
        _ = embedder.dim

    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings(embedder)

    # 3. 生成测试集
    if from_milvus:
        testset = generate_testset_from_chunks(
            chunks=documents,
            testset_size=testset_size,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )
    else:
        testset = generate_testset(
            documents=documents,
            testset_size=testset_size,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )

    # 4. 保存
    saved_path = save_testset_to_jsonl(testset, output_path)
    print(f"\n测试集已保存到: {saved_path}（{len(testset.samples)} 条）")
    return saved_path


async def run_generate_and_eval(
    data_dir: Optional[str] = None,
    from_milvus: bool = False,
    collection_names: Optional[List[str]] = None,
    testset_size: int = 10,
    metrics: Optional[List[str]] = None,
    top_k: int = 5,
    search_mode: str = "hybrid",
    output_dir: str = "evaluation/results",
):
    """生成测试集 + 评估一条龙"""
    metrics = metrics or ["faithfulness", "answer_relevancy"]

    # 1. 生成测试集
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    testset_path = f"evaluation/generated_testset_{timestamp}.jsonl"

    result = await run_generate(
        data_dir=data_dir,
        from_milvus=from_milvus,
        collection_names=collection_names,
        testset_size=testset_size,
        output_path=testset_path,
    )
    if result is None:
        logger.error("测试集生成失败，跳过评估")
        return

    # 2. 执行评估
    await run_evaluation(
        dataset_path=testset_path,
        metrics=metrics,
        top_k=top_k,
        search_mode=search_mode,
        output_dir=output_dir,
        collection_names=collection_names,
    )


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation with ragas")
    parser.add_argument(
        "--mode",
        default="evaluate",
        choices=["evaluate", "generate", "generate_and_eval"],
        help="运行模式: evaluate=评估(默认), generate=生成测试集, generate_and_eval=生成+评估",
    )

    # 评估模式参数
    parser.add_argument(
        "--dataset",
        help="Path to test dataset JSONL file (evaluate 模式必填)",
    )
    parser.add_argument(
        "--metrics",
        help="Comma-separated metric names: faithfulness,answer_relevancy,context_precision,context_recall",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument(
        "--search-mode", default="hybrid", choices=["hybrid", "vector"],
        help="Search mode: hybrid (vector+BM25) or vector only",
    )
    parser.add_argument("--output-dir", default="evaluation/results", help="Output directory for results")
    parser.add_argument(
        "--collections", default=None,
        help="Comma-separated collection names to search (default: use MilvusClient default)",
    )

    # 生成模式参数
    parser.add_argument(
        "--data-dir", default=None,
        help="文档目录路径（generate 模式: 从原始文档生成测试集）",
    )
    parser.add_argument(
        "--from-milvus", action="store_true",
        help="从 Milvus 已有切片生成测试集（generate 模式）",
    )
    parser.add_argument(
        "--testset-size", type=int, default=10,
        help="生成测试样本数量（generate 模式，默认 10）",
    )
    parser.add_argument(
        "--output", default="evaluation/generated_testset.jsonl",
        help="生成测试集的输出 JSONL 路径",
    )

    args = parser.parse_args()
    collection_names = [c.strip() for c in args.collections.split(",")] if args.collections else None

    if args.mode == "evaluate":
        if not args.dataset:
            parser.error("evaluate 模式需要 --dataset 参数")
        if not args.metrics:
            parser.error("evaluate 模式需要 --metrics 参数")
        metrics = [m.strip() for m in args.metrics.split(",")]
        asyncio.run(run_evaluation(
            dataset_path=args.dataset,
            metrics=metrics,
            top_k=args.top_k,
            search_mode=args.search_mode,
            output_dir=args.output_dir,
            collection_names=collection_names,
        ))

    elif args.mode == "generate":
        if not args.data_dir and not args.from_milvus:
            parser.error("generate 模式需要 --data-dir 或 --from-milvus 参数")
        asyncio.run(run_generate(
            data_dir=args.data_dir,
            from_milvus=args.from_milvus,
            collection_names=collection_names,
            testset_size=args.testset_size,
            output_path=args.output,
        ))

    elif args.mode == "generate_and_eval":
        if not args.data_dir and not args.from_milvus:
            parser.error("generate_and_eval 模式需要 --data-dir 或 --from-milvus 参数")
        metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else None
        asyncio.run(run_generate_and_eval(
            data_dir=args.data_dir,
            from_milvus=args.from_milvus,
            collection_names=collection_names,
            testset_size=args.testset_size,
            metrics=metrics,
            top_k=args.top_k,
            search_mode=args.search_mode,
            output_dir=args.output_dir,
        ))


if __name__ == "__main__":
    main()
