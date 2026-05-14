import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

from evaluation.rag_pipeline import RAGPipeline

logger = logging.getLogger("evaluation.dataset")


def load_dataset(path: str) -> List[Dict]:
    """从 JSONL 文件加载测试数据集

    每行格式: {"user_input": "...", "reference": "...", "reference_contexts": ["..."]}
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                records.append(json.loads(line))
    return records


def build_eval_dataset(
    records: List[Dict],
    pipeline: RAGPipeline,
    top_k: int = 5,
    mode: str = "hybrid",
    collection_names: Optional[List[str]] = None,
) -> EvaluationDataset:
    """对每条测试数据执行 RAG 流程，构建 ragas EvaluationDataset

    records 中的 user_input 会通过 pipeline.invoke() 获取 response 和 retrieved_contexts。
    reference 和 reference_contexts 从 records 中直接取。
    """
    samples = []
    for i, rec in enumerate(records):
        query = rec["user_input"]
        logger.info("Query %d/%d: %s", i + 1, len(records), query[:50])
        answer, contexts = pipeline.invoke(
            query, top_k=top_k, mode=mode, collection_names=collection_names,
        )
        logger.info("Query %d/%d done (retrieved %d contexts)", i + 1, len(records), len(contexts))

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference=rec.get("reference"),
            reference_contexts=rec.get("reference_contexts"),
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def save_results(results: List[Dict], metrics: List[str], output_dir: str, experiment_name: str) -> str:
    """保存评估结果到 JSON 文件，汇总分数在前，逐条分数在后"""
    summary = {}
    for metric_name in metrics:
        values = [r[metric_name] for r in results if r.get(metric_name) is not None]
        if values:
            summary[metric_name] = round(sum(values) / len(values), 4)
        else:
            summary[metric_name] = None

    output = {
        "summary": summary,
        "samples": results,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / f"{experiment_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return str(file_path)
