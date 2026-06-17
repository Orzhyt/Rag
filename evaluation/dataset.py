import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

from llm.service import RAGChatService

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
    rag_chat_service: RAGChatService,
    top_k: int = 5,
    collection_names: Optional[List[str]] = None,
) -> tuple:
    """对每条测试数据执行 RAG 流程，构建 ragas EvaluationDataset

    通过 rag_chat_service.chat() 调用接口侧，获取和页面一致的检索结果与生成回答。
    完整 chunk 内容从 source["fields"]["content"] 提取（非截断版本）。

    返回 (EvaluationDataset, debug_info_list)，debug_info_list 包含每条的 response 和 retrieved_contexts。
    """
    samples = []
    debug_info = []
    for i, rec in enumerate(records):
        query = rec["user_input"]
        logger.info("Query %d/%d: %s", i + 1, len(records), query[:50])

        result = asyncio.run(
            rag_chat_service.chat(
                query=query,
                top_k=top_k,
                collection_names=collection_names,
            )
        )

        answer = result["answer"]
        # 从 fields 中提取完整内容，fallback 到截断的 content
        contexts = [
            s.get("fields", {}).get("content", s.get("content", ""))
            for s in result.get("sources", [])
        ]

        logger.info("Query %d/%d done (retrieved %d contexts)", i + 1, len(records), len(contexts))

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference=rec.get("reference"),
            reference_contexts=rec.get("reference_contexts"),
        )
        samples.append(sample)
        debug_info.append({
            "response": answer,
            "retrieved_contexts": contexts,
        })

    return EvaluationDataset(samples=samples), debug_info


def save_results(
    results: List[Dict],
    metrics: List[str],
    output_dir: str,
    experiment_name: str,
    debug_info: Optional[List[Dict]] = None,
) -> str:
    """保存评估结果到 JSON 文件，汇总分数在前，逐条分数在后"""
    summary = {}
    for metric_name in metrics:
        values = [r[metric_name] for r in results if r.get(metric_name) is not None]
        if values:
            summary[metric_name] = round(sum(values) / len(values), 4)
        else:
            summary[metric_name] = None

    # 合并 debug_info（response + retrieved_contexts）到 samples
    samples = results
    if debug_info:
        samples = []
        for r, d in zip(results, debug_info):
            merged = {**r, "response": d.get("response", ""), "retrieved_contexts": d.get("retrieved_contexts", [])}
            samples.append(merged)

    output = {
        "summary": summary,
        "samples": samples,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / f"{experiment_name}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return str(file_path)