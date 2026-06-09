"""评估结果分组聚合与前后对比工具。

用法:
    # 前后对比
    python -m evaluation.compare evaluation/results/before.json evaluation/results/after.json

    # 代码调用
    from evaluation.compare import compare_results, print_comparison
    comp = compare_results("before.json", "after.json")
    print_comparison(comp)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 已知的元数据字段，不参与指标聚合
_METADATA_KEYS = frozenset({
    "user_input", "response", "retrieved_contexts", "reference",
    "reference_contexts", "optimization_target",
    "optimization_note", "source", "persona_name", "query_style",
    "query_length", "synthesizer_name",
})


def _detect_metric_keys(samples: List[Dict[str, Any]]) -> List[str]:
    """自动检测指标字段：值为 int/float 且不是元数据字段的 key。"""
    metric_keys: set = set()
    for sample in samples:
        for k, v in sample.items():
            if k in _METADATA_KEYS:
                continue
            if isinstance(v, (int, float)):
                metric_keys.add(k)
    return sorted(metric_keys)


def _mean(values: List[float]) -> Optional[float]:
    """计算均值，忽略 None。空列表返回 None。"""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def aggregate_by_target(
    samples: List[Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """按 optimization_target 分组聚合指标均值。

    Args:
        samples: 逐条评估结果列表，每条含 optimization_target 和各指标分数字段
        metric_keys: 要聚合的指标名列表。None 则自动检测。

    Returns:
        {optimization_target: {metric: avg_score}, "__all__": {metric: overall_avg}}
    """
    if not samples:
        return {}

    if metric_keys is None:
        metric_keys = _detect_metric_keys(samples)

    # 按 target 分组
    groups: Dict[str, List[Dict]] = {}
    for sample in samples:
        target = sample.get("optimization_target", "unknown")
        groups.setdefault(target, []).append(sample)

    result: Dict[str, Dict[str, Optional[float]]] = {}

    for target, group_samples in sorted(groups.items()):
        agg: Dict[str, Optional[float]] = {}
        for mk in metric_keys:
            values = [s.get(mk) for s in group_samples]
            agg[mk] = _mean(values)
        result[target] = agg

    # 全局均值
    all_agg: Dict[str, Optional[float]] = {}
    for mk in metric_keys:
        values = [s.get(mk) for s in samples]
        all_agg[mk] = _mean(values)
    result["__all__"] = all_agg

    return result


def compare_results(
    before_path: str,
    after_path: str,
) -> Dict[str, Any]:
    """加载两个评估结果 JSON，计算 per-metric 和 per-target delta。

    Returns:
        {
            "before_summary": {...},
            "after_summary": {...},
            "metric_deltas": {metric: delta},
            "target_deltas": {target: {metric: delta}},
            "improved": [metric, ...],
            "regressed": [metric, ...],
            "unchanged": [metric, ...],
        }
    """
    with open(before_path, "r", encoding="utf-8") as f:
        before = json.load(f)
    with open(after_path, "r", encoding="utf-8") as f:
        after = json.load(f)

    before_summary = before.get("summary", {})
    after_summary = after.get("summary", {})

    # Per-metric delta
    all_metrics = sorted(set(before_summary.keys()) | set(after_summary.keys()))
    metric_deltas: Dict[str, Optional[float]] = {}
    for mk in all_metrics:
        bv = before_summary.get(mk)
        av = after_summary.get(mk)
        if bv is not None and av is not None:
            metric_deltas[mk] = round(av - bv, 4)
        else:
            metric_deltas[mk] = None

    # Per-target delta
    before_agg = aggregate_by_target(before.get("samples", []))
    after_agg = aggregate_by_target(after.get("samples", []))
    all_targets = sorted(set(before_agg.keys()) | set(after_agg.keys()))

    target_deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for target in all_targets:
        ba = before_agg.get(target, {})
        aa = after_agg.get(target, {})
        all_mks = sorted(set(ba.keys()) | set(aa.keys()))
        deltas: Dict[str, Optional[float]] = {}
        for mk in all_mks:
            bv = ba.get(mk)
            av = aa.get(mk)
            if bv is not None and av is not None:
                deltas[mk] = round(av - bv, 4)
            else:
                deltas[mk] = None
        target_deltas[target] = deltas

    # 分类
    improved, regressed, unchanged = [], [], []
    for mk, delta in metric_deltas.items():
        if delta is None:
            unchanged.append(mk)
        elif delta > 0.01:
            improved.append(mk)
        elif delta < -0.01:
            regressed.append(mk)
        else:
            unchanged.append(mk)

    return {
        "before_summary": before_summary,
        "after_summary": after_summary,
        "metric_deltas": metric_deltas,
        "target_deltas": target_deltas,
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
    }


def print_aggregation(agg: Dict[str, Dict[str, Optional[float]]]) -> None:
    """格式化打印分组聚合报告。"""
    if not agg:
        print("  (无数据)")
        return

    # 确定指标列
    all_metrics = sorted(set(mk for group in agg.values() for mk in group.keys()))

    # 表头
    header = f"{'target':<20}" + "".join(f"{mk:>14}" for mk in all_metrics)
    print(header)
    print("-" * len(header))

    for target, scores in agg.items():
        label = target if target != "__all__" else "(all)"
        row = f"{label:<20}"
        for mk in all_metrics:
            v = scores.get(mk)
            row += f"{v:>14.4f}" if v is not None else f"{'N/A':>14}"
        print(row)


def print_comparison(comparison: Dict[str, Any]) -> None:
    """格式化打印前后对比报告。"""
    print("=" * 60)
    print("  评估结果前后对比")
    print("=" * 60)

    # Per-metric delta
    print("\n【指标变化】")
    print(f"{'metric':<24}{'before':>10}{'after':>10}{'delta':>10}")
    print("-" * 54)
    for mk in sorted(comparison["metric_deltas"].keys()):
        delta = comparison["metric_deltas"][mk]
        bv = comparison["before_summary"].get(mk)
        av = comparison["after_summary"].get(mk)
        bv_str = f"{bv:.4f}" if bv is not None else "N/A"
        av_str = f"{av:.4f}" if av is not None else "N/A"
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        print(f"{mk:<24}{bv_str:>10}{av_str:>10}{delta_str:>10}")

    # 分类
    print(f"\n  ✅ 改善: {', '.join(comparison['improved']) or '无'}")
    print(f"  ❌ 退步: {', '.join(comparison['regressed']) or '无'}")
    print(f"  ➖ 持平: {', '.join(comparison['unchanged']) or '无'}")

    # Per-target delta
    target_deltas = comparison.get("target_deltas", {})
    if target_deltas:
        print("\n【按优化目标分组变化】")
        for target in sorted(target_deltas.keys()):
            if target == "__all__":
                continue
            deltas = target_deltas[target]
            parts = []
            for mk, d in sorted(deltas.items()):
                if d is not None and abs(d) > 0.01:
                    sign = "↑" if d > 0 else "↓"
                    parts.append(f"{mk} {sign}{abs(d):.3f}")
            summary = ", ".join(parts) if parts else "无明显变化"
            print(f"  {target:<20} {summary}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估结果前后对比")
    parser.add_argument("before", help="优化前评估结果 JSON 路径")
    parser.add_argument("after", help="优化后评估结果 JSON 路径")
    args = parser.parse_args()

    comparison = compare_results(args.before, args.after)
    print_comparison(comparison)
