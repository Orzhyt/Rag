"""评估结果前后对比工具。

用法:
    # 前后对比
    python -m evaluation.compare evaluation/results/before.json evaluation/results/after.json

    # 代码调用
    from evaluation.compare import compare_results, print_comparison
    comp = compare_results("before.json", "after.json")
    print_comparison(comp)
"""

import json
from typing import Any, Dict, List, Optional


def compare_results(
    before_path: str,
    after_path: str,
) -> Dict[str, Any]:
    """加载两个评估结果 JSON，计算 per-metric delta。

    Returns:
        {
            "before_summary": {...},
            "after_summary": {...},
            "metric_deltas": {metric: delta},
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
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
    }


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估结果前后对比")
    parser.add_argument("before", help="优化前评估结果 JSON 路径")
    parser.add_argument("after", help="优化后评估结果 JSON 路径")
    args = parser.parse_args()

    comparison = compare_results(args.before, args.after)
    print_comparison(comparison)