import argparse
import json
import os
import re
from typing import Dict, List, Any


def load_runtime_results(runtime_json: str) -> Dict[str, Any]:
    with open(runtime_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["config_name"]: item for item in data}


# TODO: 按实际 check_cost_model 输出格式调整此正则
# 示例假设行格式类似:
#   [pp1_tp2_dp4] pred_time=0.123 ...
STRATEGY_LINE_RE = re.compile(
    r"\[(?P<strategy>[^]]+)\].*pred_time=(?P<pred_time>[0-9.]+)",
    re.IGNORECASE,
)


def parse_cost_model_log(log_path: str) -> List[Dict[str, Any]]:
    """从 cost model 日志中解析策略名与预测时间。"""
    results: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = STRATEGY_LINE_RE.search(line)
            if not m:
                continue
            strategy = m.group("strategy").strip()
            pred_time = float(m.group("pred_time"))
            results.append({"strategy": strategy, "pred_time": pred_time})
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare Galvatron cost model vs runtime for LLaMA-7B."
    )
    parser.add_argument(
        "--cost-meta",
        default="logs/cost_model/llama7b_cost_model_meta.json",
        help="run_llama7b_check_cost_model.py 生成的 meta JSON（相对仓库根目录）",
    )
    parser.add_argument(
        "--runtime-json",
        default="logs/runtime/llama7b_runtime_results.json",
        help="run_llama7b_runtime_benchmark.py 生成的 JSON（相对仓库根目录）",
    )
    parser.add_argument(
        "--output",
        default="logs/analysis/llama7b_costmodel_vs_runtime.json",
        help="分析结果输出 JSON（相对仓库根目录）",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    cost_meta_path = os.path.join(repo_root, args.cost_meta)
    runtime_json_path = os.path.join(repo_root, args.runtime_json)
    output_path = os.path.join(repo_root, args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(cost_meta_path, "r", encoding="utf-8") as f:
        cost_meta = json.load(f)

    runtime_index = load_runtime_results(runtime_json_path)

    all_records: List[Dict[str, Any]] = []

    for item in cost_meta:
        bsz = item["bsz"]
        chunk = item["chunk"]
        min_tp = item["min_tp"]
        log_path = item["log_path"]

        cost_entries = parse_cost_model_log(log_path)
        for c in cost_entries:
            strategy_name = c["strategy"]
            # 简单的策略名到 config_name 映射规则
            config_name = f"llama7b_{strategy_name}"

            runtime = runtime_index.get(config_name)
            if not runtime:
                continue

            pred = c["pred_time"]
            real = runtime["time_per_iter_sec"]
            if real is None or real <= 0:
                continue

            abs_err = abs(pred - real)
            rel_err = abs_err / real

            all_records.append(
                {
                    "strategy": strategy_name,
                    "config_name": config_name,
                    "bsz": bsz,
                    "chunk": chunk,
                    "min_tp": min_tp,
                    "pred_time": pred,
                    "real_time": real,
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    if all_records:
        avg_rel = sum(r["rel_error"] for r in all_records) / len(all_records)
        print(
            f"[DONE] 共对齐 {len(all_records)} 条策略，平均相对误差 = {avg_rel*100:.2f}%"
        )
        print(f"      详细结果见: {output_path}")
    else:
        print("[WARN] 没有成功对齐任何策略，请检查日志解析和 config_name 映射。")


if __name__ == "__main__":
    main()
