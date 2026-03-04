import argparse
import json
import os
import time
import subprocess
from datetime import datetime


def run_one_benchmark(config_name, extra_args, train_iters, log_dir):
    """调用 llama_hf train_dist，统计整体运行时间。"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{config_name}.log")

    # 注意：下面 CLI 参数需要与你们实际的 train_dist 保持一致
    cmd = [
        "python",
        "-m",
        "galvatron.models.llama_hf.train_dist",
        "--galvatron-config-path",
        os.path.join(
            "galvatron", "models", "llama_hf", "configs", f"{config_name}.json"
        ),
        "--train-iters",
        str(train_iters),
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"[BENCH] running: {' '.join(cmd)}")
    print(f"[BENCH] log -> {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# runtime benchmark for {config_name}\n")
        f.write(f"# time = {datetime.now().isoformat()}\n\n")
        f.flush()

        start = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        retcode = proc.wait()
        end = time.time()

    elapsed = end - start
    print(f"[BENCH] config={config_name}, elapsed={elapsed:.3f}s, exit_code={retcode}")
    return {
        "config_name": config_name,
        "elapsed_sec": elapsed,
        "train_iters": train_iters,
        "time_per_iter_sec": elapsed / train_iters if train_iters > 0 else None,
        "exit_code": retcode,
        "log_path": log_path.replace("\\", "/"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaMA-7B Galvatron runtime benchmarks for multiple configs."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="不带 .json 后缀的配置名列表，例如: llama7b_pp1_tp1_dp8 llama7b_pp1_tp2_dp4",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=60,
        help="train_dist 总迭代步数，用于计算平均每步时间",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/runtime",
        help="训练日志输出目录（相对仓库根目录）",
    )
    parser.add_argument(
        "--result-json",
        default="logs/runtime/llama7b_runtime_results.json",
        help="汇总结果 JSON 保存路径（相对仓库根目录）",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="透传给 train_dist 的额外参数，例如: --num-nodes 1 --num-gpus-per-node 8",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 仓库根目录: ../../..
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    log_dir = os.path.join(repo_root, args.log_dir)
    result_json = os.path.join(repo_root, args.result_json)
    os.makedirs(os.path.dirname(result_json), exist_ok=True)

    # 在仓库根目录下运行子进程，方便使用相对路径
    os.chdir(repo_root)

    results = []
    for cfg in args.configs:
        res = run_one_benchmark(
            config_name=cfg,
            extra_args=args.extra_args,
            train_iters=args.train_iters,
            log_dir=log_dir,
        )
        results.append(res)

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] runtime 结果写入 {result_json}")


if __name__ == "__main__":
    main()
