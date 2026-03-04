import os
import json
from contextlib import redirect_stdout
from datetime import datetime

from galvatron.core import GalvatronSearchEngine, initialize_galvatron
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config
from galvatron.models.llama_hf.meta_configs import model_layer_configs, model_name


def build_search_engine():
    """初始化用于 LLaMA HF 的 GalvatronSearchEngine。"""
    args = initialize_galvatron(model_args, mode="search")
    config = get_llama_config(args)

    # 当前文件在 llama_hf/scripts 下，向上一层是 llama_hf 目录
    llama_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(
        llama_dir,
        model_layer_configs(config),
        model_name(config),
    )
    search_engine.set_model_type("gpt")
    search_engine.initialize_search_engine()
    return search_engine, args


def main():
    """批量调用 check_cost_model，保存日志与 meta 信息。"""
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    logs_dir = os.path.join(repo_root, "logs", "cost_model")
    os.makedirs(logs_dir, exist_ok=True)

    # 可根据需要调整的 (bsz, chunk, min_tp) 组合
    test_settings = [
        {"bsz": 16, "chunk": 1, "min_tp": 1},
        {"bsz": 32, "chunk": 1, "min_tp": 1},
        {"bsz": 32, "chunk": 2, "min_tp": 1},
        {"bsz": 64, "chunk": 1, "min_tp": 2},
    ]

    search_engine, args = build_search_engine()
    meta_records = []

    for s in test_settings:
        bsz, chunk, min_tp = s["bsz"], s["chunk"], s["min_tp"]

        log_name = f"cost_llama7b_bsz{bsz}_chunk{chunk}_mintp{min_tp}.log"
        log_path = os.path.join(logs_dir, log_name)

        print(
            f"[check_cost_model] bsz={bsz}, chunk={chunk}, "
            f"min_tp={min_tp}, log -> {log_path}"
        )

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                f"# check_cost_model for llama7b "
                f"(bsz={bsz}, chunk={chunk}, min_tp={min_tp})\n"
            )
            f.write(f"# time = {datetime.now().isoformat()}\n\n")
            # 将 check_cost_model 的输出重定向到文件
            with redirect_stdout(f):
                search_engine.check_cost_model(bsz=bsz, chunk=chunk, min_tp=min_tp)

        meta_records.append(
            {
                "bsz": bsz,
                "chunk": chunk,
                "min_tp": min_tp,
                "log_path": log_path.replace("\\", "/"),
            }
        )

    meta_path = os.path.join(logs_dir, "llama7b_cost_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_records, f, indent=2, ensure_ascii=False)

    print(f"[DONE] cost model 日志已写入 {logs_dir}，meta -> {meta_path}")


if __name__ == "__main__":
    main()
