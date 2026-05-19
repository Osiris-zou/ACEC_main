import os
import sys
import csv
import json
import time
import argparse
import subprocess
from datetime import datetime


def parse_args():
    """
    解析命令行参数。
    作用：
    1. 指定 eval 脚本路径；
    2. 指定数据集、权重、batch size；
    3. 指定 r 列表和 beta 列表；
    4. 指定结果保存路径。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-script",
        type=str,
        default="vit_b_eval.py",
        help="正式评估脚本路径。"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=r"D:\imagenet-1k\val",
        help="ImageNet-1k 验证集路径。"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=r"E:\zp\vision_transformer\vit_base_patch16_224.pth",
        help="ViT-B/16 ImageNet-1k 权重路径。"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_base_patch16_224",
        help="timm 模型名称。"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="评估 batch size。"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="DataLoader worker 数量。"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="运行设备。"
    )

    parser.add_argument(
        "--preprocess",
        type=str,
        default="inception",
        choices=["inception", "imagenet"],
        help="预处理方式。你当前 84.41% full baseline 对应 inception。"
    )

    parser.add_argument(
        "--lambda-spatial",
        type=float,
        default=0.01,
        help="ETM-RL 的 lambda_spatial，默认固定为 0.01。"
    )

    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=80,
        help="模型吞吐量 warmup 次数。"
    )

    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=300,
        help="每次 benchmark 的正式计时轮数。"
    )

    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=3,
        help="benchmark 重复次数。大规模 sweep 建议先用 3，最终表格再用 5 或 7。"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="sweep_etmrl_beta_results.csv",
        help="结果 CSV 保存路径。"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果 CSV 里已经有对应 r 和 beta，就跳过。"
    )

    return parser.parse_args()


def load_existing_results(csv_path):
    """
    读取已有 CSV 结果。
    作用：
    如果中途断了，重新运行时可以跳过已经完成的组合。
    """
    existing = set()

    if not os.path.exists(csv_path):
        return existing

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["r"])
            beta = float(row["beta_margin"])
            existing.add((r, round(beta, 6)))

    return existing


def ensure_csv_header(csv_path):
    """
    创建 CSV 表头。
    作用：
    第一次运行时自动写入列名。
    """
    if os.path.exists(csv_path):
        return

    fieldnames = [
        "time",
        "method",
        "r",
        "beta_margin",
        "lambda_spatial",
        "top1",
        "top5",
        "e2e_tps",
        "model_tps_mean",
        "model_tps_median",
        "model_tps_min",
        "model_tps_max",
        "status",
        "note",
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_result(csv_path, row):
    """
    追加一行结果到 CSV。
    作用：
    每完成一个组合就立刻保存，避免中途断电或报错丢失全部结果。
    """
    fieldnames = [
        "time",
        "method",
        "r",
        "beta_margin",
        "lambda_spatial",
        "top1",
        "top5",
        "e2e_tps",
        "model_tps_mean",
        "model_tps_median",
        "model_tps_min",
        "model_tps_max",
        "status",
        "note",
    ]

    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def run_one(args, r_value, beta_value):
    """
    运行一次 ETM-RL 验证。
    作用：
    1. 设置环境变量 ETMRL_BETA_MARGIN；
    2. 调用 eval_vit_methods_stable.py；
    3. 从输出中解析 RESULT_JSON；
    4. 返回结果字典。
    """
    env = os.environ.copy()

    # 通过环境变量控制 merge.py 里的 beta_margin。
    env["ETMRL_BETA_MARGIN"] = str(beta_value)

    # 通过环境变量控制 merge.py 里的 lambda_spatial。
    env["ETMRL_LAMBDA_SPATIAL"] = str(args.lambda_spatial)

    cmd = [
        sys.executable,
        args.eval_script,
        "--method", "etmrl",
        "--r", str(r_value),
        "--data-path", args.data_path,
        "--weights", args.weights,
        "--model-name", args.model_name,
        "--num-classes", "1000",
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--preprocess", args.preprocess,
        "--benchmark-warmup", str(args.benchmark_warmup),
        "--benchmark-runs", str(args.benchmark_runs),
        "--benchmark-repeats", str(args.benchmark_repeats),
        "--prop-attn",
    ]

    print("\n" + "=" * 100)
    print(f"[RUN] r={r_value}, beta_margin={beta_value}, lambda_spatial={args.lambda_spatial}")
    print("[CMD]", " ".join(cmd))
    print("=" * 100)

    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(proc.stdout)

    result_json = None

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result_json = line[len("RESULT_JSON:"):]
            break

    if proc.returncode != 0:
        return {
            "status": "failed",
            "note": f"returncode={proc.returncode}",
            "raw_output": proc.stdout,
        }

    if result_json is None:
        return {
            "status": "failed",
            "note": "RESULT_JSON not found",
            "raw_output": proc.stdout,
        }

    try:
        result = json.loads(result_json)
        result["status"] = "ok"
        result["note"] = ""
        return result
    except Exception as e:
        return {
            "status": "failed",
            "note": f"json parse error: {e}",
            "raw_output": proc.stdout,
        }


def main():
    args = parse_args()

    r_list = [4, 8, 12, 16, 20, 25]

    beta_list = [
        0.000,
        0.005,
        0.010,
        0.015,
        0.020,
        0.025,
        0.030,
        0.035,
        0.040,
        0.045,
        0.050,
        0.055,
        0.060,
        0.065,
        0.070,
    ]

    ensure_csv_header(args.output)

    existing = load_existing_results(args.output) if args.skip_existing else set()

    total_jobs = len(r_list) * len(beta_list)
    job_idx = 0

    print("\n========== Sweep Config ==========")
    print(f"eval_script       : {args.eval_script}")
    print(f"data_path         : {args.data_path}")
    print(f"weights           : {args.weights}")
    print(f"r_list            : {r_list}")
    print(f"beta_list         : {beta_list}")
    print(f"lambda_spatial    : {args.lambda_spatial}")
    print(f"benchmark_repeats : {args.benchmark_repeats}")
    print(f"output            : {args.output}")
    print(f"total_jobs        : {total_jobs}")
    print("==================================\n")

    for r_value in r_list:
        for beta_value in beta_list:
            job_idx += 1

            beta_key = round(beta_value, 6)

            if (r_value, beta_key) in existing:
                print(f"[SKIP] ({job_idx}/{total_jobs}) r={r_value}, beta={beta_value} already exists.")
                continue

            start_time = time.time()

            result = run_one(args, r_value, beta_value)

            elapsed = time.time() - start_time

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if result.get("status") == "ok":
                row = {
                    "time": now,
                    "method": result.get("method", "etmrl"),
                    "r": r_value,
                    "beta_margin": beta_value,
                    "lambda_spatial": args.lambda_spatial,
                    "top1": result.get("top1", ""),
                    "top5": result.get("top5", ""),
                    "e2e_tps": result.get("e2e_tps", ""),
                    "model_tps_mean": result.get("model_tps_mean", ""),
                    "model_tps_median": result.get("model_tps_median", ""),
                    "model_tps_min": result.get("model_tps_min", ""),
                    "model_tps_max": result.get("model_tps_max", ""),
                    "status": "ok",
                    "note": f"elapsed_sec={elapsed:.1f}",
                }

                print(
                    f"[DONE] ({job_idx}/{total_jobs}) "
                    f"r={r_value}, beta={beta_value}, "
                    f"Top1={row['top1']:.2f}, "
                    f"MedianTPS={row['model_tps_median']:.2f}, "
                    f"elapsed={elapsed:.1f}s"
                )

            else:
                row = {
                    "time": now,
                    "method": "etmrl",
                    "r": r_value,
                    "beta_margin": beta_value,
                    "lambda_spatial": args.lambda_spatial,
                    "top1": "",
                    "top5": "",
                    "e2e_tps": "",
                    "model_tps_mean": "",
                    "model_tps_median": "",
                    "model_tps_min": "",
                    "model_tps_max": "",
                    "status": "failed",
                    "note": result.get("note", "unknown error"),
                }

                print(
                    f"[FAILED] ({job_idx}/{total_jobs}) "
                    f"r={r_value}, beta={beta_value}, note={row['note']}"
                )

            append_result(args.output, row)

    print("\n========== Sweep Finished ==========")
    print(f"Results saved to: {args.output}")
    print("====================================\n")


if __name__ == "__main__":
    main()