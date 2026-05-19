import argparse
import csv
from typing import List, Union, Tuple


def parse_r(num_layers: int, r: Union[int, List[int], Tuple[int, float]]) -> List[int]:
    """
    将 r 设置转换为每层合并数量列表。

    作用：
    1. 支持 constant r；
    2. 支持 list r；
    3. 支持 ToMe 风格 tuple schedule。

    参数：
    num_layers: Transformer block 数量。
    r: 每层合并 token 数，或 ToMe 风格 schedule。

    返回：
    每一层实际请求合并的 r 列表。
    """
    inflect = 0

    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)

    if isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val

    if num_layers == 1:
        return [int(r)]

    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def patch_embed_flops(
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 3,
    embed_dim: int = 768,
) -> int:
    """
    估算 ViT patch embedding FLOPs。

    目的：
    ViT-B/16 的 patch embedding 等价于 kernel=16、stride=16 的卷积。

    计算：
    num_patches * patch_size * patch_size * in_channels * embed_dim
    """
    grid = image_size // patch_size
    num_patches = grid * grid

    return num_patches * patch_size * patch_size * in_channels * embed_dim


def layernorm_flops(
    num_tokens: int,
    embed_dim: int = 768,
) -> int:
    """
    估算 LayerNorm FLOPs。

    近似口径：
    每个元素约 5 次操作，用于估计：
    1. 均值相关操作；
    2. 方差相关操作；
    3. 标准化；
    4. 缩放；
    5. 平移。

    说明：
    这个项很小，但加入后可以让 ViT-B/16 full GFLOPs
    从约 17.56 对齐到常见口径约 17.58。
    """
    return 5 * num_tokens * embed_dim


def attention_part_flops(
    tokens_in: int,
    embed_dim: int = 768,
) -> int:
    """
    估算 Attention 部分 FLOPs。

    计算内容：
    1. QKV projection: 3 * N * D * D
    2. Attention QK^T: N * N * D
    3. Attention AV: N * N * D
    4. Output projection: N * D * D

    注意：
    ToMe/Ours 在 attention 后、MLP 前执行 token merging，
    所以 attention 部分必须按 merge 前 tokens_in 计算。
    """
    n = tokens_in
    d = embed_dim

    qkv = 3 * n * d * d
    attn_qk = n * n * d
    attn_av = n * n * d
    proj = n * d * d

    return qkv + attn_qk + attn_av + proj


def mlp_part_flops(
    tokens_out: int,
    embed_dim: int = 768,
    mlp_ratio: float = 4.0,
) -> int:
    """
    估算 MLP 部分 FLOPs。

    计算内容：
    1. fc1: N_out * D * hidden_dim
    2. fc2: N_out * hidden_dim * D

    注意：
    ToMe/Ours 在 attention 后执行 token merging，
    所以 MLP 应该按 merge 后 tokens_out 计算。
    """
    n = tokens_out
    d = embed_dim
    hidden_dim = int(embed_dim * mlp_ratio)

    return 2 * n * d * hidden_dim


def classifier_head_flops(
    embed_dim: int = 768,
    num_classes: int = 1000,
) -> int:
    """
    分类头 FLOPs。

    对 ViT-B/16 来说这部分非常小，但保留用于完整估算。
    """
    return embed_dim * num_classes


def token_schedule(
    r: int,
    num_layers: int = 12,
    init_tokens: int = 197,
    protected_tokens: int = 1,
) -> List[dict]:
    """
    计算每层 token 数变化。

    ViT-B/16@224:
    196 patch tokens + 1 cls token = 197 tokens。

    注意：
    每层最多只能合并非保护 token 的一半。
    """
    r_list = parse_r(num_layers, r)

    schedule = []
    tokens = init_tokens

    for layer_idx, r_layer in enumerate(r_list):
        actual_merge = min(int(r_layer), max(0, (tokens - protected_tokens) // 2))
        next_tokens = tokens - actual_merge

        schedule.append({
            "layer": layer_idx + 1,
            "tokens_in": tokens,
            "r_requested": int(r_layer),
            "r_actual": actual_merge,
            "tokens_out": next_tokens,
        })

        tokens = next_tokens

    return schedule


def estimate_vit_b16_flops(
    r: int = 0,
    image_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    mlp_ratio: float = 4.0,
    num_classes: int = 1000,
    protected_tokens: int = 1,
) -> dict:
    """
    估算 ViT-B/16 在指定 r 下的 GFLOPs。

    核心口径：
    1. Patch embedding 固定计算；
    2. norm1 使用 tokens_in；
    3. Attention 使用 tokens_in；
    4. token merging 发生在 attention 后；
    5. norm2 使用 tokens_out；
    6. MLP 使用 tokens_out；
    7. final norm 使用最终 token 数；
    8. classifier head 使用 cls token 特征，计算量很小。

    该口径可以对齐：
    Full ≈ 17.58 GFLOPs
    r=20 ≈ 7.14 GFLOPs
    r=25 ≈ 5.80 GFLOPs
    """
    num_patches = (image_size // patch_size) ** 2
    init_tokens = num_patches + protected_tokens

    total_flops = 0

    # 1. Patch embedding
    total_flops += patch_embed_flops(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
    )

    # 2. Token schedule
    schedule = token_schedule(
        r=r,
        num_layers=depth,
        init_tokens=init_tokens,
        protected_tokens=protected_tokens,
    )

    # 3. Transformer blocks
    for item in schedule:
        tokens_in = item["tokens_in"]
        tokens_out = item["tokens_out"]

        # norm1 在 attention 前，使用合并前 token 数
        total_flops += layernorm_flops(
            num_tokens=tokens_in,
            embed_dim=embed_dim,
        )

        # attention 部分使用合并前 token 数
        total_flops += attention_part_flops(
            tokens_in=tokens_in,
            embed_dim=embed_dim,
        )

        # norm2 在 MLP 前，但 token merging 已经完成，所以使用合并后 token 数
        total_flops += layernorm_flops(
            num_tokens=tokens_out,
            embed_dim=embed_dim,
        )

        # MLP 部分使用合并后 token 数
        total_flops += mlp_part_flops(
            tokens_out=tokens_out,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
        )

    final_tokens = schedule[-1]["tokens_out"] if schedule else init_tokens

    # 4. final norm，使用最后一层输出 token 数
    total_flops += layernorm_flops(
        num_tokens=final_tokens,
        embed_dim=embed_dim,
    )

    # 5. Classifier head
    total_flops += classifier_head_flops(
        embed_dim=embed_dim,
        num_classes=num_classes,
    )

    total_merged = init_tokens - final_tokens

    return {
        "r": r,
        "init_tokens": init_tokens,
        "final_tokens": final_tokens,
        "total_merged": total_merged,
        "flops": total_flops,
        "gflops": total_flops / 1e9,
        "schedule": schedule,
    }


def print_schedule(info: dict) -> None:
    """
    打印某个 r 下每层 token schedule。

    用途：
    检查 r=20/r=25 为什么会严重掉精度。
    """
    print(f"\n========== Token Schedule: r={info['r']} ==========")
    print(f"{'Layer':<8} {'Tokens In':<12} {'r actual':<10} {'Tokens Out':<12}")

    for item in info["schedule"]:
        print(
            f"{item['layer']:<8} "
            f"{item['tokens_in']:<12} "
            f"{item['r_actual']:<10} "
            f"{item['tokens_out']:<12}"
        )

    print("============================================\n")


def run(args):
    """
    批量输出 Full / ToMe / Ours 的估算 FLOPs。

    注意：
    ToMe 和 Ours 在相同 r 下 token schedule 相同，
    因此主干 GFLOPs 相同；
    二者速度差异主要由额外的 top-2、margin、排序和安全处理开销体现，
    应通过实测 throughput 反映。
    """
    r_list = [int(x.strip()) for x in args.r_list.split(",") if x.strip()]

    full_info = estimate_vit_b16_flops(
        r=0,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
    )

    full_gflops = full_info["gflops"]

    rows = []

    rows.append({
        "method": "Full patched",
        "r": 0,
        "gflops": full_gflops,
        "flops_reduction_percent": 0.0,
        "init_tokens": full_info["init_tokens"],
        "final_tokens": full_info["final_tokens"],
        "total_merged": full_info["total_merged"],
    })

    for method in ["ToMe", "Ours"]:
        for r in r_list:
            info = estimate_vit_b16_flops(
                r=r,
                image_size=args.image_size,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                depth=args.depth,
                mlp_ratio=args.mlp_ratio,
                num_classes=args.num_classes,
            )

            reduction = (full_gflops - info["gflops"]) / full_gflops * 100.0

            rows.append({
                "method": method,
                "r": r,
                "gflops": info["gflops"],
                "flops_reduction_percent": reduction,
                "init_tokens": info["init_tokens"],
                "final_tokens": info["final_tokens"],
                "total_merged": info["total_merged"],
            })

    print("\n========== Estimated FLOPs with LayerNorm ==========")
    print(f"{'Method':<14} {'r':<5} {'GFLOPs':<12} {'FLOPs Red.':<14} {'Final Tokens':<14}")

    for row in rows:
        print(
            f"{row['method']:<14} "
            f"{row['r']:<5} "
            f"{row['gflops']:<12.3f} "
            f"{row['flops_reduction_percent']:<14.2f} "
            f"{row['final_tokens']:<14}"
        )

    print("====================================================\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "r",
                    "gflops",
                    "flops_reduction_percent",
                    "init_tokens",
                    "final_tokens",
                    "total_merged",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"[SAVE] Results saved to: {args.output}")

    if args.print_schedule:
        for r in r_list:
            info = estimate_vit_b16_flops(
                r=r,
                image_size=args.image_size,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                depth=args.depth,
                mlp_ratio=args.mlp_ratio,
                num_classes=args.num_classes,
            )
            print_schedule(info)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument(
        "--r-list",
        type=str,
        default="4,8,12,16,20,25",
        help="需要计算的 r 列表，例如 8,12,16",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="flops_results.csv",
        help="输出 CSV 文件路径。",
    )

    parser.add_argument(
        "--print-schedule",
        action="store_true",
        help="是否打印每个 r 的逐层 token schedule。",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())