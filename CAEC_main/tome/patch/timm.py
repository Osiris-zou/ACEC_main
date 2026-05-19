# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from ..merge import (
    merge_source,
    merge_wavg,
    bipartite_soft_matching,
    bipartite_soft_matching_xincheng,
    bipartite_soft_matching_xincheng_old,
)
from ..utils import parse_r


class ToMeBlock(Block):
    """
    ToMe / ETM-RL Block.

    改动目的：
    1. 在 Attention 和 MLP 之间执行 token merging。
    2. 通过 _tome_info["merge_method"] 明确选择原始 ToMe 或 ETM-RL。
    3. 避免之前写死调用 bipartite_soft_matching_xincheng 导致方法切换混乱。
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # prop_attn=True 时，attention 会考虑 token size。
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None

        # ToMeAttention 返回两个值：
        # x_attn: attention 输出
        # metric: 用于 token matching 的特征，通常是 k.mean(1)
        x_attn, metric = self.attn(self.norm1(x), attn_size)

        # 标准 Transformer 残差
        x = x + self._drop_path1(x_attn)

        # 每一层要合并的 token 数
        r = self._tome_info["r"].pop(0)

        if r > 0:
            merge_method = self._tome_info.get("merge_method", "etmrl")

            if merge_method == "tome":
                merge_fn = bipartite_soft_matching
            elif merge_method == "etmrl":
                merge_fn = bipartite_soft_matching_xincheng
            elif merge_method == "etmrl_old":
                merge_fn = bipartite_soft_matching_xincheng_old
            else:
                raise ValueError(f"Unsupported merge_method: {merge_method}")

            merge, _ = merge_fn(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )

            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )

            x, self._tome_info["size"] = merge_wavg(
                merge, x, self._tome_info["size"]
            )

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    ToMe Attention.

    和原始 timm Attention 的区别：
    1. 支持 proportional attention。
    2. 除了输出 x，还返回 k.mean(1) 作为 token matching metric。
    """

    def forward(
        self,
        x: torch.Tensor,
        size: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        给 timm VisionTransformer 增加 ToMe/ETM-RL 所需状态。
        """

        def forward(self, *args, **kwargs) -> torch.Tensor:
            # 每次 forward 前重置 r、size、source。
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer,
    trace_source: bool = False,
    prop_attn: bool = True,
    merge_method: str = "etmrl",
):
    """
    给 timm VisionTransformer 应用 ToMe/ETM-RL patch。

    参数说明：
    trace_source:
        是否追踪 token 来源，用于可视化。

    prop_attn:
        是否使用 proportional attention。
        用 off-the-shelf 预训练权重评估时建议 True。

    merge_method:
    "tome"      : 原始 ToMe BSM。
    "etmrl"     : 当前 margin 置信度 ETM-RL。
    "etmrl_old" : 旧版上下文窗口 + KNN + L2 距离惩罚 ETM-RL。
    """

    if merge_method not in ["tome", "etmrl", "etmrl_old"]:
        raise ValueError(f"Unsupported merge_method: {merge_method}")

    # 防止重复 patch。
    if hasattr(model, "_tome_info"):
        model._tome_info["trace_source"] = trace_source
        model._tome_info["prop_attn"] = prop_attn
        model._tome_info["merge_method"] = merge_method
        return

    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0

    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
        "merge_method": merge_method,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention