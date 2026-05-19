# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
import os
from typing import Callable, Tuple

import torch
import torch.nn.functional as F


def do_nothing(x, mode="mean"):
    return x

def _gather_b_candidates(b: torch.Tensor, candidate_idx: torch.Tensor) -> torch.Tensor:
    """
    根据 candidate_idx 从 B 组 token 中收集候选 token。

    参数：
    b: [B, Tb, C]
        B 组 token 特征。
    candidate_idx: [B, Ta, K]
        每个 A token 对应的 top-k 候选 B token 索引。

    返回：
    candidates: [B, Ta, K, C]
        每个 A token 的 K 个候选 B token 特征。
    """
    batch_size, tb, channels = b.shape
    _, ta, k = candidate_idx.shape

    # 扩展 B 组 token，便于按每个 A token 的候选索引 gather。
    b_expand = b[:, None, :, :].expand(batch_size, ta, tb, channels)

    # 扩展索引到最后一维 C。
    idx_expand = candidate_idx[..., None].expand(batch_size, ta, k, channels)

    # 在 B token 维度上收集候选。
    candidates = b_expand.gather(dim=2, index=idx_expand)

    return candidates

def _safe_l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    L2 normalize，避免除 0。
    """
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _spatial_penalty_like_scores(
    scores: torch.Tensor,
    original_t: int,
) -> torch.Tensor:
    """
    构造轻量结构距离惩罚。

    scores 形状：[B, Ta, Tb]
    A 组对应原始 token 索引 0,2,4...
    B 组对应原始 token 索引 1,3,5...

    这里不再用 torch.cdist(a,b)，而是用序列位置距离模拟结构接近性。
    """
    _, Ta, Tb = scores.shape
    device = scores.device
    dtype = scores.dtype

    a_pos = torch.arange(0, original_t, 2, device=device, dtype=dtype)[:Ta]
    b_pos = torch.arange(1, original_t, 2, device=device, dtype=dtype)[:Tb]

    penalty = (a_pos[:, None] - b_pos[None, :]).abs()

    if original_t > 1:
        penalty = penalty / float(original_t - 1)

    return penalty.view(1, Ta, Tb)


def _make_merge_unmerge(
    metric_t: int,
    r: int,
    unm_idx: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    distill_token: bool,
) -> Tuple[Callable, Callable]:
    """
    根据索引构造 merge/unmerge 函数。
    """

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape

        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric_t, c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    原始 ToMe Bipartite Soft Matching。

    这个函数尽量保持原始 ToMe 逻辑，用作基线。
    """

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = _safe_l2_normalize(metric)

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    return _make_merge_unmerge(
        metric_t=t,
        r=r,
        unm_idx=unm_idx,
        src_idx=src_idx,
        dst_idx=dst_idx,
        distill_token=distill_token,
    )


def bipartite_soft_matching_xincheng(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    window_size: int = 4,
    knn_k: int = 1,
) -> Tuple[Callable, Callable]:
    """
    ETM-RL 安全回退版。

    目的：
    1. 保留 ToMe 原始 cosine matching 作为主逻辑；
    2. 避免 class token 行出现 NaN；
    3. 避免 margin 项破坏原始 ToMe 选边；
    4. 先保证 ETM-RL 的精度不低于或接近 ToMe，再逐步加创新项。
    """

    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # 1. 与 ToMe 一样，对 K 特征做 L2 normalize。
        metric = _safe_l2_normalize(metric)

        # 2. A/B 交替分组。
        a, b = metric[..., ::2, :], metric[..., 1::2, :]

        # 3. cosine similarity。
        scores = a @ b.transpose(-1, -2)

        # 4. 保护 class token 和 distill token。
        if class_token:
            scores[..., 0, :] = -math.inf

        if distill_token:
            scores[..., :, 0] = -math.inf

        # 5. ToMe 原始 top-1 匹配。
        node_max, node_idx = scores.max(dim=-1)

        # 6. 计算 top-2 margin，但必须安全处理 -inf / NaN。
        if scores.shape[-1] >= 2:
            top2_scores, _ = scores.topk(
                k=2,
                dim=-1,
                largest=True,
                sorted=True,
            )

            margin = top2_scores[..., 0] - top2_scores[..., 1]

            # class token 行可能出现 -inf - -inf = NaN，必须清理。
            margin = torch.nan_to_num(
                margin,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        else:
            margin = torch.zeros_like(node_max)

        # 7. 极小 margin 修正。
        # 注意：这里不能用 0.20，太大。
        # 先用 0.00 等价于 ToMe，确认无误后再尝试 0.01 / 0.02。
        beta_margin = float(os.getenv("ETMRL_BETA_MARGIN", "0.015"))

        edge_score = node_max + beta_margin * margin

        # 8. 再次强制保护 class token，避免 NaN 或 margin 影响。
        if class_token:
            edge_score[..., 0] = -math.inf

        # 9. 清理所有异常值，防止 argsort 出问题。
        edge_score = torch.nan_to_num(
            edge_score,
            nan=-math.inf,
            posinf=math.inf,
            neginf=-math.inf,
        )

        # 10. 选择全局 top-r 条边。
        edge_idx = edge_score.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]

        dst_idx = node_idx[..., None].gather(
            dim=-2,
            index=src_idx,
        )

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    return _make_merge_unmerge(
        metric_t=t,
        r=r,
        unm_idx=unm_idx,
        src_idx=src_idx,
        dst_idx=dst_idx,
        distill_token=distill_token,
    )


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 token size 做加权平均合并。
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size.clamp_min(1e-6)
    return x, size


def merge_source(
    merge: Callable,
    x: torch.Tensor,
    source: torch.Tensor = None,
) -> torch.Tensor:
    """
    token 来源追踪，用于可视化。
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device, dtype=x.dtype)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source