"""低冗余、多维度特征选择。

阶段二采用 mRMR 风格的贪心选择：先看训练集相关性，再用特征间
相关性惩罚冗余，同时约束特征组分布。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from .config import Stage2Config
from .features import FEATURE_GROUP_MAP


def _corr_lookup(corr_pairs: pd.DataFrame) -> Dict[tuple, float]:
    out: Dict[tuple, float] = {}
    if corr_pairs.empty:
        return out
    for _, row in corr_pairs.iterrows():
        a = row["feature_a"]
        b = row["feature_b"]
        value = abs(float(row["spearman_corr"])) if pd.notna(row["spearman_corr"]) else 0.0
        out[(a, b)] = value
        out[(b, a)] = value
    return out


def _max_corr_to_selected(feature: str, selected: List[str], corr_map: Dict[tuple, float]) -> float:
    if not selected:
        return 0.0
    return max(corr_map.get((feature, other), 0.0) for other in selected)


def _mean_corr_to_selected(feature: str, selected: List[str], corr_map: Dict[tuple, float]) -> float:
    if not selected:
        return 0.0
    return float(np.mean([corr_map.get((feature, other), 0.0) for other in selected]))


def _candidate_pool(feature_summary_train: pd.DataFrame) -> pd.DataFrame:
    pool = feature_summary_train.copy()
    pool = pool[pool["coverage"] >= 0.70]
    pool = pool[pool["unique_count"] >= 10]
    pool = pool[pool["inf_count"] == 0]
    pool = pool[pool["spearman_ic_5"].notna()]
    pool = pool[pool["spearman_ic_5"].abs() > 0]
    pool["abs_spearman_ic_5"] = pool["spearman_ic_5"].abs()
    return pool.sort_values("abs_spearman_ic_5", ascending=False).reset_index(drop=True)


def select_features(feature_summary_train: pd.DataFrame, corr_pairs: pd.DataFrame, config: Stage2Config) -> pd.DataFrame:
    """选择低冗余且覆盖多个维度的特征。

    先保证每个可用组至少有一个代表，再按 mRMR 分数补满。用户偏好是
    “多特征优先”，所以默认最多选 18 个、相关阈值 0.85。
    """

    pool = _candidate_pool(feature_summary_train)
    corr_map = _corr_lookup(corr_pairs)
    selected: List[str] = []
    selected_rows: List[dict] = []
    selected_set: Set[str] = set()
    group_cap = max(1, math.floor(config.max_selected_features * config.max_group_share))

    def can_add(feature: str) -> bool:
        if feature in selected_set:
            return False
        if _max_corr_to_selected(feature, selected, corr_map) > config.corr_threshold:
            return False
        group = FEATURE_GROUP_MAP[feature]
        current_group_count = Counter(FEATURE_GROUP_MAP[x] for x in selected)[group]
        return current_group_count + 1 <= group_cap

    def add_row(row: pd.Series, score: float) -> None:
        feature = row["feature"]
        selected.append(feature)
        selected_set.add(feature)
        selected_rows.append(
            {
                "selection_order": len(selected),
                "feature": feature,
                "group": FEATURE_GROUP_MAP[feature],
                "spearman_ic_5": float(row["spearman_ic_5"]),
                "abs_spearman_ic_5": float(abs(row["spearman_ic_5"])),
                "ic_direction": int(np.sign(row["spearman_ic_5"])),
                "coverage": float(row["coverage"]),
                "unique_count": int(row["unique_count"]),
                "mrmr_score": float(score),
                "max_abs_corr_to_selected": _max_corr_to_selected(feature, selected[:-1], corr_map),
                "mean_abs_corr_to_selected": _mean_corr_to_selected(feature, selected[:-1], corr_map),
            }
        )

    # 先给每个特征组一个名额，让信号维度不会被价格结构类完全占掉。
    for group in sorted(pool["group"].dropna().unique()):
        group_pool = pool[pool["group"] == group]
        for _, row in group_pool.iterrows():
            if can_add(row["feature"]):
                add_row(row, float(row["abs_spearman_ic_5"]))
                break

    while len(selected) < config.max_selected_features:
        best_row = None
        best_score = -np.inf
        for _, row in pool.iterrows():
            feature = row["feature"]
            if not can_add(feature):
                continue
            redundancy = _mean_corr_to_selected(feature, selected, corr_map)
            group = FEATURE_GROUP_MAP[feature]
            group_counts = Counter(FEATURE_GROUP_MAP[x] for x in selected)
            diversity_bonus = 0.10 if group_counts[group] == 0 else 0.0
            score = float(row["abs_spearman_ic_5"]) * (1.0 - 0.5 * redundancy + diversity_bonus)
            if score > best_score:
                best_score = score
                best_row = row
        if best_row is None:
            break
        add_row(best_row, best_score)

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        return selected_df
    selected_df["selected_feature_count"] = len(selected_df)
    selected_df["selected_group_count"] = selected_df["group"].nunique()
    selected_df["max_group_share_actual"] = selected_df["group"].value_counts(normalize=True).max()
    selected_df["corr_threshold"] = config.corr_threshold
    selected_df["max_group_share_limit"] = config.max_group_share
    return selected_df
