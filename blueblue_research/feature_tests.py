"""阶段二单特征检验。

训练集和测试集分开输出，避免阶段一 `feature_quintiles.csv` 全样本统计
被误解为选特征依据。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FEATURE_GROUP_MAP


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="pearson")) if len(x) >= 20 else np.nan


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 20:
        return np.nan
    ranked = pd.DataFrame({"x": x, "y": y}).rank(method="average")
    return float(ranked["x"].corr(ranked["y"], method="pearson"))


def build_feature_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total = len(df)
    for feature in FEATURE_COLUMNS:
        values = pd.to_numeric(df[feature], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "feature": feature,
                "group": FEATURE_GROUP_MAP[feature],
                "non_null_count": int(finite.notna().sum()),
                "missing_count": int(finite.isna().sum()),
                "missing_rate": float(finite.isna().mean()) if total else np.nan,
                "coverage": float(finite.notna().mean()) if total else np.nan,
                "unique_count": int(finite.nunique(dropna=True)),
                "inf_count": int(np.isinf(values.dropna()).sum()),
                "std": float(finite.std(ddof=0)) if finite.notna().any() else np.nan,
                "min": float(finite.min()) if finite.notna().any() else np.nan,
                "p01": float(finite.quantile(0.01)) if finite.notna().any() else np.nan,
                "median": float(finite.median()) if finite.notna().any() else np.nan,
                "p99": float(finite.quantile(0.99)) if finite.notna().any() else np.nan,
                "max": float(finite.max()) if finite.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def feature_quintile_stats(df: pd.DataFrame, feature: str, label_col: str, split_name: str) -> pd.DataFrame:
    valid = df[[feature, label_col, "mfe_10_long", "mae_10_long"]].dropna(subset=[feature, label_col]).copy()
    if valid[feature].nunique(dropna=True) < 5:
        return pd.DataFrame()
    try:
        valid["quintile_num"] = pd.qcut(valid[feature], 5, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for q, part in valid.groupby("quintile_num"):
        rows.append(
            {
                "split": split_name,
                "feature": feature,
                "group": FEATURE_GROUP_MAP[feature],
                "quintile": f"Q{int(q)}",
                "count": int(len(part)),
                "mean_feature": float(part[feature].mean()),
                "mean_return": float(part[label_col].mean()),
                "median_return": float(part[label_col].median()),
                "win_rate": float((part[label_col] > 0).mean()),
                "mean_mfe_10_long": float(part["mfe_10_long"].mean()),
                "mean_mae_10_long": float(part["mae_10_long"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_features(df: pd.DataFrame, label_col: str, split_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    quintile_frames = []
    for feature in FEATURE_COLUMNS:
        valid = df[[feature, label_col, "mfe_10_long", "mae_10_long"]].replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, label_col])
        quintiles = feature_quintile_stats(df, feature, label_col, split_name)
        if not quintiles.empty:
            quintile_frames.append(quintiles)

        row: Dict[str, Any] = {
            "split": split_name,
            "feature": feature,
            "group": FEATURE_GROUP_MAP[feature],
            "non_null_count": int(df[feature].replace([np.inf, -np.inf], np.nan).notna().sum()),
            "valid_count": int(len(valid)),
            "coverage": float(len(valid) / len(df)) if len(df) else np.nan,
            "unique_count": int(valid[feature].nunique(dropna=True)) if len(valid) else 0,
            "inf_count": int(np.isinf(pd.to_numeric(df[feature], errors="coerce").dropna()).sum()),
            "pearson_ic_5": pearson_corr(valid[feature], valid[label_col]) if len(valid) else np.nan,
            "spearman_ic_5": spearman_corr(valid[feature], valid[label_col]) if len(valid) else np.nan,
        }
        row["abs_spearman_ic_5"] = abs(row["spearman_ic_5"]) if pd.notna(row["spearman_ic_5"]) else np.nan
        row["ic_direction"] = int(np.sign(row["spearman_ic_5"])) if pd.notna(row["spearman_ic_5"]) else 0
        if not quintiles.empty and {"Q1", "Q5"}.issubset(set(quintiles["quintile"])):
            q1 = quintiles.loc[quintiles["quintile"] == "Q1"].iloc[0]
            q5 = quintiles.loc[quintiles["quintile"] == "Q5"].iloc[0]
            row["quintile_spread_mean_return"] = float(q5["mean_return"] - q1["mean_return"])
            row["top_quintile_win_rate"] = float(q5["win_rate"])
            row["top_quintile_mean_mfe"] = float(q5["mean_mfe_10_long"])
            row["top_quintile_mean_mae"] = float(q5["mean_mae_10_long"])
        else:
            row["quintile_spread_mean_return"] = np.nan
            row["top_quintile_win_rate"] = np.nan
            row["top_quintile_mean_mfe"] = np.nan
            row["top_quintile_mean_mae"] = np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("abs_spearman_ic_5", ascending=False)
    quintiles = pd.concat(quintile_frames, ignore_index=True) if quintile_frames else pd.DataFrame()
    return summary, quintiles


def build_monthly_ic(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    working = df.copy()
    working["month"] = working["datetime"].dt.to_period("M").astype(str)
    for month, part in working.groupby("month"):
        for feature in FEATURE_COLUMNS:
            valid = part[[feature, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 20:
                continue
            rows.append(
                {
                    "month": month,
                    "feature": feature,
                    "group": FEATURE_GROUP_MAP[feature],
                    "count": int(len(valid)),
                    "pearson_ic_5": pearson_corr(valid[feature], valid[label_col]),
                    "spearman_ic_5": spearman_corr(valid[feature], valid[label_col]),
                }
            )
    return pd.DataFrame(rows)


def build_feature_correlation(train_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pearson = train_df[FEATURE_COLUMNS].corr(method="pearson")
    # 避免依赖 scipy：Spearman 等价于对 rank 后的变量做 Pearson 相关。
    spearman = train_df[FEATURE_COLUMNS].rank(method="average").corr(method="pearson")
    for i, feature_a in enumerate(FEATURE_COLUMNS):
        for feature_b in FEATURE_COLUMNS[i + 1 :]:
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "group_a": FEATURE_GROUP_MAP[feature_a],
                    "group_b": FEATURE_GROUP_MAP[feature_b],
                    "pearson_corr": float(pearson.loc[feature_a, feature_b]) if pd.notna(pearson.loc[feature_a, feature_b]) else np.nan,
                    "spearman_corr": float(spearman.loc[feature_a, feature_b]) if pd.notna(spearman.loc[feature_a, feature_b]) else np.nan,
                }
            )
    return pd.DataFrame(rows)
