"""阶段二特征工程。

特征设计遵循三条原则：
1. 只使用当前 Bar 和历史 Bar，避免未来函数。
2. 覆盖价格结构、订单流、POC、成交量、持仓量、市场状态、交互项等维度。
3. 训练集拟合 winsorize 和标准化参数，测试集只复用这些参数。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BASE_COLUMNS, Stage2Config


FEATURE_SPECS: List[Dict[str, str]] = [
    {"feature": "bar_return", "group": "price_structure", "description": "当前Bar收盘相对开盘收益", "formula_note": "close/open - 1"},
    {"feature": "close_to_prev_close_return", "group": "price_structure", "description": "当前收盘相对上一Bar收盘收益", "formula_note": "close/prev_close - 1"},
    {"feature": "gap_return", "group": "price_structure", "description": "当前开盘相对上一Bar收盘跳空", "formula_note": "open/prev_close - 1"},
    {"feature": "true_range_ticks", "group": "price_structure", "description": "真实波幅tick数", "formula_note": "max(high-low, abs(high-prev_close), abs(low-prev_close))/tick_size"},
    {"feature": "range_expansion_20", "group": "price_structure", "description": "真实波幅相对20期均值扩张", "formula_note": "true_range_ticks/rolling_mean_20 - 1"},
    {"feature": "body_ratio", "group": "price_structure", "description": "实体占Bar振幅比例", "formula_note": "abs(close-open)/(high-low)"},
    {"feature": "close_location_value", "group": "price_structure", "description": "收盘在Bar内部的位置，范围约为[-1,1]", "formula_note": "((close-low)-(high-close))/(high-low)"},
    {"feature": "upper_shadow_ratio", "group": "price_structure", "description": "上影线占比", "formula_note": "upper_shadow/(high-low)"},
    {"feature": "lower_shadow_ratio", "group": "price_structure", "description": "下影线占比", "formula_note": "lower_shadow/(high-low)"},
    {"feature": "wick_imbalance", "group": "price_structure", "description": "上下影线不平衡", "formula_note": "(lower_shadow-upper_shadow)/(high-low)"},
    {"feature": "delta_strength", "group": "orderflow_delta", "description": "Delta占成交量比例", "formula_note": "delta/volume"},
    {"feature": "abs_delta_strength", "group": "orderflow_delta", "description": "Delta强度绝对值", "formula_note": "abs(delta_strength)"},
    {"feature": "delta_zscore_20", "group": "orderflow_delta", "description": "Delta 20期标准分", "formula_note": "rolling_zscore(delta, 20)"},
    {"feature": "delta_zscore_60", "group": "orderflow_delta", "description": "Delta 60期标准分", "formula_note": "rolling_zscore(delta, 60)"},
    {"feature": "delta_change_1", "group": "orderflow_delta", "description": "Delta一阶变化", "formula_note": "delta - delta.shift(1)"},
    {"feature": "delta_accel_3", "group": "orderflow_delta", "description": "Delta三期加速度", "formula_note": "delta.diff(1) - delta.diff(1).shift(3)"},
    {"feature": "cvd_change_5", "group": "orderflow_delta", "description": "5期累计Delta变化", "formula_note": "cumsum(delta) - shift(5)"},
    {"feature": "cvd_change_10", "group": "orderflow_delta", "description": "10期累计Delta变化", "formula_note": "cumsum(delta) - shift(10)"},
    {"feature": "cvd_change_20", "group": "orderflow_delta", "description": "20期累计Delta变化", "formula_note": "cumsum(delta) - shift(20)"},
    {"feature": "cvd_slope_20", "group": "orderflow_delta", "description": "20期CVD平均斜率", "formula_note": "(cvd-cvd.shift(20))/20"},
    {"feature": "delta_price_agreement", "group": "orderflow_delta", "description": "价格方向与Delta方向一致性", "formula_note": "sign(close-open)*sign(delta)"},
    {"feature": "delta_price_divergence", "group": "orderflow_delta", "description": "价格与Delta背离强度", "formula_note": "abs(delta_strength) when sign(prev-return)*sign(delta)<0"},
    {"feature": "poc_distance_ticks", "group": "auction_poc", "description": "收盘价相对POC距离tick数", "formula_note": "(close-poc)/tick_size"},
    {"feature": "abs_poc_distance_ticks", "group": "auction_poc", "description": "收盘价相对POC距离绝对值", "formula_note": "abs(poc_distance_ticks)"},
    {"feature": "poc_distance_zscore_20", "group": "auction_poc", "description": "POC距离20期标准分", "formula_note": "rolling_zscore(poc_distance_ticks, 20)"},
    {"feature": "poc_shift_ticks", "group": "auction_poc", "description": "POC相对上一Bar位移tick数", "formula_note": "(poc-poc.shift(1))/tick_size"},
    {"feature": "poc_shift_zscore_20", "group": "auction_poc", "description": "POC位移20期标准分", "formula_note": "rolling_zscore(poc_shift_ticks, 20)"},
    {"feature": "close_above_poc", "group": "auction_poc", "description": "收盘价在POC上方/下方", "formula_note": "sign(close-poc)"},
    {"feature": "poc_price_agreement", "group": "auction_poc", "description": "POC位移方向与价格方向一致性", "formula_note": "sign(close-prev_close)*sign(poc_shift_ticks)"},
    {"feature": "value_acceptance_ratio", "group": "auction_poc", "description": "收盘价贴近POC的接受度", "formula_note": "1 - abs(close-poc)/(high-low)"},
    {"feature": "volume_zscore_20", "group": "volume_liquidity", "description": "成交量20期标准分", "formula_note": "rolling_zscore(volume, 20)"},
    {"feature": "volume_zscore_60", "group": "volume_liquidity", "description": "成交量60期标准分", "formula_note": "rolling_zscore(volume, 60)"},
    {"feature": "volume_change_1", "group": "volume_liquidity", "description": "成交量一阶变化率", "formula_note": "volume/volume.shift(1)-1"},
    {"feature": "volume_burst_flag", "group": "volume_liquidity", "description": "成交量异常放大标记", "formula_note": "1 if volume_zscore_20 > 2 else 0"},
    {"feature": "range_per_volume", "group": "volume_liquidity", "description": "单位成交量带来的价格振幅", "formula_note": "true_range_ticks/volume"},
    {"feature": "return_per_volume", "group": "volume_liquidity", "description": "单位成交量带来的有方向收益", "formula_note": "close_to_prev_close_return/volume"},
    {"feature": "amihud_like_illiquidity", "group": "volume_liquidity", "description": "类Amihud非流动性", "formula_note": "abs(close_to_prev_close_return)/volume"},
    {"feature": "volume_adjusted_delta", "group": "volume_liquidity", "description": "相对近期成交量的Delta", "formula_note": "delta/rolling_mean(volume, 20)"},
    {"feature": "open_interest_change", "group": "open_interest", "description": "持仓量变化", "formula_note": "open_interest.diff()"},
    {"feature": "open_interest_change_pct", "group": "open_interest", "description": "持仓量变化率", "formula_note": "open_interest.diff()/open_interest.shift(1)"},
    {"feature": "open_interest_zscore_20", "group": "open_interest", "description": "持仓变化20期标准分", "formula_note": "rolling_zscore(open_interest_change, 20)"},
    {"feature": "open_interest_zscore_60", "group": "open_interest", "description": "持仓变化60期标准分", "formula_note": "rolling_zscore(open_interest_change, 60)"},
    {"feature": "price_oi_state", "group": "open_interest", "description": "价格方向与持仓变化组合状态", "formula_note": "sign(close-prev_close)*sign(open_interest_change)"},
    {"feature": "delta_oi_agreement", "group": "open_interest", "description": "Delta方向与持仓变化一致性", "formula_note": "sign(delta)*sign(open_interest_change)"},
    {"feature": "oi_volume_ratio", "group": "open_interest", "description": "持仓变化占成交量比例", "formula_note": "open_interest_change/volume"},
    {"feature": "open_interest_fracdiff_04", "group": "open_interest", "description": "持仓量0.4阶分数差分，降低非平稳性同时保留长记忆", "formula_note": "fractional_diff(open_interest, d=0.4)"},
    {"feature": "open_interest_fracdiff_zscore_60", "group": "open_interest", "description": "持仓量分数差分60期标准分", "formula_note": "rolling_zscore(open_interest_fracdiff_04, 60)"},
    {"feature": "fracdiff_oi_delta_confirm", "group": "open_interest", "description": "分数差分持仓方向与Delta方向确认", "formula_note": "sign(open_interest_fracdiff_04)*sign(delta)"},
    {"feature": "trend_return_12", "group": "regime_time", "description": "12期趋势收益", "formula_note": "close/close.shift(12)-1"},
    {"feature": "trend_return_48", "group": "regime_time", "description": "48期趋势收益", "formula_note": "close/close.shift(48)-1"},
    {"feature": "trend_strength_48", "group": "regime_time", "description": "波动调整后的48期趋势强度", "formula_note": "trend_return_48/(volatility_60*sqrt(48))"},
    {"feature": "volatility_20", "group": "regime_time", "description": "20期收益波动率", "formula_note": "rolling_std(close.pct_change(), 20)"},
    {"feature": "volatility_60", "group": "regime_time", "description": "60期收益波动率", "formula_note": "rolling_std(close.pct_change(), 60)"},
    {"feature": "volatility_ratio_20_60", "group": "regime_time", "description": "短长波动率比值", "formula_note": "volatility_20/volatility_60"},
    {"feature": "is_night_session", "group": "regime_time", "description": "夜盘标记", "formula_note": "1 if hour>=21 or hour<9 else 0"},
    {"feature": "minute_of_day_sin", "group": "regime_time", "description": "日内时间正弦编码", "formula_note": "sin(2*pi*minute/1440)"},
    {"feature": "minute_of_day_cos", "group": "regime_time", "description": "日内时间余弦编码", "formula_note": "cos(2*pi*minute/1440)"},
    {"feature": "flow_at_bar_extreme", "group": "interaction", "description": "订单流在Bar极端位置的确认", "formula_note": "delta_strength*close_location_value"},
    {"feature": "absorption_score", "group": "interaction", "description": "强Delta但价格反向的吸收分数", "formula_note": "signed abs(delta_strength) on price-delta divergence"},
    {"feature": "exhaustion_score", "group": "interaction", "description": "放量且收在极端位置的衰竭线索", "formula_note": "abs(delta_strength)*abs(close_location_value)*max(volume_zscore_20,0)"},
    {"feature": "trend_delta_confirm", "group": "interaction", "description": "趋势方向与Delta强度确认", "formula_note": "sign(trend_return_12)*delta_strength"},
    {"feature": "poc_rejection_score", "group": "interaction", "description": "偏离POC后的拒绝/延续线索", "formula_note": "poc_distance_zscore_20*close_location_value"},
    {"feature": "volume_delta_confirm", "group": "interaction", "description": "成交量与Delta共同确认", "formula_note": "volume_zscore_20*delta_strength"},
    {"feature": "oi_delta_confirm", "group": "interaction", "description": "持仓变化与Delta共同确认", "formula_note": "open_interest_zscore_20*delta_strength"},
    {"feature": "volatility_adjusted_flow", "group": "interaction", "description": "波动调整后的订单流强度", "formula_note": "delta_strength/volatility_20"},
]

FEATURE_COLUMNS = [item["feature"] for item in FEATURE_SPECS]
FEATURE_GROUP_MAP = {item["feature"]: item["group"] for item in FEATURE_SPECS}


def feature_catalog() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_SPECS)


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill: Optional[float] = np.nan) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.fillna(fill) if fill is not None else out


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def fractional_diff(series: pd.Series, order: float = 0.4, threshold: float = 1e-4, max_size: int = 200) -> pd.Series:
    """固定宽度分数阶差分，用于降低长周期非平稳性。

    权重只依赖历史值，按单合约调用，因此不会引入跨合约污染或未来函数。
    """

    weights = [1.0]
    for k in range(1, max_size):
        weight = -weights[-1] * (order - k + 1.0) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
    width = len(weights)
    out = np.full(len(series), np.nan, dtype=float)
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if len(values) < width:
        return pd.Series(out, index=series.index)

    windows = np.lib.stride_tricks.sliding_window_view(values, width)
    valid = ~np.isnan(windows).any(axis=1)
    ordered_weights = np.asarray(weights[::-1], dtype=float)
    diffed = np.full(len(windows), np.nan, dtype=float)
    diffed[valid] = windows[valid] @ ordered_weights
    out[width - 1 :] = diffed
    return pd.Series(out, index=series.index)


def add_features_for_contract(group: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    """按单合约生成特征，防止 rolling 和 diff 跨合约污染。"""

    g = group.copy()
    prev_close = g["close"].shift(1)
    price_range = g["high"] - g["low"]
    body = (g["close"] - g["open"]).abs()
    upper_shadow = g["high"] - np.maximum(g["open"], g["close"])
    lower_shadow = np.minimum(g["open"], g["close"]) - g["low"]
    true_range = pd.concat([(g["high"] - g["low"]), (g["high"] - prev_close).abs(), (g["low"] - prev_close).abs()], axis=1).max(axis=1)
    close_return = g["close"].pct_change()
    price_direction = np.sign(g["close"] - prev_close)
    bar_direction = np.sign(g["close"] - g["open"])
    delta_direction = np.sign(g["delta"])
    cvd = g["delta"].cumsum()
    oi_change = g["open_interest"].diff()

    g["bar_return"] = g["close"] / g["open"].replace(0, np.nan) - 1
    g["close_to_prev_close_return"] = g["close"] / prev_close.replace(0, np.nan) - 1
    g["gap_return"] = g["open"] / prev_close.replace(0, np.nan) - 1
    g["true_range_ticks"] = true_range / tick_size
    g["range_expansion_20"] = safe_divide(g["true_range_ticks"], g["true_range_ticks"].rolling(20, min_periods=20).mean(), fill=None) - 1
    g["body_ratio"] = safe_divide(body, price_range, fill=0.0)
    g["close_location_value"] = safe_divide((g["close"] - g["low"]) - (g["high"] - g["close"]), price_range, fill=0.0)
    g["upper_shadow_ratio"] = safe_divide(upper_shadow, price_range, fill=0.0)
    g["lower_shadow_ratio"] = safe_divide(lower_shadow, price_range, fill=0.0)
    g["wick_imbalance"] = safe_divide(lower_shadow - upper_shadow, price_range, fill=0.0)

    g["delta_strength"] = safe_divide(g["delta"], g["volume"], fill=0.0)
    g["abs_delta_strength"] = g["delta_strength"].abs()
    g["delta_zscore_20"] = rolling_zscore(g["delta"], 20)
    g["delta_zscore_60"] = rolling_zscore(g["delta"], 60)
    g["delta_change_1"] = g["delta"].diff()
    g["delta_accel_3"] = g["delta"].diff() - g["delta"].diff().shift(3)
    for window in (5, 10, 20):
        g[f"cvd_change_{window}"] = cvd - cvd.shift(window)
    g["cvd_slope_20"] = (cvd - cvd.shift(20)) / 20.0
    g["delta_price_agreement"] = bar_direction * delta_direction
    g["delta_price_divergence"] = np.where(price_direction * delta_direction < 0, g["abs_delta_strength"], 0.0)

    g["poc_distance_ticks"] = (g["close"] - g["poc"]) / tick_size
    g["abs_poc_distance_ticks"] = g["poc_distance_ticks"].abs()
    g["poc_distance_zscore_20"] = rolling_zscore(g["poc_distance_ticks"], 20)
    g["poc_shift_ticks"] = (g["poc"] - g["poc"].shift(1)) / tick_size
    g["poc_shift_zscore_20"] = rolling_zscore(g["poc_shift_ticks"], 20)
    g["close_above_poc"] = np.sign(g["close"] - g["poc"])
    g["poc_price_agreement"] = price_direction * np.sign(g["poc_shift_ticks"])
    g["value_acceptance_ratio"] = 1.0 - safe_divide((g["close"] - g["poc"]).abs(), price_range, fill=0.0)

    g["volume_zscore_20"] = rolling_zscore(g["volume"], 20)
    g["volume_zscore_60"] = rolling_zscore(g["volume"], 60)
    g["volume_change_1"] = g["volume"] / g["volume"].shift(1).replace(0, np.nan) - 1
    g["volume_burst_flag"] = (g["volume_zscore_20"] > 2.0).astype(float)
    g["range_per_volume"] = safe_divide(g["true_range_ticks"], g["volume"], fill=0.0)
    g["return_per_volume"] = safe_divide(g["close_to_prev_close_return"], g["volume"], fill=0.0)
    g["amihud_like_illiquidity"] = safe_divide(g["close_to_prev_close_return"].abs(), g["volume"], fill=0.0)
    g["volume_adjusted_delta"] = safe_divide(g["delta"], g["volume"].rolling(20, min_periods=20).mean(), fill=None)

    g["open_interest_change"] = oi_change
    g["open_interest_change_pct"] = safe_divide(oi_change, g["open_interest"].shift(1), fill=None)
    g["open_interest_zscore_20"] = rolling_zscore(oi_change, 20)
    g["open_interest_zscore_60"] = rolling_zscore(oi_change, 60)
    g["price_oi_state"] = price_direction * np.sign(oi_change)
    g["delta_oi_agreement"] = delta_direction * np.sign(oi_change)
    g["oi_volume_ratio"] = safe_divide(oi_change, g["volume"], fill=0.0)
    g["open_interest_fracdiff_04"] = fractional_diff(g["open_interest"], order=0.4)
    g["open_interest_fracdiff_zscore_60"] = rolling_zscore(g["open_interest_fracdiff_04"], 60)
    g["fracdiff_oi_delta_confirm"] = np.sign(g["open_interest_fracdiff_04"]) * delta_direction

    g["trend_return_12"] = g["close"] / g["close"].shift(12).replace(0, np.nan) - 1
    g["trend_return_48"] = g["close"] / g["close"].shift(48).replace(0, np.nan) - 1
    g["volatility_20"] = close_return.rolling(20, min_periods=20).std(ddof=0)
    g["volatility_60"] = close_return.rolling(60, min_periods=60).std(ddof=0)
    g["trend_strength_48"] = safe_divide(g["trend_return_48"], g["volatility_60"] * math.sqrt(48), fill=None)
    g["volatility_ratio_20_60"] = safe_divide(g["volatility_20"], g["volatility_60"], fill=None)
    minutes = g["datetime"].dt.hour * 60 + g["datetime"].dt.minute
    angle = 2.0 * math.pi * minutes / 1440.0
    g["is_night_session"] = ((g["datetime"].dt.hour >= 21) | (g["datetime"].dt.hour < 9)).astype(float)
    g["minute_of_day_sin"] = np.sin(angle)
    g["minute_of_day_cos"] = np.cos(angle)

    g["flow_at_bar_extreme"] = g["delta_strength"] * g["close_location_value"]
    g["absorption_score"] = np.where(price_direction * delta_direction < 0, g["abs_delta_strength"] * delta_direction, 0.0)
    g["exhaustion_score"] = g["abs_delta_strength"] * g["close_location_value"].abs() * g["volume_zscore_20"].clip(lower=0.0)
    g["trend_delta_confirm"] = np.sign(g["trend_return_12"]) * g["delta_strength"]
    g["poc_rejection_score"] = g["poc_distance_zscore_20"] * g["close_location_value"]
    g["volume_delta_confirm"] = g["volume_zscore_20"] * g["delta_strength"]
    g["oi_delta_confirm"] = g["open_interest_zscore_20"] * g["delta_strength"]
    g["volatility_adjusted_flow"] = safe_divide(g["delta_strength"], g["volatility_20"], fill=None)
    return g


def build_raw_features(bars: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    frames = [add_features_for_contract(group, tick_size) for _, group in bars.groupby("contract", sort=True)]
    out = pd.concat(frames, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _fit_one_transform(train_values: pd.Series) -> Dict[str, Any]:
    values = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"lower": np.nan, "upper": np.nan, "center": np.nan, "scale": np.nan, "scale_type": "none"}
    lower = float(values.quantile(0.01))
    upper = float(values.quantile(0.99))
    clipped = values.clip(lower=lower, upper=upper)
    median = float(clipped.median())
    mad = float((clipped - median).abs().median())
    if math.isfinite(mad) and mad > 0:
        return {"lower": lower, "upper": upper, "center": median, "scale": 1.4826 * mad, "scale_type": "mad"}
    mean = float(clipped.mean())
    std = float(clipped.std(ddof=0))
    if math.isfinite(std) and std > 0:
        return {"lower": lower, "upper": upper, "center": mean, "scale": std, "scale_type": "std"}
    return {"lower": lower, "upper": upper, "center": median, "scale": np.nan, "scale_type": "constant"}


def fit_transform_features(features: pd.DataFrame, config: Stage2Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """用训练集拟合每个合约的 winsorize/标准化参数，并返回裁剪后的特征。"""

    cleaned = features.copy()
    rows: List[Dict[str, Any]] = []
    train_all_mask = cleaned["datetime"] <= config.train_end
    global_params: Dict[str, Dict[str, Any]] = {}
    for feature in FEATURE_COLUMNS:
        params = _fit_one_transform(cleaned.loc[train_all_mask, feature])
        params.update({"contract": "__GLOBAL__", "feature": feature, "param_source": "global_train"})
        global_params[feature] = params.copy()
        rows.append(params)

    for contract, idx in cleaned.groupby("contract").groups.items():
        contract_idx = list(idx)
        train_mask = (cleaned.loc[contract_idx, "datetime"] <= config.train_end)
        for feature in FEATURE_COLUMNS:
            params = _fit_one_transform(cleaned.loc[contract_idx, feature][train_mask])
            if params["scale_type"] == "none" or pd.isna(params["lower"]) or pd.isna(params["upper"]):
                params = global_params[feature].copy()
                params["param_source"] = "global_fallback_no_contract_train"
            else:
                params["param_source"] = "contract_train"
            params.update({"contract": contract, "feature": feature})
            rows.append(params)
            if pd.notna(params["lower"]) and pd.notna(params["upper"]):
                cleaned.loc[contract_idx, feature] = pd.to_numeric(cleaned.loc[contract_idx, feature], errors="coerce").clip(params["lower"], params["upper"])
    return cleaned, pd.DataFrame(rows)


def standardize_selected_features(features: pd.DataFrame, transform_params: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """按训练集拟合出的 center/scale 生成 zscore，用于阶段二信号组合。"""

    z = pd.DataFrame(index=features.index)
    param_map = {
        (row["contract"], row["feature"]): row
        for _, row in transform_params.iterrows()
        if row["feature"] in selected_features
    }
    for feature in selected_features:
        values = pd.Series(np.nan, index=features.index, dtype=float)
        for contract, idx in features.groupby("contract").groups.items():
            params = param_map.get((contract, feature))
            if params is None or pd.isna(params["scale"]) or params["scale"] == 0:
                continue
            raw = pd.to_numeric(features.loc[idx, feature], errors="coerce")
            values.loc[idx] = (raw - float(params["center"])) / float(params["scale"])
        z[feature] = values
    return z.replace([np.inf, -np.inf], np.nan)
