"""阶段二配置与 CLI 参数。

这里集中保存路径、切分日期、交易假设和特征选择约束，避免各模块
隐式复制常量。阶段二仍然使用单次训练/测试切分，walk-forward 留到阶段三。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


DEFAULT_INPUT_DIR = Path("/Users/wangrendong/Projects/BlueBlue/orderflow_data/SHFE.cu")
DEFAULT_OUTPUT_DIR = Path("/Users/wangrendong/Projects/BlueBlue/stage2_outputs/SHFE.cu")
DEFAULT_START_DATE = "1900-01-01"
DEFAULT_TRAIN_END = None
DEFAULT_TEST_START = None
DEFAULT_SPLIT_TRAIN_RATIO = 0.8

BASE_COLUMNS = [
    "contract",
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "poc",
    "delta",
    "open_interest",
]

LABEL_COLUMNS = [
    "future_return_1",
    "future_return_3",
    "future_return_5",
    "future_return_10",
    "mfe_10_long",
    "mae_10_long",
    "entry_open_next",
    "exit_close_after_5",
    "trade_return_5_gross",
    "trade_return_5_net",
]


@dataclass(frozen=True)
class Stage2Config:
    """阶段二运行配置。

    所有日期边界都在入口处转成 Timestamp，后续模块只消费明确的配置对象。
    """

    input_dir: Path
    output_dir: Path
    start_date: pd.Timestamp
    train_end: Optional[pd.Timestamp]
    test_start: Optional[pd.Timestamp]
    primary_horizon: int
    holding_bars: int
    cost_ticks_roundtrip: float
    tick_size: float
    contracts: Optional[List[str]]
    max_selected_features: int
    corr_threshold: float
    max_group_share: float
    min_selected_groups: int
    long_quantile: float
    short_quantile: float
    split_train_ratio: float

    @property
    def label_col(self) -> str:
        return f"future_return_{self.primary_horizon}"

    @property
    def roundtrip_cost_price(self) -> float:
        return self.cost_ticks_roundtrip * self.tick_size


def normalize_contract(contract: str) -> str:
    """把 cu2604 这类短名规范化成 SHFE.cu2604。"""

    contract = contract.strip()
    return contract if contract.startswith("SHFE.") else f"SHFE.{contract}"


def parse_contracts(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [normalize_contract(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> Stage2Config:
    parser = argparse.ArgumentParser(description="Stage 2 low-redundancy order-flow research pipeline")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start_date", type=str, default=DEFAULT_START_DATE, help="默认1900-01-01，即读取全部可用历史")
    parser.add_argument("--train_end", type=str, default=DEFAULT_TRAIN_END, help="可选；不填时按全样本时间前80%自动切分")
    parser.add_argument("--test_start", type=str, default=DEFAULT_TEST_START, help="可选；不填时自动取train_end之后第一根Bar")
    parser.add_argument("--primary_horizon", type=int, default=5)
    parser.add_argument("--holding_bars", type=int, default=5)
    parser.add_argument("--cost_ticks_roundtrip", type=float, default=2.0)
    parser.add_argument("--tick_size", type=float, default=10.0)
    parser.add_argument("--contracts", type=str, default=None)
    parser.add_argument("--max_selected_features", type=int, default=18)
    parser.add_argument("--corr_threshold", type=float, default=0.85)
    parser.add_argument("--max_group_share", type=float, default=0.45)
    parser.add_argument("--min_selected_groups", type=int, default=5)
    parser.add_argument("--long_quantile", type=float, default=0.85)
    parser.add_argument("--short_quantile", type=float, default=0.15)
    parser.add_argument("--split_train_ratio", type=float, default=DEFAULT_SPLIT_TRAIN_RATIO)
    args = parser.parse_args(argv)

    return Stage2Config(
        input_dir=Path(args.input_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        start_date=pd.Timestamp(args.start_date),
        train_end=pd.Timestamp(args.train_end) if args.train_end else None,
        test_start=pd.Timestamp(args.test_start) if args.test_start else None,
        primary_horizon=args.primary_horizon,
        holding_bars=args.holding_bars,
        cost_ticks_roundtrip=args.cost_ticks_roundtrip,
        tick_size=args.tick_size,
        contracts=parse_contracts(args.contracts),
        max_selected_features=args.max_selected_features,
        corr_threshold=args.corr_threshold,
        max_group_share=args.max_group_share,
        min_selected_groups=args.min_selected_groups,
        long_quantile=args.long_quantile,
        short_quantile=args.short_quantile,
        split_train_ratio=args.split_train_ratio,
    )


def resolve_time_split(config: Stage2Config, bars: pd.DataFrame) -> Stage2Config:
    """在读取全部历史后解析单次训练/测试切分。

    默认按日历时间跨度做 80/20 切分，十年数据约等于前八年训练、后两年测试。
    用户显式传入 `--train_end/--test_start` 时，以用户指定为准。
    """

    if config.train_end is not None and config.test_start is not None:
        return config

    times = bars["datetime"].dropna().sort_values()
    if times.empty:
        raise ValueError("Cannot resolve train/test split because bars datetime is empty")

    min_time = times.iloc[0]
    max_time = times.iloc[-1]
    ratio = min(max(config.split_train_ratio, 0.1), 0.9)
    cutoff = min_time + (max_time - min_time) * ratio

    resolved_train_end = config.train_end
    if resolved_train_end is None:
        train_candidates = times[times <= cutoff]
        resolved_train_end = train_candidates.iloc[-1] if not train_candidates.empty else times.iloc[int(len(times) * ratio) - 1]

    resolved_test_start = config.test_start
    if resolved_test_start is None:
        test_candidates = times[times > resolved_train_end]
        if test_candidates.empty:
            raise ValueError("Cannot resolve test_start because no bars exist after train_end")
        resolved_test_start = test_candidates.iloc[0]

    return replace(config, train_end=pd.Timestamp(resolved_train_end), test_start=pd.Timestamp(resolved_test_start))
