#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tick → 订单流聚合（支持命令行传参 + 动态文件名命名 + 品种归档目录自动创建）

【升级版 v3】：
1. 移除硬编码参数，支持通过 --input_csv 和 --window_min 从外部输入。
2. 动态捕获聚合后数据的开始时间、结束时间和总条数。
3. 按照格式 period_of_{WINDOW_MIN}_{start}_{end}_{contract}_{count}.csv 命名。
4. 自动创建并保存至 ./orderflow_data/{symbol}/ 目录下。
"""
import os
import sys
import re
import datetime as _dt
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from config_loader import SYMBOL_INFO

DEFAULT_INPUT_DIR = Path("/Users/wangrendong/Projects/QuotesOfTianqin/Ticks/SHFE.cu")
DEFAULT_OUTPUT_DIR = Path("/Users/wangrendong/Projects/BlueBlue/orderflow_data/SHFE.cu")
DEFAULT_REFERENCE_DIR = Path("/Users/wangrendong/Projects/tmp")
OUTPUT_COLUMNS = ["datetime", "open", "high", "low", "close", "volume", "poc", "delta", "buy_imbalance", "sell_imbalance", "open_interest"]
FLOAT_COLUMNS = ["open", "high", "low", "close", "poc"]
INT_COLUMNS = ["volume", "delta", "open_interest"]
TEXT_COLUMNS = ["datetime", "buy_imbalance", "sell_imbalance"]
VALIDATION_REPORT_COLUMNS = ["contract", "input_csv", "generated_csv", "reference_csv", "status", "generated_rows", "reference_rows", "mismatch_count", "first_mismatch_column", "first_mismatch_row", "first_generated_value", "first_reference_value"]
FLOAT_TOL = 1e-6

# ========== 交易时段与基础工具 ==========
def _parse_time_str(s: str) -> Tuple[int, int, int]:
    parts = s.strip().split(":")
    return int(parts[0]), int(parts[1]), int(parts[2])

def _build_scheme_from_symbol(symbol_key_upper: str) -> List[Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]]:
    for k, v in SYMBOL_INFO.items():
        if k.upper() == symbol_key_upper:
            day_sessions = v.get("day", [])
            night_sessions = v.get("night", [])
            all_sessions = day_sessions + night_sessions
            scheme = []
            for seg in all_sessions:
                start_str, end_str = seg
                sh, sm, ss = _parse_time_str(start_str)
                eh, em, es = _parse_time_str(end_str)
                scheme.append(((sh, sm, ss, 0), (eh, em, es, 0)))
            return scheme
    return [
        ((9, 0, 0, 0),    (10, 15, 0, 0)), ((10, 30, 0, 0),  (11, 30, 0, 0)),
        ((13, 30, 0, 0),  (15, 0, 0, 0)),  ((21, 0, 0, 0),   (23, 0, 0, 0)),
    ]

def extract_contract_from_filename(csv_path: str):
    stem = Path(csv_path).stem
    match = re.match(r"^(.+?)_ticks?(?:_\d{8}_\d{6})?$", stem)
    if not match:
        match = re.match(r"^(.+?)_tick.*$", stem)
        if not match:
            raise ValueError(f"文件名格式不符合预期：{csv_path}")

    contract_id = match.group(1)
    if "." in contract_id:
        master = contract_id
        underlying = contract_id.split(".")[-1]
    else:
        master = contract_id
        underlying = contract_id

    symbol_match = re.match(r"^([A-Za-z]+)", underlying)
    if not symbol_match:
        raise ValueError(f"无法提取品种字母: '{underlying}'")

    symbol = symbol_match.group(1)

    if symbol not in SYMBOL_INFO:
        raise ValueError(f"品种 '{symbol}' 未配置")
    one_tick = SYMBOL_INFO[symbol]["tick"]
    return symbol, master, one_tick

def _is_valid_price_row(row, strict_zero_block: bool = True) -> bool:
    key_vals = [row.get("last_price"), row.get("highest"), row.get("lowest"), row.get("bid_price1"), row.get("ask_price1")]
    if any(pd.isna(v) for v in key_vals): return False
    if strict_zero_block and any(float(v) == 0 for v in key_vals): return False
    try:
        if float(row.get("highest")) < float(row.get("lowest")): return False
    except Exception:
        return False
    return True

def filter_dirty_ticks(df: pd.DataFrame, symbol: str, master_symbol: str, session_scheme: List, *, strict_zero_block: bool = True) -> pd.DataFrame:
    ts = pd.to_datetime(df["datetime"], errors="coerce")
    ms = pd.to_numeric(df["datetime_nano"], errors="coerce") / 1_000_000.0

    basic_ok = (ts.notna() & ms.notna() & (ms > 0) & pd.to_numeric(df["volume"], errors="coerce").fillna(-1).ge(0))
    price_ok = df.apply(lambda r: _is_valid_price_row(r, strict_zero_block=strict_zero_block), axis=1)

    cache: Dict[_dt.date, SessionSet] = {}
    def in_sessions(i: int) -> bool:
        if not basic_ok.iat[i] or not price_ok.iat[i]: return False
        dt = ts.iat[i]
        d = dt.date()
        if dt.hour < 8: d = d - _dt.timedelta(days=1)
        ss = cache.get(d)
        if ss is None:
            ss = SessionSet.for_day(d, session_scheme)
            cache[d] = ss
        return ss.current(ms.iat[i]) is not None

    mask = pd.Series([in_sessions(i) for i in range(len(df))], index=df.index)
    return df.loc[mask].reset_index(drop=True)

def _strip_symbol_prefix(col: str, master_symbol: str) -> str:
    prefix = f"{master_symbol}."
    return col[len(prefix):] if str(col).startswith(prefix) else str(col)

def _normalize_tick_columns(df: pd.DataFrame, master_symbol: str) -> pd.DataFrame:
    df = df.rename(columns={c: _strip_symbol_prefix(c, master_symbol) for c in df.columns})
    num_cols = ["datetime_nano", "last_price", "highest", "lowest", "average", "volume", "amount", "open_interest", "bid_price1", "bid_volume1", "ask_price1", "ask_volume1"]
    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = df[num_cols].ffill().fillna(0)
    df["datetime"] = df["datetime"].astype(str)
    return df

def _resolve_symbol_key(symbol: str, master_symbol: str) -> str:
    if symbol: return ''.join([c for c in symbol if c.isalpha()]).upper()
    if master_symbol and '.' in master_symbol: return ''.join([c for c in master_symbol.split('.')[-1] if c.isalpha()]).upper()
    return symbol or master_symbol

# ========== Tick 方向 & 类型枚举 ==========
class OpenInterestDeltaForwardEnum(Enum): OPEN = "Open"; CLOSE = "Close"; EXCHANGE = "Exchange"; NONE = "None"; OPENFWDOUBLE = "OpenFwDouble"; CLOSEFWDOUBLE = "CloseFwDOuble"
class OrderForwardEnum(Enum): UP = "Up"; DOWN = "Down"; MIDDLE = "Middle"
class TickTypeEnum(Enum):
    OPENLONG = "OpenLong"; OPENSHORT = "OpenShort"; OPENDOUBLE = "OpenDouble"; CLOSELONG = "CloseLong"
    CLOSESHORT = "CloseShort"; CLOSEDOUBLE = "Double"; EXCHANGELONG = "ExchangeLong"
    EXCHANGESHORT = "ExchangeShort"; OPENUNKOWN = "OpenUnkown"; CLOSEUNKOWN = "CloseUnkown"
    EXCHANGEUNKOWN = "ExchangeUnkown"; UNKOWN = "Unkown"; NOCHANGE = "NoChange"
class TickTypeKeyEnum(Enum): TICKTYPE = "TickType"

tick_type_cal_dict = {
    OpenInterestDeltaForwardEnum.NONE: {
        OrderForwardEnum.UP: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.NOCHANGE, "DESC": "未知"},
        OrderForwardEnum.DOWN: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.NOCHANGE, "DESC": "未知"},
        OrderForwardEnum.MIDDLE: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.NOCHANGE, "DESC": "未知"},
    },
    OpenInterestDeltaForwardEnum.EXCHANGE: {
        OrderForwardEnum.UP: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.EXCHANGELONG, "DESC": "多换"},
        OrderForwardEnum.DOWN: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.EXCHANGESHORT, "DESC": "空换"},
        OrderForwardEnum.MIDDLE: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.NOCHANGE, "DESC": "换未知"},
    },
    OpenInterestDeltaForwardEnum.OPENFWDOUBLE: {
        OrderForwardEnum.UP: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.OPENDOUBLE, "DESC": "双开"},
        OrderForwardEnum.DOWN: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.OPENDOUBLE, "DESC": "双开"},
        OrderForwardEnum.MIDDLE: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.OPENDOUBLE, "DESC": "双开"},
    },
    OpenInterestDeltaForwardEnum.OPEN: {
        OrderForwardEnum.UP: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.OPENLONG, "DESC": "多开"},
        OrderForwardEnum.DOWN: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.OPENSHORT, "DESC": "空开"},
        OrderForwardEnum.MIDDLE: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.OPENUNKOWN, "DESC": "开仓未知"},
    },
    OpenInterestDeltaForwardEnum.CLOSEFWDOUBLE: {
        OrderForwardEnum.UP: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.CLOSEDOUBLE, "DESC": "双平"},
        OrderForwardEnum.DOWN: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.CLOSEDOUBLE, "DESC": "双平"},
        OrderForwardEnum.MIDDLE: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.CLOSEDOUBLE, "DESC": "双平"},
    },
    OpenInterestDeltaForwardEnum.CLOSE: {
        OrderForwardEnum.UP: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.CLOSESHORT, "DESC": "空平"},
        OrderForwardEnum.DOWN: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.CLOSELONG, "DESC": "多平"},
        OrderForwardEnum.MIDDLE: {TickTypeKeyEnum.TICKTYPE: TickTypeEnum.CLOSEUNKOWN, "DESC": "平未知"},
    },
}

class TickAnalysis:
    @staticmethod
    def get_order_forward(last_price, ask_price1, bid_price1, pre_last_price, pre_ask_price1, pre_bid_price1) -> OrderForwardEnum:
        if last_price >= pre_ask_price1: return OrderForwardEnum.UP
        elif last_price <= pre_bid_price1: return OrderForwardEnum.DOWN
        else:
            if last_price >= ask_price1: return OrderForwardEnum.UP
            elif last_price <= bid_price1: return OrderForwardEnum.DOWN
            else: return OrderForwardEnum.MIDDLE

    @staticmethod
    def get_open_interest_delta_forward(oi_delta: int, vol_delta: int) -> OpenInterestDeltaForwardEnum:
        if oi_delta == 0 and vol_delta == 0: return OpenInterestDeltaForwardEnum.NONE
        elif oi_delta == 0 and vol_delta > 0: return OpenInterestDeltaForwardEnum.EXCHANGE
        elif oi_delta > 0: return OpenInterestDeltaForwardEnum.OPENFWDOUBLE if (oi_delta - vol_delta == 0) else OpenInterestDeltaForwardEnum.OPEN
        else: return OpenInterestDeltaForwardEnum.CLOSEFWDOUBLE if (oi_delta + vol_delta == 0) else OpenInterestDeltaForwardEnum.CLOSE

# ========== 会话时段 ==========
@dataclass
class Session:
    begin_ms: float
    end_ms: float
    def contains(self, ms: float) -> bool: return self.begin_ms <= ms <= self.end_ms

@dataclass
class SessionSet:
    sessions: List[Session]
    @staticmethod
    def for_day(day: _dt.date, scheme: List) -> "SessionSet":
        S: List[Session] = []
        for (bh,bm,bs,bbias), (eh,em,es,ebias) in scheme:
            b_dt = _dt.datetime.combine(day, _dt.time(bh,bm,bs)) + _dt.timedelta(milliseconds=bbias)
            e_dt = _dt.datetime.combine(day, _dt.time(eh,em,es)) + _dt.timedelta(milliseconds=ebias)
            if e_dt <= b_dt: e_dt = e_dt + _dt.timedelta(days=1)
            S.append(Session(b_dt.timestamp()*1000.0, e_dt.timestamp()*1000.0))
        S.sort(key=lambda x: x.begin_ms)
        return SessionSet(S)
    def current(self, ms: float) -> Optional[int]:
        for i, s in enumerate(self.sessions):
            if s.contains(ms): return i
        return None
    def next_begin_after(self, ms: float) -> Optional[float]:
        for s in self.sessions:
            if ms < s.begin_ms: return s.begin_ms
        return None
    def last_end(self) -> float: return self.sessions[-1].end_ms

# ========== 价位聚合 & 失衡色带 ==========
class OrderFlowMaker:
    def __init__(self, one_tick: float, symbol: str, master_symbol: str):
        self.one_tick = one_tick; self.symbol = symbol; self.master_symbol = master_symbol
        self.ticks_map: Dict[float, Dict[str, float]] = {}; self.tick_price_list: List[float] = []
        self.add_volume: float = 0.0; self.datetime_list: List[str] = []; self.datetime_stamp_list: List[float] = []
        self.last_oi: float = 0.0 

    def store_tick(self, price: float, volume: float, tick_type: TickTypeEnum, dt_str: str, ms: float, oi: float):
        self.tick_price_list.append(price); self.datetime_list.append(dt_str); self.datetime_stamp_list.append(ms)
        self.last_oi = oi 
        m = self.ticks_map.get(price)
        if m is None:
            m = {"buy_qty_total": 0.0, "sells_qty_total": 0.0}
            self.ticks_map[price] = m
        if tick_type in (TickTypeEnum.OPENLONG, TickTypeEnum.CLOSESHORT, TickTypeEnum.EXCHANGELONG): m["buy_qty_total"] += volume
        elif tick_type in (TickTypeEnum.OPENSHORT, TickTypeEnum.CLOSELONG, TickTypeEnum.EXCHANGESHORT): m["sells_qty_total"] += volume
        else: self.add_volume += volume

    def _price_table_and_imbalance(self, IMB_MIN_VOL, IMB_RATIO_THR, ZERO_SIDE_MINVOL, IMB_RUN_LEN):
        rows = [[price, float(v['buy_qty_total']), float(v['sells_qty_total']), float(v['buy_qty_total']) + float(v['sells_qty_total'])] for price, v in sorted(self.ticks_map.items())]
        df = pd.DataFrame(rows, columns=["price","buy_qty_total","sells_qty_total","volume"])
        if df.empty: return df, 0.0, [], []
        poc = float(df.loc[df["volume"].idxmax(),"price"])
        df["buy_ok"]  = (df["buy_qty_total"] >= IMB_MIN_VOL) & (df["sells_qty_total"] >= IMB_MIN_VOL) & ((df["sells_qty_total"] > 0) & (df["buy_qty_total"] / df["sells_qty_total"] >= IMB_RATIO_THR))
        df["sell_ok"] = (df["sells_qty_total"] >= IMB_MIN_VOL) & (df["buy_qty_total"] >= IMB_MIN_VOL) & ((df["buy_qty_total"] > 0) & (df["sells_qty_total"] / df["buy_qty_total"] >= IMB_RATIO_THR))
        df["buy_zero_ok"]  = (df["sells_qty_total"] == 0) & (df["buy_qty_total"] >= ZERO_SIDE_MINVOL)
        df["sell_zero_ok"] = (df["buy_qty_total"] == 0) & (df["sells_qty_total"] >= ZERO_SIDE_MINVOL)
        df["buy_flag"]  = df["buy_ok"] | df["buy_zero_ok"]; df["sell_flag"] = df["sell_ok"] | df["sell_zero_ok"]

        grid = np.arange(df["price"].min(), df["price"].max() + self.one_tick*0.5, self.one_tick)
        gdf = pd.DataFrame({"price": grid}).merge(df[["price", "buy_flag", "sell_flag"]], on="price", how="left")
        for col in ("buy_flag", "sell_flag"): gdf[col] = gdf[col].astype("boolean").where(gdf[col].notna(), False)
        gdf["buy_run"]  = (gdf["buy_flag"].astype(int).rolling(IMB_RUN_LEN, min_periods=IMB_RUN_LEN).sum() == IMB_RUN_LEN)
        gdf["sell_run"] = (gdf["sell_flag"].astype(int).rolling(IMB_RUN_LEN, min_periods=IMB_RUN_LEN).sum() == IMB_RUN_LEN)
        
        def _collapse(levels: List[float]) -> List[Tuple[float, float]]:
            if not levels: return []
            lv = sorted(set(levels))
            out = []; s = lv[0]; e = lv[0]
            for x in lv[1:]:
                if abs(x - e - self.one_tick) < 1e-9: e = x
                else: out.append((s, e)); s = e = x
            out.append((s, e))
            return out
        return df.sort_values("price").reset_index(drop=True), poc, _collapse(list(set(gdf.loc[gdf["buy_run"], "price"].round(10).tolist()))), _collapse(list(set(gdf.loc[gdf["sell_run"], "price"].round(10).tolist())))

    def transform_and_summary(self) -> Optional[dict]:
        if not self.tick_price_list: return None
        df, poc, buy_ranges, sell_ranges = self._price_table_and_imbalance(0, 3.0, 1, 3)
        buy_qty_sum, sell_qty_sum = float(df["buy_qty_total"].sum()) if not df.empty else 0.0, float(df["sells_qty_total"].sum()) if not df.empty else 0.0
        
        begin_dt = self.datetime_list[0]
        d, t = begin_dt.split(" ")
        begin_dot = f"{d.replace('-', '.')} {t.split('.')[0][:5]}"

        row = {
            "datetime":      begin_dot,
            "open":          round(self.tick_price_list[0], 1),
            "high":          round(max(self.tick_price_list), 1),
            "low":           round(min(self.tick_price_list), 1),
            "close":         round(self.tick_price_list[-1], 1),
            "volume":        int(round(buy_qty_sum + sell_qty_sum + float(self.add_volume), 0)),
            "poc":           round(poc, 1),
            "delta":         int(round(buy_qty_sum - sell_qty_sum, 0)),
            "buy_imbalance": "", 
            "sell_imbalance": "", 
            "open_interest": int(self.last_oi) 
        }
        self.ticks_map.clear(); self.tick_price_list.clear(); self.add_volume = 0.0
        self.datetime_list.clear(); self.datetime_stamp_list.clear()
        return row

# ========== 主流程 ==========
def translate_tick_to_flow(file_path: str, symbol: str, master_symbol: str, delta_minutes: int, one_tick: float, output_dir: Optional[Path] = None, plot: bool = False, verbose: bool = False) -> Optional[Path]:
    df = pd.read_csv(file_path, header=0, sep=",", low_memory=False, dtype={"datetime": str})
    df = _normalize_tick_columns(df, master_symbol)

    key = _resolve_symbol_key(symbol, master_symbol)
    scheme = _build_scheme_from_symbol(key)
    if verbose:
        print(f"获取到交易时间: {scheme}")
    df = filter_dirty_ticks(df, symbol, master_symbol, scheme, strict_zero_block=True)

    ofm = OrderFlowMaker(one_tick=one_tick, symbol=symbol, master_symbol=master_symbol)
    begin_ms, end_ms, in_str, out_str = None, None, None, None
    cur_day, sess = None, None
    last_row = None
    sum_volume = buy_volume = sell_volume = 0
    last_price = 0.0
    summary_rows = []

    for i, row in enumerate(df.itertuples(index=False)):
        if i == 0:
            last_row = row
            continue

        dt_str = str(getattr(row, "datetime"))
        ms_now = float(getattr(row, "datetime_nano")) / 1_000_000.0

        date_part, time_part = dt_str.split(" ")
        day = _dt.datetime.strptime(date_part, "%Y-%m-%d").date()
        hour = int(time_part.split(":")[0])
        base_day = day - _dt.timedelta(days=1) if hour < 8 else day

        if end_ms is not None and ms_now >= end_ms:
            if sum_volume > 0 or len(ofm.tick_price_list) > 0:
                if verbose:
                    print(f"=开始时间：{in_str}, 结束时间：{out_str}, volume: {sum_volume}, delta: {buy_volume - sell_volume}, close: {int(last_price)}")
                one_row = ofm.transform_and_summary()
                if one_row: summary_rows.append(one_row)
            sum_volume = buy_volume = sell_volume = 0
            end_ms = None 

        if (sess is None) or (cur_day != base_day):
            cur_day = base_day
            sess = SessionSet.for_day(cur_day, scheme)

        if end_ms is None or sess.current(ms_now) is None:
            cur_idx = sess.current(ms_now)
            if cur_idx is None:
                nb = sess.next_begin_after(ms_now)
                if nb is None:
                    last_row = row
                    continue
                begin_ms = nb
            else:
                win_sec = delta_minutes * 60
                base = int(ms_now // (win_sec * 1000)) * (win_sec * 1000)
                begin_ms = float(base)
                if begin_ms < sess.sessions[cur_idx].begin_ms:
                    begin_ms = sess.sessions[cur_idx].begin_ms

            in_dt = _dt.datetime.fromtimestamp(begin_ms / 1000.0)
            end_ms = (in_dt + _dt.timedelta(minutes=delta_minutes)).timestamp() * 1000.0
            in_str = in_dt.strftime("%Y-%m-%d %H:%M:%S")
            out_str = (in_dt + _dt.timedelta(minutes=delta_minutes)).strftime("%Y-%m-%d %H:%M:%S")
            sum_volume = buy_volume = sell_volume = 0
            
            if ms_now < begin_ms:
                last_row = row
                continue

        vol_now = int(getattr(row, "volume")); vol_pre = int(getattr(last_row, "volume"))
        oi_now  = int(getattr(row, "open_interest")); oi_pre  = int(getattr(last_row, "open_interest"))
        
        vol_delta = vol_now - vol_pre
        if vol_delta < 0:
            last_ms = float(getattr(last_row, "datetime_nano")) / 1_000_000.0
            if (ms_now - last_ms) > 600000: vol_delta = vol_now 
            else: vol_delta = 0 
                
        oi_delta  = oi_now  - oi_pre  

        last_p = float(getattr(last_row, "last_price"))
        last_price = float(getattr(row, "last_price"))
        ask1_now = float(getattr(row, "ask_price1")); bid1_now = float(getattr(row, "bid_price1"))
        ask1_pre = float(getattr(last_row, "ask_price1")); bid1_pre = float(getattr(last_row, "bid_price1"))

        order_forward = TickAnalysis.get_order_forward(last_price, ask1_now, bid1_now, last_p, ask1_pre, bid1_pre)
        oi_forward    = TickAnalysis.get_open_interest_delta_forward(oi_delta, vol_delta)

        if begin_ms <= ms_now < end_ms:
            sum_volume += vol_delta
            mapping   = tick_type_cal_dict[oi_forward][order_forward]
            tick_type = mapping[TickTypeKeyEnum.TICKTYPE]
            desc      = mapping["DESC"]

            ofm.store_tick(price=last_price, volume=vol_delta, tick_type=tick_type, dt_str=dt_str, ms=ms_now, oi=oi_now)

            if desc in ("多开", "空平", "多换"): buy_volume += vol_delta
            if desc in ("空开", "多平", "空换"): sell_volume += vol_delta

        last_row = row

    one_row = ofm.transform_and_summary()
    if one_row: summary_rows.append(one_row)

    if summary_rows:
        out_df = pd.DataFrame(summary_rows, columns=OUTPUT_COLUMNS)
        out_csv_path = _write_orderflow_csv(out_df, Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR, master_symbol, delta_minutes)
        if verbose:
            print(f"✅ 订单流数据成功生成并保存至: {out_csv_path}")
        if plot:
            _plot_orderflow_chart(out_df, out_csv_path, master_symbol, delta_minutes)
        return out_csv_path

    if verbose:
        print("⚠ 过滤后无可用的聚合订单流数据，未执行写出操作。")
    return None


def _format_output_time(value: str) -> str:
    return str(value).replace('.', '').replace(' ', '_').replace(':', '')


def _write_orderflow_csv(out_df: pd.DataFrame, output_dir: Path, master_symbol: str, delta_minutes: int) -> Path:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(out_df)
    start_clean = _format_output_time(out_df.iloc[0]["datetime"])
    end_clean = _format_output_time(out_df.iloc[-1]["datetime"])
    csv_name = f"period_of_{delta_minutes}_{start_clean}_{end_clean}_{master_symbol}_{count}.csv"
    out_csv_path = output_dir / csv_name
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return out_csv_path


def _plot_orderflow_chart(out_df: pd.DataFrame, out_csv_path: Path, master_symbol: str, delta_minutes: int) -> None:
    try:
        import plotter
        out_png_dir = out_csv_path.parent / "images"
        out_png_dir.mkdir(parents=True, exist_ok=True)
        out_png_path = out_png_dir / f"orderflow_{master_symbol}_{delta_minutes}m.png"
        plotter.plot_orderflow_chart(df=out_df, out_png=str(out_png_path), recent_n=380, title=f"OrderFlow {master_symbol} {delta_minutes}m")
        print(f"✅ 自动绘图已保存至: {out_png_path}")
    except Exception as e:
        print(f"❌ 自动绘图失败: {e}")


def _contract_from_tick_path(path: Path) -> Optional[str]:
    match = re.match(r"^(SHFE\.cu\d+)_tick\.csv$", path.name)
    if match:
        return match.group(1)
    try:
        _, master, _ = extract_contract_from_filename(str(path))
        return master
    except Exception:
        return None


def _contract_from_reference_path(path: Path) -> Optional[str]:
    match = re.search(r"_(SHFE\.cu\d+)_\d+\.csv$", path.name)
    return match.group(1) if match else None


def _build_tick_index(input_dir: Path) -> Dict[str, Path]:
    input_dir = Path(input_dir).expanduser()
    result: Dict[str, Path] = {}
    for tick_path in sorted(input_dir.glob("SHFE.cu*_tick.csv")):
        contract = _contract_from_tick_path(tick_path)
        if contract:
            result[contract] = tick_path
    return result


def _build_reference_index(reference_dir: Path, window_min: int) -> Dict[str, Path]:
    reference_dir = Path(reference_dir).expanduser()
    result: Dict[str, Path] = {}
    for ref_path in sorted(reference_dir.glob(f"period_of_{window_min}_*_SHFE.cu*.csv")):
        contract = _contract_from_reference_path(ref_path)
        if contract and contract not in result:
            result[contract] = ref_path
    return result


def _empty_report_row(contract: str, input_csv: Optional[Path] = None, generated_csv: Optional[Path] = None, reference_csv: Optional[Path] = None, status: str = "") -> Dict[str, Any]:
    return {
        "contract": contract,
        "input_csv": str(input_csv) if input_csv else "",
        "generated_csv": str(generated_csv) if generated_csv else "",
        "reference_csv": str(reference_csv) if reference_csv else "",
        "status": status,
        "generated_rows": "",
        "reference_rows": "",
        "mismatch_count": "",
        "first_mismatch_column": "",
        "first_mismatch_row": "",
        "first_generated_value": "",
        "first_reference_value": "",
    }


def _read_orderflow_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False)
    df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
    return df


def _normalize_text_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _text_mismatch(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.map(_normalize_text_value) != right.map(_normalize_text_value)


def _float_mismatch(left: pd.Series, right: pd.Series) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_missing = left_num.isna() & right_num.isna()
    both_present = left_num.notna() & right_num.notna()
    close = pd.Series(False, index=left.index)
    close.loc[both_present] = np.isclose(left_num.loc[both_present], right_num.loc[both_present], atol=FLOAT_TOL, rtol=0.0)
    return ~(both_missing | close)


def _int_mismatch(left: pd.Series, right: pd.Series) -> pd.Series:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    both_missing = left_num.isna() & right_num.isna()
    both_present = left_num.notna() & right_num.notna()
    equal = pd.Series(False, index=left.index)
    equal.loc[both_present] = left_num.loc[both_present].round().astype("int64") == right_num.loc[both_present].round().astype("int64")
    return ~(both_missing | equal)


def validate_generated_against_reference(contract: str, input_csv: Optional[Path], generated_csv: Path, reference_csv: Path) -> Dict[str, Any]:
    row = _empty_report_row(contract, input_csv=input_csv, generated_csv=generated_csv, reference_csv=reference_csv)
    try:
        generated_df = _read_orderflow_csv(generated_csv)
        reference_df = _read_orderflow_csv(reference_csv)
    except Exception as e:
        row.update(status="generation_failed", first_mismatch_column="read_csv", first_generated_value=str(e))
        return row

    row["generated_rows"] = len(generated_df)
    row["reference_rows"] = len(reference_df)

    if list(generated_df.columns) != OUTPUT_COLUMNS or list(reference_df.columns) != OUTPUT_COLUMNS:
        row.update(
            status="column_mismatch",
            mismatch_count=1,
            first_mismatch_column="columns",
            first_generated_value="|".join(map(str, generated_df.columns)),
            first_reference_value="|".join(map(str, reference_df.columns)),
        )
        return row

    if len(generated_df) != len(reference_df):
        row.update(status="row_count_mismatch", mismatch_count=abs(len(generated_df) - len(reference_df)))
        return row

    mismatch_count = 0
    first_mismatch = None
    for column in OUTPUT_COLUMNS:
        if column in FLOAT_COLUMNS:
            mask = _float_mismatch(generated_df[column], reference_df[column])
        elif column in INT_COLUMNS:
            mask = _int_mismatch(generated_df[column], reference_df[column])
        else:
            mask = _text_mismatch(generated_df[column], reference_df[column])

        count = int(mask.sum())
        mismatch_count += count
        if count and first_mismatch is None:
            first_index = int(np.flatnonzero(mask.to_numpy())[0])
            first_mismatch = (
                column,
                first_index + 1,
                generated_df.iloc[first_index][column],
                reference_df.iloc[first_index][column],
            )

    if mismatch_count:
        column, row_no, generated_value, reference_value = first_mismatch
        row.update(
            status="value_mismatch",
            mismatch_count=mismatch_count,
            first_mismatch_column=column,
            first_mismatch_row=row_no,
            first_generated_value=generated_value,
            first_reference_value=reference_value,
        )
    else:
        row.update(status="passed", mismatch_count=0)
    return row


def _write_validation_report(report_rows: List[Dict[str, Any]], output_dir: Path) -> Path:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_report.csv"
    pd.DataFrame(report_rows, columns=VALIDATION_REPORT_COLUMNS).to_csv(report_path, index=False, encoding="utf-8-sig")
    return report_path


def _generate_one_contract(contract: str, input_csv: Optional[Path], reference_csv: Optional[Path], args: argparse.Namespace) -> Dict[str, Any]:
    if input_csv is None:
        return _empty_report_row(contract, reference_csv=reference_csv, status="missing_input")

    try:
        symbol, master, one_tick = extract_contract_from_filename(str(input_csv))
        generated_csv = translate_tick_to_flow(
            file_path=str(input_csv),
            symbol=symbol,
            master_symbol=master,
            delta_minutes=args.window_min,
            one_tick=one_tick,
            output_dir=Path(args.output_dir),
            plot=args.plot,
            verbose=args.verbose,
        )
    except Exception as e:
        row = _empty_report_row(contract, input_csv=input_csv, reference_csv=reference_csv, status="generation_failed")
        row["first_mismatch_column"] = "exception"
        row["first_generated_value"] = str(e)
        return row

    if generated_csv is None:
        return _empty_report_row(contract, input_csv=input_csv, reference_csv=reference_csv, status="generation_failed")

    if not args.validate:
        row = _empty_report_row(contract, input_csv=input_csv, generated_csv=generated_csv, reference_csv=reference_csv, status="not_validated")
        try:
            row["generated_rows"] = len(_read_orderflow_csv(generated_csv))
        except Exception:
            pass
        return row

    if reference_csv is None:
        row = _empty_report_row(contract, input_csv=input_csv, generated_csv=generated_csv, status="missing_reference")
        try:
            row["generated_rows"] = len(_read_orderflow_csv(generated_csv))
        except Exception:
            pass
        return row

    return validate_generated_against_reference(contract, input_csv, generated_csv, reference_csv)


def _print_summary(report_rows: List[Dict[str, Any]], report_path: Path) -> None:
    total = len(report_rows)
    passed = sum(1 for row in report_rows if row["status"] == "passed")
    missing_input = sum(1 for row in report_rows if row["status"] == "missing_input")
    missing_reference = sum(1 for row in report_rows if row["status"] == "missing_reference")
    failed = total - passed - missing_input - missing_reference
    print("=" * 60)
    print(f"validation_report: {report_path}")
    print(f"total={total}, passed={passed}, failed={failed}, missing_input={missing_input}, missing_reference={missing_reference}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick 数据转换为 5 分钟订单流数据，并可与 period_of_5 基准文件严格校验")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR), help="原始 Tick CSV 目录")
    parser.add_argument("--input_csv", type=str, default=None, help="只处理单个原始 Tick CSV 文件")
    parser.add_argument("--window_min", type=int, default=5, help="订单流聚合分钟窗口，默认 5")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="订单流 CSV 输出目录")
    parser.add_argument("--reference_dir", type=str, default=str(DEFAULT_REFERENCE_DIR), help="period_of_5 基准 CSV 目录")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True, help="生成后是否自动校验，默认开启")
    parser.add_argument("--plot", action="store_true", help="显式开启绘图；默认关闭，避免 plotter 缺失导致启动失败")
    parser.add_argument("--verbose", action="store_true", help="打印逐 Bar 调试日志")
    parser.add_argument("--workers", type=int, default=max(1, min(10, os.cpu_count() or 1)), help="全目录批处理的并行进程数，默认最多 10 个")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    reference_index = _build_reference_index(Path(args.reference_dir), args.window_min) if args.validate else {}

    if args.input_csv:
        input_csv = Path(args.input_csv).expanduser()
        contract = _contract_from_tick_path(input_csv)
        if contract is None:
            _, contract, _ = extract_contract_from_filename(str(input_csv))
        input_index = {contract: input_csv}
        contracts = sorted(input_index)
    else:
        input_index = _build_tick_index(Path(args.input_dir))
        contracts = sorted(set(input_index) | set(reference_index))

    print(f"input_contracts={len(input_index)}, reference_contracts={len(reference_index)}, output_dir={output_dir}, workers={args.workers}")
    report_by_contract: Dict[str, Dict[str, Any]] = {}
    runnable_contracts: List[str] = []
    for contract in contracts:
        input_csv = input_index.get(contract)
        if input_csv is None:
            report_by_contract[contract] = _empty_report_row(contract, reference_csv=reference_index.get(contract), status="missing_input")
            print(f"[missing_input] {contract}")
        else:
            runnable_contracts.append(contract)

    worker_count = max(1, min(int(args.workers), len(runnable_contracts) or 1))
    if worker_count > 1:
        print(f"parallel_workers={worker_count}, runnable_contracts={len(runnable_contracts)}")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_generate_one_contract, contract, input_index.get(contract), reference_index.get(contract), args): contract
                for contract in runnable_contracts
            }
            for done_index, future in enumerate(as_completed(futures), start=1):
                contract = futures[future]
                try:
                    row = future.result()
                except Exception as e:
                    row = _empty_report_row(contract, input_csv=input_index.get(contract), reference_csv=reference_index.get(contract), status="generation_failed")
                    row["first_mismatch_column"] = "worker_exception"
                    row["first_generated_value"] = str(e)
                report_by_contract[contract] = row
                print(f"[{done_index}/{len(runnable_contracts)}] {contract} {row['status']}")
    else:
        for index, contract in enumerate(runnable_contracts, start=1):
            input_csv = input_index.get(contract)
            reference_csv = reference_index.get(contract)
            print(f"[{index}/{len(runnable_contracts)}] {contract}")
            report_by_contract[contract] = _generate_one_contract(contract, input_csv, reference_csv, args)

    report_rows = [report_by_contract[contract] for contract in contracts]
    report_path = _write_validation_report(report_rows, output_dir)
    _print_summary(report_rows, report_path)
    return 0 if all(row["status"] in {"passed", "missing_input"} for row in report_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
