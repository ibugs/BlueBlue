#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2026/05/09 21:25
# @Author  : 王仁东
# @File    : order_flow.py
# @Desc    : 类描述
from enum import Enum

class Direction(Enum):
    UP = 1    # 多
    DOWN = -1 # 空
    NIL = 0 # 无方向


class FutureOrderFlowData(object):

    def __init__(self, id, symbol, master_symbol, date_of_day, open_price, close_price,
                 high_price, low_price, begin_time, end_time, begin_time_stamp,
                 end_time_stamp, delta, volume, main_buy_imbalance_area,
                 main_sell_imbalance_area, increase_tick, delta_percentage,
                 poc_price, poc_pct, upper_shadow_pct,
                 down_shadow_pct, prefer):
        self.id = id                                              # 主键ID
        self.master_symbol = master_symbol                        # 主连标识
        self.symbol = symbol                                      # '币种',
        self.date_of_day = date_of_day                            # '日期 YYYY-MM-DD',
        self.open_price = float(open_price)                       # '开盘价',
        self.close_price = float(close_price)                     # '收盘价',
        self.high_price = float(high_price)                       # '最高价',
        self.low_price = float(low_price)                         # '最低价',
        self.begin_time = begin_time                              # '开盘时间',
        self.end_time = end_time                                  # '收盘时间',
        self.begin_time_stamp = str(begin_time_stamp)             # '开盘时间戳',
        self.end_time_stamp = str(end_time_stamp)                 # '收盘时间戳',
        self.delta = float(delta)                                 # 'Delta=主买-主卖',
        self.volume = int(volume)                                 # '该K线内的主买和主卖的成交量总和',
        self.main_buy_imbalance_area = main_buy_imbalance_area    # '主买失衡区域，3倍，逗号分隔',
        self.main_sell_imbalance_area = main_sell_imbalance_area  # '主卖失衡区域，3倍，逗号分隔',
        self.increase_tick = float(increase_tick)                 # '涨幅点数',
        self.delta_percentage = float(delta_percentage)           # 'delta占volume的比值，百分比描述，带符号',
        self.poc_price = float(poc_price)                         # '该Bar内的成交量最大的价格位置',
        self.poc_pct = float(poc_pct)                             # '该POC在bar内的占比百分比一定为正数',
        self.upper_shadow_pct = float(upper_shadow_pct)           # '上影线占实体的百分比',
        self.down_shadow_pct = float(down_shadow_pct)             # '下影线占实体的百分比',
        self.prefer = prefer                                      # tinyint '趋势方向多空偏好 0 震荡, -1 偏空, 1 偏多',
        if self.increase_tick > 0:
            self.direction = 1
        if self.increase_tick < 0:
            self.direction = -1

        # 入场bar的操作方向， 0 表示无方向，1表示做多，-1表示做空
        # Direction.UP or Direction.DOWN
        self.op_direction = 0

    def to_string(self):
        print("end:%s, close:%s, increase:%s, volume:%s, delta:%s, upper_pct:%s, down_pct:%s" % (
            self.end_time, self.close_price, self.increase_tick, self.volume, self.delta,
            self.upper_shadow_pct, self.down_shadow_pct))
