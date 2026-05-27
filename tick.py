#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/29 11:23 上午
# @Author  : 王浩
# @File    : tick.py
# @Desc    : 类描述
import time


class Tick(object):

    def __init__(self, symbol, master_symbol,
                 datetime_nano, last_price, highest, lowest, volume, amount,
                 open_interest, bid_price1, bid_volume1, ask_price1, ask_volume1):
        # 有些值可以float，volume这些值只能够
        self.symbol = str(symbol)
        self.master_symbol = str(master_symbol)
        self.datetime_nano = str(datetime_nano)
        self.last_price = float(last_price)
        self.highest = float(highest)
        self.lowest = float(lowest)
        self.volume = float(volume)
        self.amount = float(amount)
        self.open_interest = float(open_interest)
        self.bid_price1 = float(bid_price1)
        self.bid_volume1 = float(bid_volume1)
        self.ask_price1 = float(ask_price1)
        self.ask_volume1 = float(ask_volume1)
        if datetime_nano is not None:
            str_nano = str(datetime_nano)
            now_timestamp = float(str_nano[0:10])
            now_timestamp_left = str_nano[10:20]
            # print("now_timestamp: %s, now_timestamp_left:%s" % (now_timestamp, now_timestamp_left))
            time_local = time.localtime(now_timestamp)
            dt_time = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
            dt_day = time.strftime("%Y-%m-%d", time_local)
            self.date_of_day = dt_day
            self.datetime = dt_time + "." + now_timestamp_left


# class TickStore(object):
# 
#     @staticmethod
#     def insert(db_util: MySQLDBUtil, tick: Tick):
#         insert_sql = "insert into future_tick_data (symbol, master_symbol," \
#                      " date_of_day, datetime, datetime_nano," \
#                      " last_price, highest, lowest," \
#                      " volume, amount, open_interest," \
#                      " bid_price1, bid_volume1, ask_price1, ask_volume1)" \
#                      " values (%s,%s," \
#                      "%s,%s,%s," \
#                      "%s,%s,%s," \
#                      "%s,%s,%s," \
#                      "%s,%s,%s,%s)"
#         values = (tick.symbol, tick.master_symbol,
#                   tick.date_of_day, tick.datetime, tick.datetime_nano,
#                   tick.last_price, tick.highest, tick.lowest,
#                   tick.volume, tick.amount, tick.open_interest,
#                   tick.bid_price1, tick.bid_volume1, tick.ask_price1, tick.ask_volume1
#                   )
#         print(values)
#         print(tick.datetime, end="\t")
#         db_util.insert_table(executor_many_sql=insert_sql, values=values)
