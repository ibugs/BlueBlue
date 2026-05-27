#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/21 10:03 下午
# @Author  : 王浩
# @File    : indicator.py
# @Desc    : 类描述

import numpy as np
import pandas as pd
from pandas import DataFrame
import talib


class Indicator(object):

    def __init__(self):
        pass

    @staticmethod
    def KAMA_INTERVAL(input_df: DataFrame, n=3, fast=10, slow=50):
        df = Indicator.KAMA(input_df, n=n, fast=fast, slow=slow)
        for i in range(df["ama"].size):
            if i < 20:
                df.loc[i, "upper"] = np.nan
                df.loc[i, "lower"] = np.nan
            else:
                df.loc[i, "upper"] = np.array(df.loc[(i - 20):i]['close']).max()
                df.loc[i, "lower"] = np.array(df.loc[(i - 20):i]['close']).min()

        return df

    @staticmethod
    def KAMA_ATR(input_df: DataFrame, n=3, fast=10, slow=50):
        """
        使用AMA+ATR 的方式能够更加贴合区间的情况
        比用最近20根或者其他上下区间的方式更加接近一些
        :param input_df: 传入的DataFrame
        :param n: ER周期
        :param fast: 快均线
        :param slow: 慢均线
        :return:
        """
        df = Indicator.KAMA(input_df, n=n, fast=fast, slow=slow)
        # 在生产ATR的过程中，是需要多一点数据才能够计算出来并且不出现Nan的情况
        prepare_number = 2 * slow
        for i in range(df["sc"].size):
            if i < prepare_number:
                df.loc[i, "upper"] = np.nan
                df.loc[i, "lower"] = np.nan
            else:
                atr_list = talib.ATR(high=np.array(df["high"].iloc[i - prepare_number:i]),
                                     close=np.array(df["close"].iloc[i - prepare_number:i]),
                                     low=np.array(df["low"].iloc[i - prepare_number:i]), timeperiod=30)
                df.loc[i, "atr"] = round(atr_list[-1], 2)
                ama_now = df.loc[i, "ama"]
                atr_now = df.loc[i, "atr"]
                if np.isnan(ama_now) or np.isnan(atr_now):
                    df.loc[i, "upper"] = np.nan
                    df.loc[i, "lower"] = np.nan
                else:
                    df.loc[i, "upper"] = round(df.loc[i, "ama"] + df.loc[i, "atr"] * 2, 2)
                    df.loc[i, "lower"] = round(df.loc[i, "ama"] - df.loc[i, "atr"] * 2, 2)
        return df

    @staticmethod
    def KAMA(df: DataFrame, n=3, fast=10, slow=50):
        """
        最正宗的KAMA计算方法，初始化设为close，传入df，返回有KAMA的df
        :param df: df
        :param n: ER周期
        :param fast: 快周期
        :param slow: 慢周期
        :return:
        """
        # 计算效率系数 ER
        df["diff"] = abs(df["close"] - df["close"].shift(1))
        # 噪声，此值不可为0
        df["noise"] = df["diff"].rolling(n).sum()
        # 方向
        df["direction"] = abs(df["close"] - df["close"].shift(n))
        # Efficient_Ratio 效率系数。越大，代表趋势越流畅
        df["ER"] = df["direction"] / df["noise"]

        # 根据ER计算平滑系数sc
        df["smooth"] = df["ER"] * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)
        df["sc"] = df["smooth"] * df["smooth"]
        # 因为direction 和 noise，有可能均为0，导致smooth和sc出现Nan的情况
        # 所以要使用fillna的方式来将里面的Nan给替换掉
        df.fillna(0, inplace=True)
        first_value = True
        for i in range(df["sc"].size):
            # 处理前排的空值
            if df.loc[i, "sc"] != df.loc[i, "sc"]:
                df.loc[i, "ama"] = np.nan
            else:
                if first_value:
                    df.loc[i, "ama"] = round(df.loc[i, "close"], 2)
                    first_value = False
                else:
                    df.loc[i, "ama"] = round(df.loc[i - 1, "ama"] + df.loc[i, "sc"] * (
                            df.loc[i, "close"] - df.loc[i - 1, "ama"]), 2)
        return df

    @staticmethod
    def ATR(df, timeperiod=30):
        """
        单独计算ATR的值
        :param df:
        :param timeperiod:
        :return:
        """
        atr_list = talib.ATR(high=np.array(df['high']), close=np.array(df['close']),
                             low=np.array(df['low']), timeperiod=timeperiod)
        return atr_list

    @staticmethod
    def EMA(df, timeperiod=30):
        if len(df['close']) < timeperiod:
            return None
        else:
            ema_list = talib.EMA(np.array(df['close']), timeperiod=timeperiod)
            return ema_list[-1]

    @staticmethod
    def PEARSON(results, num):
        """
        测试皮尔森的值
        :param results:
        :param num:
        :return:
        """
        yy = []
        xx = []
        last_num_list = results[-num:]
        for index, item in enumerate(last_num_list):
            yy.append(item)
            xx.append(index)
        # 50 根的pearson
        X1 = pd.Series(xx)
        Y1 = pd.Series(yy)
        pearson_number = X1.corr(Y1, method="pearson")
        # print(len(xx), "pearson值为：", round(pearson_number, 4))
        return round(pearson_number, 4)

    @staticmethod
    def KC(df, timeperiod=60, nAtr=2):
        """
        计算KC通道，EMA + ATR
        :param close_list:
        :param high_list:
        :param low_list:
        :param timeperiod:
        :return:
        """
        ema_list = talib.EMA(np.array(df['close']), timeperiod=timeperiod)
        atr_list = Indicator.ATR(df, timeperiod=timeperiod)
        if np.isnan(ema_list[-1]) or np.isnan(atr_list[-1]):
            pass
        else:
            ema = round(ema_list[-1], 2)
            atr = round(atr_list[-1], 2)
            upper = round(ema + nAtr * atr, 2)
            lower = round(ema - nAtr * atr, 2)
            return upper, ema, lower

    @staticmethod
    def TANGANQI(df, timeperiod=30):
        """
        很早之前的海龟通道，唐安琪通道
        使用df 来描述当前震荡的情况
        并且以df返回去
        :param df:
        :param timeperiod:
        :param nAtr:
        :return:
        """
        prepare_number = timeperiod
        for i in range(df["close"].size):
            if i < prepare_number:
                df.loc[i, "upper"] = np.nan
                df.loc[i, "lower"] = np.nan
            else:
                df.loc[i, "upper"] = np.array(df["close"].iloc[i - prepare_number:i]).max()
                df.loc[i, "lower"] = np.array(df["close"].iloc[i - prepare_number:i]).min()
        return df
