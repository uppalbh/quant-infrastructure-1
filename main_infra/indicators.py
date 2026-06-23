import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

def simple_return(self):
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    return self.data["Close"].pct_change()

def log_return(self):
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    return np.log(self.data["Close"]/self.data["Close"].shift(1))

def cumulative_returns(self):
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    return (1 + self.simple_return()).cumprod() - 1

def simple_moving_avg(self, N=20): 
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    return self.data["Close"].rolling(window=N).mean()

def exp_moving_avg(self, N=20):
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    return self.data["Close"].ewm(span=N, adjust=False).mean()

def MACD(self): 
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    EMA_12 = self.exp_moving_avg(12)
    EMA_26 = self.exp_moving_avg(26)
    MACD_line = EMA_12 - EMA_26
    signal = MACD_line.ewm(span=9, adjust=False).mean()
    histogram = MACD_line - signal
    return MACD_line, signal, histogram

def rolling_std_dev(self, N):
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    return ((self.simple_return()).rolling(window=N)).std()

def bollinger_bands(self, N, num_std=2):
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    SMA_val = self.simple_moving_avg(N)
    std_val = self.data["Close"].rolling(window=N).std()
    upper_val = SMA_val + (num_std * std_val)
    lower_val = SMA_val - (num_std * std_val)
    middle_val = SMA_val
    return upper_val, lower_val, middle_val

def RSI_indicator(self, N=14):
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    close_Vals = self.data["Close"].pct_change()
    gain = close_Vals.clip(lower=0)
    loss = -close_Vals.clip(upper=0)
    avg_gain = gain.rolling(window=N).mean()
    avg_loss = loss.rolling(window=N).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs)))

def rolling_max_min(self, N):
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    return self.data["Close"].rolling(window=N).max(), self.data["Close"].rolling(window=N).min()

def ATR(self, N=14):
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    high_low = self.data["High"] - self.data["Low"]
    high_close = abs(self.data["High"] - self.data["Close"].shift(1))
    low_close = abs(self.data["Low"] - self.data["Close"].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=N).mean()
    return atr

def lag_val(self, N): 
    if self.data is None:
        print("Call self.fetch_data(period, interval) first")
        return
    return self.data['Close'].shift(N)

def calculate_all_indicators(self):
    if os.path.exists(f"{self.symbol}_{self.interval}_indicator.csv"):
        print("Loading from cache ...")
        self.indicator_data = pd.read_csv(f"{self.symbol}_{self.interval}_indicator.csv", index_col=0, parse_dates=True)
        return self.indicator_data
    
    simp_ret = self.simple_return()
    log_ret = self.log_return()
    cumu_ret = self.cumulative_returns()
    SMA_20 = self.simple_moving_avg(20)
    EMA_20 = self.exp_moving_avg(20)
    macd_line, signal_line, hist = self.MACD()
    rolling_std = self.rolling_std_dev(20)
    upperBand, lowerBand, middleBand = self.bollinger_bands(20)
    RSI_14 = self.RSI_indicator(14)
    rolling_max_20, rolling_min_20 = self.rolling_max_min(20)
    atr_14 = self.ATR()

    intVals = {
        "Simple_Return": simp_ret,
        "Log_Return": log_ret,
        "Cumulative_Return": cumu_ret,
        "SMA_20": SMA_20,
        "EMA_20": EMA_20,
        "MACD_Line": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Histogram": hist,
        "Rolling_STD_20": rolling_std,
        "Bollinger_Upper_20": upperBand,
        "Bollinger_Middle_20": middleBand,
        "Bollinger_Lower_20": lowerBand,
        "RSI_14": RSI_14,
        "ATR_14": atr_14,
        "Rolling_Max_20": rolling_max_20,
        "Rolling_Min_20": rolling_min_20
    }
    df = pd.DataFrame(intVals)
    self.indicator_data = pd.concat([self.data, df], axis=1)
    self.validation_report = self.verify_indicators()
    if not self.validation_report["passed"]:
        return "Something went wrong."
    return self.indicator_data
