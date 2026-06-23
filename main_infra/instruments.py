import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class InstrumentsSample:
    
    def __init__(self, symbol, asset_type):
        self.symbol = symbol
        self.asset_type = asset_type
        self.data = None
        self.indicator_data = None
        self.interval = None
        self.period = None
        self.validation_report = None
        self.clean_report = None

    
    def fetch_data(self, periodA = '1mo', intervalA = '1h'):
        self.interval = intervalA
        self.period = periodA
        if os.path.exists(f"{self.symbol}_{self.period}_{self.interval}.csv"):
            print("Loading from cache ...")
            self.data = pd.read_csv(f"{self.symbol}_{self.interval}.csv", index_col=0, parse_dates=True)
            self.clean_data()
        else:
            self.data = yf.download(self.symbol, period = self.period, interval = self.interval, multi_level_index=False) #Extracts data
            self.clean_data()
            print("Saving to cache ...")
            self.data.to_csv(f"{self.symbol}_{self.interval}.csv", index=True)

        return

    def clean_data(self):
        initial_rows = len(self.data)
        self.data = self.data[~self.data.index.duplicated(keep='first')] #Data Cleaning 1: Removes Duplicates
        self.data = self.data.ffill() #Data cleaning 2: replaces NANs with previous data
        self.data = self.data.dropna() #Data Cleaning 3: Removes leftover Nan values
        self.data.sort_index(inplace = True)
        final_rows = len(self.data)

        clean_list = {"nan_values": self.data.isna().sum(), "data_increasing": self.data.index.is_monotonic_increasing, 
                      "initial_rows": initial_rows, "final_rows": final_rows}
        self.clean_report = clean_list

        return

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

    def simple_moving_avg(self, N = 20): 
        if self.data is None:
            print("Call self.fetch_data(period, interval) first")
            return
        return self.data["Close"].rolling(window = N).mean()
    
    def exp_moving_avg(self, N = 20):
        if self.data is None:
            print("Call self.fetch_data(period, interval) first")
            return
        return self.data["Close"].ewm(span = N, adjust = False).mean()
    
    def MACD(self): 
        if self.data is None:
            print("Call self.fetch_data(period, interval) first")
            return
        EMA_12 = self.exp_moving_avg(12)
        EMA_26 = self.exp_moving_avg(26)
        MACD_line = EMA_12 - EMA_26
        signal = MACD_line.ewm(span = 9, adjust = False).mean()
        histogram = MACD_line - signal
        return MACD_line, signal, histogram #Maybe change later on how to return
    
    def rolling_std_dev(self, N):
        if self.data is None:
            print("Call self.fetch_data(period, interval) first")
            return
        return ((self.simple_return()).rolling(window = N)).std()
    
    def bollinger_bands(self, N, num_std=2):
        if self.data is None:
            print("Call self.fetch_data(period, interval) first")
            return
        SMA_val = self.simple_moving_avg(N)
        std_val = self.data["Close"].rolling(window = N).std()
        upper_val = SMA_val + (num_std * std_val)
        lower_val = SMA_val - (num_std * std_val)
        middle_val = SMA_val
        return upper_val, lower_val, middle_val #Maybe change later on how to return
    
    def RSI_indicator(self, N = 14):
        if self.data is None:
            print("Call self.fetch_data(period, interval) first")
            return
        close_Vals = self.data["Close"].pct_change()
        gain = close_Vals.clip(lower = 0)
        loss = -close_Vals.clip(upper = 0)
        avg_gain = gain.rolling(window=N).mean()
        avg_loss = loss.rolling(window=N).mean()
        rs = avg_gain/avg_loss
        return (100 - (100/(1 + rs)))
    
    def rolling_max_min(self, N):
        if self.data is None:
            print("Call self.fetch_data(period, interval) first")
            return
        return self.data["Close"].rolling(window = N).max(), self.data["Close"].rolling(window = N).min()
    
    def ATR(self, N = 14):
        if self.data is None:
            print("Call self.fetch_data(period, interval) first")
            return
        high_low = self.data["High"] - self.data["Low"]
        high_close = abs(self.data["High"]- self.data["Close"].shift(1))
        low_close = abs(self.data["Low"]- self.data["Close"].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
        atr = true_range.rolling(window = N).mean()
        return atr

    def lag_val(self, N): 
        if self.data is None:
            print("Call self.fetch_data(period, interval) first")
            return
        return self.data['Close'].shift(N)
    
    def calculate_all_indicators(self): #Can be optimized later
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
        if (not self.validation_report["passed"]):
            return "Something went wrong."
        return self.indicator_data
    
    def verify_indicators(self):
        errors = []
        warnings = []
        
        #Overall Data checks
        if (not self.indicator_data.index.is_monotonic_increasing):
            errors.append("Dates not Increasing")
        if (self.indicator_data.index.duplicated().sum()>0):
            errors.append("Date Duplicates Error")
        numeric_df = self.indicator_data.select_dtypes(include= np.number)
        has_inf = np.isinf(numeric_df).any().any()
        if (has_inf):
            errors.append("Infinite Data Error")
        
        #Checking RSI
        invalid_rsi = (self.indicator_data["RSI_14"] < 0) | (self.indicator_data["RSI_14"] > 100 )
        if (invalid_rsi.sum()>0):
            warnings.append("RSI Invalid Error")
        #Checking Rolling STD
        negative_std = self.indicator_data["Rolling_STD_20"] < 0
        if (negative_std.sum() > 0):
            errors.append("Negative STD Error")
        #Checking Simple Returns
        returns = abs(self.indicator_data["Simple_Return"])>0.5
        if (returns.sum() > 0):
            warnings.append(f"Simple Returns {returns.sum()} Large Returns error")        
        #Checking Bollinder Bands
        clean_bands = self.indicator_data[['Bollinger_Upper_20', 'Bollinger_Middle_20', 'Bollinger_Lower_20']].dropna()
        if not clean_bands.empty:
            boll_error = (clean_bands["Bollinger_Upper_20"] < clean_bands["Bollinger_Middle_20"]) | \
                 (clean_bands["Bollinger_Middle_20"] < clean_bands["Bollinger_Lower_20"])
            if boll_error.any():
                errors.append(f"Bollinger Band Cross Error: {boll_error.sum()} rows invalid.")

        #Checking SMA:
        smaCheck1 = self.indicator_data["SMA_20"] > self.indicator_data["Rolling_Max_20"]
        smaCheck2 = self.indicator_data["SMA_20"] < self.indicator_data["Rolling_Min_20"]
        if ((smaCheck1 | smaCheck2).any()):
            errors.append("SMA error")

        if len(errors)!=0:
            return {
                "passed": False,
                "errors": errors,
                "warnings": warnings
                }
        return {
                "passed": True,
                "errors": errors,
                "warnings": warnings
                }
    
    def save_indicators_csv(self):
        if self.indicator_data is None:
            print("No indicator data to save")
            return
        filename = f"{self.symbol}_{self.period}_{self.interval}.csv"
        self.indicator_data.to_csv(filename, index=True)
        print(f"Indicator data saved to {filename}")
    
    def load_indicators_csv(self):
        filename = f"{self.symbol}_{self.period}_{self.interval}.csv"
        if not os.path.exists(filename):
            print("Indicator cache file does not exist")
            return
        self.indicator_data = pd.read_csv(
            filename,
            index_col=0,
            parse_dates=True
        )
        print(f"Loaded indicator data from {filename}")
        return self.indicator_data




class Stock(InstrumentsSample):
    def __init__(self, symbol, asset_type = "stock"):
        super().__init__(symbol, asset_type)
