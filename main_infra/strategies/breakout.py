import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Breakout(StrategyBase):
    def __init__(self, instrument, lookback_period = 50):
        self.instrument = instrument
        self.lookback_period = lookback_period
    def generate_signals(self):
        rolling_high, rolling_low = self.instrument.rolling_max_min(self.lookback_period)
        close = self.instrument.indicator_data["Close"]
        signal = pd.Series(0, index = close.index)
        signal[close > rolling_high.shift(1)] = 1
        signal[close < rolling_low.shift(1)] = -1
        return pd.DataFrame(
            {
                "Close": close,
                "Rolling_High": rolling_high,
                "Rolling_Low": rolling_low,
                "Signal": signal
            }
        )
