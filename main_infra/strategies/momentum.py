import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Momentum(StrategyBase):
    def __init__(self, instrument, rsi_period = 14, buy_level = 70, sell_level = 30):
        super().__init__(instrument)
        self.rsi_period = rsi_period
        self.buy_level = buy_level
        self.sell_level = sell_level

    def generate_signals(self):
        rsi = self.instrument.RSI_indicator(self.rsi_period)
        signal = pd.Series(0, index = rsi.index)
        signal[rsi > self.buy_level] = -1
        signal[rsi < self.sell_level] = 1
        return pd.DataFrame(
            {
                "Close": self.instrument.indicator_data["Close"],
                "RSI": rsi,
                "Signal": signal
            }
        )
