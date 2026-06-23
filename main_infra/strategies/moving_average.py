import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class MovingAverageCrossover(StrategyBase):
    def __init__(self, instrument, fast_period = 10, slow_period = 50):
        super().__init__(instrument)
        self.fastP = fast_period
        self.slowP = slow_period
    
    def generate_signals(self):
        #Fast and Slow SMA
        fast_ma = self.instrument.simple_moving_avg(self.fastP)
        slow_ma = self.instrument.simple_moving_avg(self.slowP)
        signal = pd.Series([0]*len(fast_ma))
        signal.index = fast_ma.index
        signal[fast_ma>slow_ma] = 1
        signal[fast_ma<slow_ma] = -1
        return pd.DataFrame({
            "Close": self.instrument.indicator_data["Close"],
            "Fast_MA": fast_ma,
            "Slow_MA": slow_ma,
            "Signal": signal,}
        )
