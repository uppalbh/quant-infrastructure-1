import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class MeanReversion(StrategyBase):
    def __init__(self, instrument, bb_period = 20, bb_std = 1.2):
        super().__init__(instrument)
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def generate_signals(self):
        close = self.instrument.indicator_data["Close"]
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        z_score = (close - sma) / std
        signal = pd.Series(0, index = close.index)
        signal[z_score<-1] = 1
        signal[z_score>1] = -1
        signal = signal.shift(1).fillna(0)
        return pd.DataFrame(
            {
                "Close": close,
                "SMA": sma,
                "STD": std,
                "Z_Score": z_score,
                "Signal": signal
            }
        )
