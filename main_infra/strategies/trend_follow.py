import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class TrendFollow(StrategyBase):
    """
    Long when EMA_20 > EMA_50 AND close > EMA_20 (bull trend).
    Short when EMA_20 < EMA_50 AND close < EMA_20 (bear trend).
    Designed for regime-exit holding — stays in until regime changes.
    """
    def __init__(self, instrument, fast=20, slow=50):
        super().__init__(instrument)
        self.fast = fast
        self.slow = slow

    def generate_signals(self):
        close = self.instrument.indicator_data["Close"]
        ema_fast = self.instrument.exp_moving_avg(self.fast)
        ema_slow = self.instrument.exp_moving_avg(self.slow)
        signal = pd.Series(0, index=close.index)
        # Bull trend: EMA20 > EMA50 AND price above EMA20
        signal[(ema_fast > ema_slow) & (close > ema_fast)] = 1
        # Bear trend: EMA20 < EMA50 AND price below EMA20
        signal[(ema_fast < ema_slow) & (close < ema_fast)] = -1
        return pd.DataFrame({
            "Close": close,
            "Signal": signal
        })
