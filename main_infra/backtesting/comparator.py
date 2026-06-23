import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class StrategyComparator:
    def __init__(self, instrument, starting_capital = 10000):
        self.instrument = instrument
        self.starting_capital = starting_capital
        self.results = None
    
    def compare(self, strategies_dict): 
        '''Strategies_dict = {"Strategy Name": Strategy(instrument, paramters),...}'''
        returns_list = []
        for name, StrategyClass in strategies_dict.items():
            bt = SimpleBacktester(self.instrument, StrategyClass, self.starting_capital)
            bt.run()
            returns_list.append({
                "Strategy Name": name,
                "Return": bt.results["Total_Return"],
                "Sharpe": bt.results["Sharpe_Ratio"],
                "Max_DD": bt.results["Max_DrawDown"],
                "Win_Rate": bt.results["Win_Rate"],
                "Trades": bt.results["Number_Of_Trades"],
            })
        self.results = pd.DataFrame(returns_list).sort_values("Sharpe", ascending= False)
        return self.results
