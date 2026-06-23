import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class SimpleBacktester:
    def __init__(self, instrument, strategy, startingCap, commission_pct=0.001, slippage_pct=0.0002):
        self.instrument = instrument
        self.strategy = strategy #Strategy is like strategy = Momentum(stock, 14, 70, 30)
        self.starting_capital = startingCap
        self.backtest_data = None
        self.results = None
        self.trade_data = None
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(self):
        trade_list = []

        signals_series = self.strategy.generate_signals()["Signal"]
        signals_series = signals_series.shift(1).fillna(0) #Prevents Look Ahead Bias and Fills first point with 0
        signals_series = signals_series.astype(int)

        self.backtest_data = pd.DataFrame(
            {
                "Close": self.instrument.indicator_data["Close"],
                "Signal": signals_series,
            }
        )
        self.backtest_data["Position"] = 0
        self.backtest_data["Cash"] = 0.0
        self.backtest_data["Shares_Held"] = 0.0
        self.backtest_data["Portfolio_Value"] = 0.0
        self.backtest_data["Action"] = ""
        self.backtest_data["Unrealized_PL"] = 0.0
        self.backtest_data["Realized_PL"] = 0.0
        self.backtest_data["Entry_Price"] = np.nan

        cash = self.starting_capital
        shares_held = 0
        current_position = 0

        entry_price = None
        entry_date = None
        realized_pl = 0.0

        for idx, row in self.backtest_data.iterrows():
            signal = row["Signal"]
            price = row["Close"]
            effective_price = price * (1 + self.slippage_pct)
            action = "Hold"
            if signal == 1 and current_position == 0:
                if math.isnan(cash) or math.isnan(effective_price):
                    shares_held = 0
                else:
                    shares_held = int(cash/effective_price)
                    commission = 0
                    if shares_held > 0:
                        action = "Buy"
                        entry_date = idx
                        entry_price = price
                        commission = (shares_held * effective_price) * self.commission_pct
                    cash = cash - (shares_held*effective_price) - commission
                
            elif signal == -1 and current_position == 1:
                sale_proceeds = shares_held * effective_price
                commission = sale_proceeds * self.commission_pct
                cash = cash + sale_proceeds - commission
                trade_profit = (price - entry_price) * shares_held
                realized_pl += trade_profit
                shares_held = 0
                action = "Sell"
                trade_return = (price-entry_price)/entry_price
                trade_info = {
                    "Entry_Date": entry_date,
                    "Entry_Price": entry_price,
                    "Exit_Date": idx,
                    "Exit_Price": price,
                    "Trade_Return": trade_return
                }
                trade_list.append(trade_info)
                entry_price = None
                entry_date = None
            
            current_position = int(shares_held > 0)

            if current_position==1 and entry_price is not None: 
                unrealized_pl = (price - entry_price) * shares_held
            else:
                unrealized_pl = 0.0  
            portfolio_value = cash + (shares_held*price)
            indList = ["Position", "Cash", "Shares_Held", "Portfolio_Value", "Action", "Unrealized_PL", "Realized_PL", "Entry_Price"]
            valList = [current_position, cash, shares_held, portfolio_value, action, unrealized_pl, realized_pl, entry_price]
            for i in range(len(indList)):
                self.backtest_data.loc[idx, indList[i]] = valList[i]
        self.trade_data = pd.DataFrame(trade_list)

        totalR = self.total_return()
        sharpeR = self.sharpe_ratio()
        maxDD = self.max_drawdown()
        noTrades = self.number_of_trades()
        winR = self.win_rate()
        self.results = {
            "Total_Return": totalR,
            "Sharpe_Ratio": sharpeR,
            "Max_DrawDown": maxDD,
            "Number_Of_Trades": noTrades,
            "Win_Rate": winR
        }

        return self.backtest_data

    def total_return(self):
        if self.backtest_data is None:
            raise ValueError("Call run() first")
        startPosition = self.starting_capital
        finalPosition = self.backtest_data["Portfolio_Value"].iloc[-1]
        return (finalPosition-startPosition)/startPosition
    
    def sharpe_ratio(self):
        if self.backtest_data is None:
            raise ValueError("Call run() first")
        pctChangeVal = (self.backtest_data["Portfolio_Value"].pct_change()).dropna()
        dailyMeanVal = pctChangeVal.mean()
        annualReturn = dailyMeanVal*252
        dailyStd = pctChangeVal.std()
        annualVolatility = dailyStd * np.sqrt(252)
        if annualVolatility<0.0001:
            if annualReturn > 0.02:
                return np.inf
            return 0
        sharpeRatio = (annualReturn-0.02)/annualVolatility
        return sharpeRatio

    def max_drawdown(self):
        if self.backtest_data is None:
            raise ValueError("Call run() first")
        runningMax = self.backtest_data["Portfolio_Value"].cummax()
        drawDown = (runningMax - self.backtest_data["Portfolio_Value"])/runningMax
        return drawDown.max()
    
    def number_of_trades(self): 
        if self.backtest_data is None:
            raise ValueError("Call run() first")
        return len(self.trade_data)
    
    def win_rate(self): 
        if self.backtest_data is None:
            raise ValueError("Call run() first")
        if self.trade_data.empty:
            return 0
        winning_trades = (self.trade_data["Trade_Return"]>0).sum()
        total_trades = self.number_of_trades()
        if total_trades == 0:
            return 0
        return winning_trades/total_trades
