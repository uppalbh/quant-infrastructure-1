class MetaModelDataset:
    def __init__(self, instrument, window_days = 20): #How many days before data should it use
        self.instrument = instrument
        self.window_days = window_days
        self.dataset = None #The final dataset with features, labels, index
    
    def create_features(self):
        #Get data from instrument
        df = self.instrument.indicator_data.copy()

        #Feature 1: Trend: EMA Distance
        df["Trend"] = (df["Close"] - df["EMA_20"]) / df["EMA_20"]
        
        # Volatility: Vol_Ratio (normalized, better than raw STD)
        df["Volatility_Regime"] = (df["Rolling_STD_20"] / df["Rolling_STD_20"].rolling(50).mean())
        
        # Momentum: RSI (standard, well-understood)
        df["Momentum"] = df["RSI_14"]
        
        # Extremes: Price_Percentile (clearer than BB_Distance)
        df["Overbought_Oversold"] = (df["Close"] - df["Rolling_Min_20"]) / (df["Rolling_Max_20"] - df["Rolling_Min_20"])
        
        # Pattern: MACD_Hist (captures momentum crossovers)
        df["Momentum_Signal"] = df["MACD_Histogram"] / df["MACD_Histogram"].std()
        
        # Regime Change: ATR_expansion (volatility is changing)
        df["Vol_Change"] = (df["ATR_14"] / df["ATR_14"].shift(5)) - 1
        
        features_df = df[["Trend", "Volatility_Regime", "Momentum", 
                        "Overbought_Oversold", "Momentum_Signal", "Vol_Change"]]
        return features_df.dropna()
    
    def create_labels(self):
        strategies = {
            "MovingAvgCrossover": MovingAverageCrossover(self.instrument),
            "Momentum": Momentum(self.instrument),
            "MeanReversion": MeanReversion(self.instrument),
            "Breakout": Breakout(self.instrument)
        }
        
        close = self.instrument.indicator_data["Close"]
        daily_returns = close.pct_change().fillna(0)

        daily_winners = []
        strategy_signals = {}

        for name, strategy in strategies.items():
            signals = strategy.generate_signals()["Signal"]
            signals = signals.shift(1).fillna(0)

            strategy_signals[name]= signals

        # Instead of calculating returns, checking which strategy's signal was correct
        for idx in range(self.window_days, len(daily_returns)-1):
            strategy_correctness = {}
            for name_, signal_series in strategy_signals.items():
                signal_today = signal_series.iloc[idx]  # 1, -1, or 0
                actual_direction = np.sign(daily_returns.iloc[idx+1:idx+6].sum())  # +1, -1, or 0
                
                if signal_today == actual_direction and actual_direction != 0:
                    score = 1.0  # Correct prediction
                else:
                    score = 0.0  # Wrong prediction
                
                strategy_correctness[name_] = score
            
            winner = max(strategy_correctness, key = strategy_correctness.get)

            daily_winners.append({
                "Strategy": winner
            })
        labels_df =  pd.DataFrame(
            daily_winners,
            index = daily_returns.index[self.window_days:len(daily_returns)-1]
            )
        return labels_df

    def create_database(self):
        features = self.create_features()
        labels = self.create_labels()
        dataset = features.join(labels, how = "inner")
        self.dataset = dataset.dropna()
        return self.dataset
