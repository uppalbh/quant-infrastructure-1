class StrategySelector: 
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, 
                                            random_state= 42, min_samples_split=10, max_features="sqrt",
                                            class_weight="balanced")
        self.features = None
        self.scaler = StandardScaler() 
    
    def train(self, dataset, train_pct=0.6, val_pct=0.2):
        y = dataset["Strategy"]
        X = dataset.drop(columns = "Strategy")
        self.features = X
        n = len(dataset)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        X_train = X.iloc[:train_end]
        X_cv = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        Y_train = y.iloc[:train_end]
        Y_cv = y.iloc[train_end:val_end]
        Y_test = y.iloc[val_end:]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_cv_scaled = self.scaler.transform(X_cv)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, Y_train)

        train_Acc = self.model.score(X_train_scaled, Y_train)
        cv_Acc = self.model.score(X_cv_scaled, Y_cv)
        test_Acc = self.model.score(X_test_scaled, Y_test)
        baseline_Acc = 1/len(y.unique())

        if (test_Acc > baseline_Acc+0.05): print("Model Pass")
        else: print("Model Failed")
        return train_Acc, test_Acc
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X) #X should have 'Trend', 'Volatility_Regime', 'Momentum', 'Overbought_Oversold',
       #'Momentum_Signal', 'Vol_Change'
        return self.model.predict(X_scaled)
