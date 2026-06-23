def verify_indicators(self):
    errors = []
    warnings = []
    
    # Overall Data checks
    if not self.indicator_data.index.is_monotonic_increasing:
        errors.append("Dates not Increasing")
    if self.indicator_data.index.duplicated().sum() > 0:
        errors.append("Date Duplicates Error")
    numeric_df = self.indicator_data.select_dtypes(include=np.number)
    has_inf = np.isinf(numeric_df).any().any()
    if has_inf:
        errors.append("Infinite Data Error")
    
    # Checking RSI
    invalid_rsi = (self.indicator_data["RSI_14"] < 0) | (self.indicator_data["RSI_14"] > 100)
    if invalid_rsi.sum() > 0:
        warnings.append("RSI Invalid Error")
        
    # Checking Rolling STD
    negative_std = self.indicator_data["Rolling_STD_20"] < 0
    if negative_std.sum() > 0:
        errors.append("Negative STD Error")
        
    # Checking Simple Returns
    returns = abs(self.indicator_data["Simple_Return"]) > 0.5
    if returns.sum() > 0:
        warnings.append(f"Simple Returns {returns.sum()} Large Returns error")        
        
    # Checking Bollinger Bands
    clean_bands = self.indicator_data[['Bollinger_Upper_20', 'Bollinger_Middle_20', 'Bollinger_Lower_20']].dropna()
    if not clean_bands.empty:
        boll_error = (clean_bands["Bollinger_Upper_20"] < clean_bands["Bollinger_Middle_20"]) | \
                     (clean_bands["Bollinger_Middle_20"] < clean_bands["Bollinger_Lower_20"])
        if boll_error.any():
            errors.append(f"Bollinger Band Cross Error: {boll_error.sum()} rows invalid.")

    # Checking SMA:
    smaCheck1 = self.indicator_data["SMA_20"] > self.indicator_data["Rolling_Max_20"]
    smaCheck2 = self.indicator_data["SMA_20"] < self.indicator_data["Rolling_Min_20"]
    if (smaCheck1 | smaCheck2).any():
        errors.append("SMA error")

    if len(errors) != 0:
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
