import pandas as pd
import numpy as np
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')

def calculate_garch_volatility(price_series):
    log_returns = np.log(price_series / price_series.shift(1)) * 100
    log_returns = log_returns.dropna()

    if log_returns.empty:
        print("Error: Not enough data points to calculate returns.")
        return None, None

    print("\nCalculated Log Returns Sample:")
    print(log_returns.head())

    try:
        model = arch_model(log_returns, vol='Garch', p=1, q=1, dist='Normal')
        garch_fit = model.fit(disp='off')

        print("\nGARCH Model Summary:")
        print(garch_fit.summary())
        return log_returns, garch_fit
    except Exception as e:
        print(f"Error fitting GARCH model: {e}")
        return None, None


def predict_next_day_volatility(garch_fit):
    if garch_fit is None:
        return None
    try:
        forecast = garch_fit.forecast(horizon=1)
        predicted_variance = forecast.variance.iloc[-1, 0]
        print(f"\nPredicted Variance for next day (h.1): {predicted_variance:.4f}")
        return predicted_variance
    except Exception as e:
        print(f"Error predicting volatility: {e}")
        return None

if __name__ == "__main__":
    try:
        df = pd.read_csv("synthetic_oilseed_data.csv", index_col='Date', parse_dates=True)
        print("Loaded synthetic data.")
    except FileNotFoundError:
        print("Error: synthetic_oilseed_data.csv not found.")
        print("Please run generate_synthetic_data.py first.")
        exit()

    ncdex_prices = df['NCDEX_Close']
    returns, fitted_model = calculate_garch_volatility(ncdex_prices)

    if fitted_model:
        garch_volatility_score = predict_next_day_volatility(fitted_model)

        if garch_volatility_score is not None:
            print(f"\nGARCH Volatility Score (Predicted Variance): {garch_volatility_score}")
        else:
            print("\nFailed to get GARCH Volatility Score.")
    else:
        print("\nFailed to fit GARCH model.")
