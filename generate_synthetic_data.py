import pandas as pd
import numpy as np
import datetime

def generate_synthetic_data(days=252):
    start_date = datetime.date.today() - datetime.timedelta(days=days - 1)
    dates = pd.date_range(start_date, periods=days, freq='B')

    price_volatility = 0.015
    ncdex_returns = np.random.normal(0.0005, price_volatility, days)
    for i in range(1, days):
        if abs(ncdex_returns[i-1]) > price_volatility * 1.5:
             ncdex_returns[i] = np.random.normal(0, price_volatility * 1.8)
        elif abs(ncdex_returns[i-1]) < price_volatility * 0.5:
             ncdex_returns[i] = np.random.normal(0, price_volatility * 0.8)
        else:
             ncdex_returns[i] = np.random.normal(0, price_volatility)

    ncdex_price = 5000 * np.exp(np.cumsum(ncdex_returns))

    volume = np.random.randint(10000, 50000, days) + (np.abs(ncdex_returns) * 500000)
    volume = volume.astype(int)

    global_soy_returns = np.random.normal(0.0003, 0.01, days)
    global_soy_price = 14 * np.exp(np.cumsum(global_soy_returns))

    inr_returns = np.random.normal(0.0001, 0.002, days)
    inr_rate = 83 * np.exp(np.cumsum(inr_returns))

    weather_feature = np.random.normal(0, 5, days)

    sowing_feature = np.random.uniform(-10, 20, days)

    news_snippets = [
        "Market remains steady despite global cues.",
        "Import duty hike expected, prices might rise.",
        "Poor monsoon forecast worries farmers, supply concerns.",
        "Excellent harvest reported in major growing regions.",
        "Government announces increase in MSP for oilseeds.",
        "Global soybean prices fall sharply.",
        "Trading volume surges on speculation.",
        "Experts predict bearish trend for coming weeks.",
        "Favorable weather boosts crop prospects.",
        "Demand expected to pick up next quarter."
    ]
    news = np.random.choice(news_snippets, days)

    df = pd.DataFrame({
        'Date': dates,
        'NCDEX_Close': ncdex_price,
        'NCDEX_Volume': volume,
        'Global_Soy_Close': global_soy_price,
        'USD_INR_Rate': inr_rate,
        'Weather_Feature': weather_feature,
        'Sowing_Feature': sowing_feature,
        'News_Snippet': news
    })

    df.set_index('Date', inplace=True)

    print("Generated Synthetic Data Sample:")
    print(df.head())
    print("\nData Shape:", df.shape)
    return df

if __name__ == "__main__":
    synthetic_df = generate_synthetic_data(days=500)
    synthetic_df.to_csv("synthetic_oilseed_data.csv")
    print("\nSynthetic data saved to synthetic_oilseed_data.csv")
