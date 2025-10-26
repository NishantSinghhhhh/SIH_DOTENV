import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def prepare_features(df):
    df['Log_Return'] = np.log(df['NCDEX_Close'] / df['NCDEX_Close'].shift(1))
    df['Volume_Change'] = df['NCDEX_Volume'].pct_change() * 100
    df['Global_Soy_Return'] = np.log(df['Global_Soy_Close'] / df['Global_Soy_Close'].shift(1))
    df['INR_Return'] = np.log(df['USD_INR_Rate'] / df['USD_INR_Rate'].shift(1))

    np.random.seed(42)
    df['GARCH_Volatility_Score'] = np.random.rand(len(df)) * 5 + 1
    df['LSTM_Trend_Signal'] = np.random.choice([-1, 0, 1], size=len(df), p=[0.4, 0.2, 0.4])
    df['BERT_Sentiment_Score'] = np.random.uniform(-1, 1, size=len(df))

    df.dropna(inplace=True)

    df['Target_Return'] = df['Log_Return'].shift(-1)
    threshold = -0.01
    df['Hedge_Label'] = df['Target_Return'].apply(lambda x: 1 if x < threshold else 0)
    df.dropna(subset=['Target_Return'], inplace=True)

    return df

if __name__ == "__main__":
    try:
        df = pd.read_csv("synthetic_oilseed_data.csv", index_col='Date', parse_dates=True)
        print("Loaded synthetic data.")
    except FileNotFoundError:
        print("Error: synthetic_oilseed_data.csv not found.")
        print("Please run generate_synthetic_data.py first.")
        exit()

    df_features = prepare_features(df.copy())

    if df_features.empty:
         print("Error: Not enough data after feature preparation.")
         exit()

    feature_columns = [
        'Log_Return', 'Volume_Change', 'Global_Soy_Return', 'INR_Return',
        'GARCH_Volatility_Score', 'LSTM_Trend_Signal', 'BERT_Sentiment_Score'
    ]
    target_column = 'Hedge_Label'

    X = df_features[feature_columns]
    y = df_features[target_column]

    if len(X) < 10:
        print("Error: Not enough data points to train/test the model.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining data shape: X_train {X_train_scaled.shape}, y_train {y_train.shape}")
    print(f"Testing data shape: X_test {X_test_scaled.shape}, y_test {y_test.shape}")

    xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)

    print("\nTraining XGBoost model...")
    xgb_classifier.fit(X_train_scaled, y_train)
    print("Training complete.")

    y_pred = xgb_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Test Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    last_features = X.iloc[-1:]
    last_features_scaled = scaler.transform(last_features)

    final_prediction_proba = xgb_classifier.predict_proba(last_features_scaled)
    final_prediction = xgb_classifier.predict(last_features_scaled)[0]

    hedge_recommendation = "HEDGE NOW" if final_prediction == 1 else "WAIT"

    print("\n--- Final Recommendation ---")
    print(f"Features for last available day ({last_features.index[0].date()}):")
    print(last_features.to_string())
    print(f"\nPredicted Probabilities [Wait (0), Hedge (1)]: {final_prediction_proba[0]}")
    print(f"Final Prediction (0=Wait, 1=Hedge): {final_prediction}")
    print(f"Recommendation: {hedge_recommendation}")

