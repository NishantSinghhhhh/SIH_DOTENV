import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
except ImportError:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.utils import to_categorical

import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

warnings.filterwarnings('ignore')


def create_sequences(data, seq_length):
    xs, ys_indices = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y_index = i + seq_length
        if y_index < len(data):
             ys_indices.append(y_index)
             xs.append(x)
    return np.array(xs), np.array(ys_indices)

def define_trend(returns, threshold=0.001):
    if returns > threshold:
        return 1
    elif returns < -threshold:
        return -1
    else:
        return 0

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("\nLSTM Model Summary:")
    # model.summary()
    return model

if __name__ == "__main__":
    try:
        df = pd.read_csv("synthetic_oilseed_data.csv", index_col='Date', parse_dates=True)
        print("Loaded synthetic data.")
    except FileNotFoundError:
        print("Error: synthetic_oilseed_data.csv not found.")
        print("Please run generate_synthetic_data.py first.")
        exit()

    df['Log_Return'] = np.log(df['NCDEX_Close'] / df['NCDEX_Close'].shift(1))
    df.dropna(subset=['Log_Return'], inplace=True)

    df['Actual_Trend'] = df['Log_Return'].shift(-1).apply(define_trend)
    df.dropna(subset=['Actual_Trend'], inplace=True)

    trend_mapping = {-1: 0, 0: 1, 1: 2} # Bearish: 0, Neutral: 1, Bullish: 2
    df['Trend_Label'] = df['Actual_Trend'].map(trend_mapping).astype(int)

    if df.empty or 'Log_Return' not in df.columns or df['Log_Return'].isnull().all():
        print("Error: Not enough valid data after calculating returns and trend.")
        exit()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Log_Return']])

    SEQUENCE_LENGTH = 10
    X_sequences, y_indices = create_sequences(scaled_data, SEQUENCE_LENGTH)

    y_labels = df['Trend_Label'].iloc[y_indices].values

    if X_sequences.shape[0] == 0:
        print(f"Error: Not enough data ({len(df)} rows) to create sequences of length {SEQUENCE_LENGTH}.")
        exit()

    num_classes = 3

    unique_labels = np.unique(y_labels)
    print(f"Unique labels found: {unique_labels}")

    y_categorical = to_categorical(y_labels, num_classes=num_classes)


    X_sequences = np.reshape(X_sequences, (X_sequences.shape[0], SEQUENCE_LENGTH, 1))

    split_ratio = 0.8
    split_index = int(len(X_sequences) * split_ratio)
    X_train, X_test = X_sequences[:split_index], X_sequences[split_index:]
    y_train, y_test = y_categorical[:split_index], y_categorical[split_index:]

    print(f"\nTraining data shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing data shape: X_test {X_test.shape}, y_test {y_test.shape}")

    if y_train.shape[1] != num_classes or y_test.shape[1] != num_classes:
        print(f"Error: Label shape mismatch. Expected {num_classes} classes, but got {y_train.shape[1]}.")
        exit()

    model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, 1), num_classes=num_classes)

    print("\nTraining LSTM model (this might take a moment)...")
    history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1, verbose=0)
    print("Training complete.")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Test Accuracy: {accuracy*100:.2f}%")

    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    last_sequence_reshaped = np.reshape(last_sequence, (1, SEQUENCE_LENGTH, 1))

    prediction = model.predict(last_sequence_reshaped, verbose=0)
    predicted_label_index = np.argmax(prediction)

    reverse_trend_mapping = {0: "Bearish (-1)", 1: "Neutral (0)", 2: "Bullish (+1)"}
    predicted_trend_name = reverse_trend_mapping.get(predicted_label_index, "Unknown")

    print(f"\nPredicted Trend for the next day: {predicted_trend_name} (Class Index: {predicted_label_index})")
    print(f"Prediction Probabilities (Bearish, Neutral, Bullish): {prediction[0]}")
