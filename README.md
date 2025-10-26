# 🛡️ Farmer's Financial Shield

An AI-driven decision support system providing intelligent hedging recommendations for Indian oilseed farmers through advanced machine learning models.

## 📋 Overview

Farmer's Financial Shield combines multiple machine learning models to analyze market conditions and provide actionable **Hedge/Wait** recommendations. The system integrates price volatility analysis, trend prediction, sentiment analysis, and advanced classification to help farmers make informed hedging decisions.

### Key Features

- **Multi-Model Architecture**: Combines four specialized ML models for comprehensive market analysis
- **Real-Time Analysis**: Processes market data, trends, and sentiment simultaneously
- **Risk Management**: Focuses on protecting farmers from adverse price movements
- **Interpretable Decisions**: Provides clear Hedge/Wait recommendations with confidence scores

## 🏗️ System Architecture

The system employs a hierarchical ensemble approach with four complementary models:

<img width="2423" height="1246" alt="image" src="https://github.com/user-attachments/assets/6aae7122-c9fb-41fd-9062-5f50a6a64950" />


### Model Components

#### 1. **GARCH Model** (`garch_volatility.py`)
Predicts price volatility using the GARCH(1,1) specification.

**Mathematical Foundation:**

Logarithmic Return:
```
r_t = ln(P_t / P_{t-1})
```

GARCH(1,1) Variance Forecast:
```
σ²_t = ω + α₁·r²_{t-1} + β₁·σ²_{t-1}
```

Where:
- `σ²_t`: Predicted variance for time t
- `ω`: Constant term
- `α₁`: ARCH coefficient (previous squared return impact)
- `r²_{t-1}`: Squared log return at t-1
- `β₁`: GARCH coefficient (previous variance persistence)
- `σ²_{t-1}`: Previous variance forecast

**Purpose**: Captures volatility clustering and heteroskedasticity in price data

#### 2. **LSTM Model** (`lstm_trend_corrected.py`)
Predicts market trend direction using Long Short-Term Memory neural networks.

**Purpose**: Captures temporal dependencies and sequential patterns in price movements

#### 3. **BERT Model** (`bert_sentiment_no_comments.py`)
Analyzes sentiment from market news and textual data using transformer-based NLP.

**Purpose**: Quantifies market sentiment and qualitative factors affecting prices

#### 4. **XGBoost Classifier** (`xgboost_classifier_no_comments.py`)
Master decision model that combines outputs from all models with additional features.

**Purpose**: Integrates all signals to produce final Hedge/Wait recommendation

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for initial BERT model download)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd farmers-financial-shield
```

2. **Install dependencies**
```bash
pip install pandas numpy arch tensorflow scikit-learn transformers torch xgboost tf-keras
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

### Usage

#### Step 1: Generate Synthetic Data

Generate the synthetic dataset for model training and testing:

```bash
python generate_synthetic_data.py
```

This creates sample price data, market indicators, and text data for demonstration purposes.

#### Step 2: Run Individual Models (Optional)

Test each model independently to understand their outputs:

```bash
# Test volatility prediction
python garch_volatility.py

# Test trend prediction
python lstm_trend_corrected.py

# Test sentiment analysis (downloads model on first run)
python bert_sentiment_no_comments.py
```

#### Step 3: Generate Final Recommendations

Run the master classifier to see integrated analysis and recommendations:

```bash
python xgboost_classifier_no_comments.py
```

**Expected Output:**
- Feature importance rankings
- Model confidence scores
- Final Hedge/Wait recommendation
- Decision rationale

## 📊 Data Requirements

The system currently uses **synthetic data** for the following reasons:

- ✅ **Privacy Protection**: No exposure of real farmer or market data
- ✅ **Security**: Prevents unauthorized access to sensitive information
- ✅ **Cost Management**: Market data APIs require paid subscriptions
- ✅ **Development Flexibility**: Easy testing and iteration

### Production Deployment

For production use, integrate real data sources:

- Historical oilseed prices (Mustard, Groundnut, Soybean, etc.)
- Market indicators (supply/demand, MSP, international prices)
- News feeds and market commentary
- Weather data and seasonal patterns

## 🔧 Configuration

Model parameters can be adjusted in individual script files:

- **GARCH**: p=1, q=1 (ARCH and GARCH orders)
- **LSTM**: Sequence length, hidden units, epochs
- **BERT**: Pre-trained model selection, max sequence length
- **XGBoost**: Tree depth, learning rate, number of estimators

## 📈 Model Performance

Each model contributes specific insights:

| Model | Primary Output | Use Case |
|-------|---------------|----------|
| GARCH | Volatility Score | Risk assessment |
| LSTM | Trend Direction | Price movement prediction |
| BERT | Sentiment Score | Market mood analysis |
| XGBoost | Final Decision | Integrated recommendation |

## 🎯 Interpretation Guide

### Recommendation Types

**HEDGE** - Recommended when:
- High volatility predicted
- Downward trend detected
- Negative market sentiment
- Combined signals indicate price risk

**WAIT** - Recommended when:
- Low volatility environment
- Upward or stable trend
- Positive market sentiment
- Limited downside risk identified

