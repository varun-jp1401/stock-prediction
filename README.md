# 🍎 Apple Stock Market Forecasting — ML-Based Analysis

> **Internal Assessment Project | Financial Data Analysis**  
> Predicting Apple Inc. (AAPL) next-day closing price using Machine Learning & Deep Learning models with live data from Yahoo Finance.

---

## 📌 Project Overview

This project builds a complete end-to-end stock price forecasting pipeline for **Apple Inc. (AAPL)** using three ML models combined into an ensemble. It fetches **live data** via the `yfinance` API — no CSV downloads needed. The notebook covers everything from raw data ingestion to technical indicator engineering, model training, and a real-time "yesterday → today" prediction comparison.

---

## 🧠 Models Used

| Model | Type | Description |
|---|---|---|
| 🌲 Random Forest | Ensemble (Trees) | 500 estimators, depth-12, feature importance |
| ⚡ XGBoost | Gradient Boosting | 1000 estimators, early stopping on validation |
| 🧠 LSTM | Deep Learning (RNN) | 3-layer LSTM + BatchNorm + Dropout, 30-day look-back |
| 🏆 Ensemble | Weighted Average | Inverse-RMSE weighted combination of all three |

---

## 📊 Features & Visualizations

### Exploratory Data Analysis
- Interactive candlestick chart with volume (Plotly)
- Daily returns, cumulative return, rolling 30-day volatility
- Return distribution histogram
- Correlation heatmap across all features

### Trend & Seasonality
- Time-series decomposition (Trend + Seasonal + Residual)
- Average return by **month** and **day of week**
- Quarterly returns year-over-year

### Technical Indicators (engineered as ML features)
- Moving Averages: MA 7/20/50/200, EMA 7/20
- Bollinger Bands (upper, lower, width)
- RSI (14-day)
- MACD + Signal + Histogram
- ATR (Average True Range)
- On-Balance Volume (OBV)
- Lag features (1, 2, 3, 5, 10 days)
- Rolling mean & std (5, 10, 20 days)
- Calendar features (day of week, month, quarter, month start/end)

### Model Results
- Predicted vs Actual price chart (60-day test period)
- Residual analysis per model
- Scatter plots with R² score
- Feature importance (RF + XGB)
- Model comparison: MAE, RMSE, MAPE, R²

### Today's Prediction (Live)
- Uses **yesterday's features** to predict **today's close**
- Compares prediction vs live actual close fetched from Yahoo Finance
- Gauge meters showing accuracy per model
- Directional accuracy (did we predict UP/DOWN correctly?)
- Roll-forward error timeline

---

## 📁 Project Structure

```
aapl-stock-forecasting/
│
├── AAPL_Stock_Prediction.ipynb     ← Main notebook (23 cells)
├── AAPL_TodayPrediction_AddOn.ipynb← Today vs Yesterday prediction (Cells A–H)
├── AAPL_Graph_Code.py              ← Standalone graph generation script
├── requirements.txt                ← All dependencies
└── README.md                       ← This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/aapl-stock-forecasting.git
cd aapl-stock-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook
```

### 4. Run the notebook
Open `AAPL_Stock_Prediction.ipynb` and run all cells top to bottom.  
**Cell 1 auto-installs all libraries** if they're missing.

---

## 📦 Requirements

```
yfinance
numpy
pandas
matplotlib
seaborn
plotly
mplfinance
scikit-learn
xgboost
tensorflow
statsmodels
jupyter
```

> Install all at once:
> ```bash
> pip install yfinance numpy pandas matplotlib seaborn plotly mplfinance scikit-learn xgboost tensorflow statsmodels jupyter
> ```

---

## 📈 Sample Results

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| Random Forest | ~2.10 | ~2.80 | ~1.1% | ~0.97 |
| XGBoost | ~1.85 | ~2.50 | ~0.95% | ~0.98 |
| LSTM | ~1.70 | ~2.30 | ~0.85% | ~0.98 |
| **Ensemble** | **~1.55** | **~2.10** | **~0.78%** | **~0.99** |

> *Results vary with market conditions. Past accuracy does not guarantee future performance.*

---

## 🔮 How the "Yesterday → Today" Prediction Works

```
Yesterday's OHLCV + All Technical Indicators
            │
            ▼
    ┌───────────────┐
    │  Random Forest│──┐
    ├───────────────┤  │
    │   XGBoost     │──┼──► Weighted Ensemble ──► Today's Predicted Close
    ├───────────────┤  │
    │     LSTM      │──┘   (weights = inverse of each model's RMSE)
    └───────────────┘
            │
            ▼
    Compare vs Live Actual Close (fetched from Yahoo Finance)
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only** as part of an internal assessment. It is **not financial advice**. Stock markets are inherently unpredictable and past model performance does not guarantee future results. Do not use this to make real investment decisions.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo%20Finance-purple)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)

---

## 👤 Author

**Your Name**  
Internal Assessment — Financial Data Analysis  
*Add your school/college name here*
