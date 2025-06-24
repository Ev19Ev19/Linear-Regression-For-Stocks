#Stock Open & Close Price Prediction with Weighted Linear Regression

from polygon import RESTClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

API_KEY = "Go to Polygon to get your free API key"
client = RESTClient(API_KEY)

# Download daily stock prices up to today
today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
aggs = client.get_aggs(
    ticker="AAPL",
    multiplier=1,
    timespan="day",
    from_="2023-01-01",
    to=today_str
)

# Convert to DataFrame and sort
data = pd.DataFrame([agg.__dict__ for agg in aggs])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data = data.sort_values('timestamp').reset_index(drop=True)

# Use only the last 10 trading days
latest = data.tail(10).copy()
print("\nLast 10 days:\n", latest[['timestamp', 'open', 'close']])

# Prepare X = day index for regression
N = len(latest)
X = np.arange(N).reshape(-1, 1)

# Weighted linear regression: Open and Close
weights = np.arange(1, N + 1)  # more recent days have higher weight

model_open = LinearRegression()
model_close = LinearRegression()

model_open.fit(X, latest['open'].values, sample_weight=weights)
model_close.fit(X, latest['close'].values, sample_weight=weights)

# Predict next day's Open and Close
pred_open = model_open.predict([[N]])[0]
pred_close = model_close.predict([[N]])[0]

print(f"\nWeighted Predicted Open: {pred_open:.2f}")
print(f"Weighted Predicted Close: {pred_close:.2f}")

# Build date axis and prediction day
dates = list(latest['timestamp'])
next_date = latest['timestamp'].iloc[-1] + pd.Timedelta(days=1)
dates_plus = dates + [next_date]

# Plot Open (early) and Close (end), trends, predictions
plt.figure(figsize=(12, 6))

# Offset Open by 6 hours earlier
open_times = [d - pd.Timedelta(hours=6) for d in dates]
close_times = dates

plt.scatter(open_times, latest['open'], color='lightblue', label='Open Price')
plt.scatter(close_times, latest['close'], color='darkblue', label='Close Price')

# Weighted trend lines
trend_open = model_open.predict(np.arange(N + 1).reshape(-1, 1))
trend_close = model_close.predict(np.arange(N + 1).reshape(-1, 1))

trend_open_times = [d - pd.Timedelta(hours=6) for d in dates_plus]
trend_close_times = dates_plus

plt.plot(trend_open_times, trend_open, color='lightblue', linestyle='--', label='Weighted Open Trend')
plt.plot(trend_close_times, trend_close, color='darkblue', linestyle='--', label='Weighted Close Trend')

# Predicted next day Open and Close
plt.scatter(next_date - pd.Timedelta(hours=6), pred_open, color='red', edgecolor='black',
            label=f"Predicted Open: {pred_open:.2f}", s=100)
plt.scatter(next_date, pred_close, color='red', edgecolor='black',
            label=f"Predicted Close: {pred_close:.2f}", s=100)

plt.title("Stock Open & Close Prices with Weighted Linear Regression")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
