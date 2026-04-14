import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of Brazilian stocks
tickers = ["VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"]

# Download historical adjusted closing prices
prices = yf.download(tickers, start="2023-01-01", end="2025-01-01")["Close"]

# Drop missing values
prices = prices.dropna()

# Calculate daily returns
returns = prices.pct_change().dropna()

# Portfolio weights (equal weights)
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Average daily returns
mean_returns = returns.mean()

# Covariance matrix
cov_matrix = returns.cov()

# Portfolio expected daily return
portfolio_return = np.dot(weights, mean_returns)

# Portfolio daily volatility
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Sharpe ratio (assuming risk-free rate = 0 for simplicity)
sharpe_ratio = portfolio_return / portfolio_volatility

print("Average daily returns:")
print(mean_returns)
print("\nCovariance matrix:")
print(cov_matrix)
print("\nPortfolio expected daily return:", portfolio_return)
print("Portfolio daily volatility:", portfolio_volatility)
print("Portfolio Sharpe ratio:", sharpe_ratio)

# Plot normalized prices
normalized_prices = prices / prices.iloc[0]

plt.figure(figsize=(10, 6))
for column in normalized_prices.columns:
    plt.plot(normalized_prices.index, normalized_prices[column], label=column)

plt.title("Normalized Stock Prices")
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.legend()
plt.grid(True)
plt.show()

# Plot correlation matrix
correlation_matrix = returns.corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap="viridis", interpolation="none")
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
