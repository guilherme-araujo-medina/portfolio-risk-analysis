import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime, timedelta

# =========================================================
# Portfolio Risk and Return Analysis with Brazilian Equities
# Data window: last 12 months
# =========================================================

tickers = [
    'ABEV3.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC4.SA', 'TIMS3.SA',
    'CSAN3.SA', 'EQTL3.SA', 'GGBR4.SA', 'HAPV3.SA', 'ITUB4.SA',
    'KLBN11.SA', 'LREN3.SA', 'MGLU3.SA', 'PETR4.SA', 'RADL3.SA',
    'RENT3.SA', 'SUZB3.SA', 'VALE3.SA', 'VIVT3.SA', 'WEGE3.SA'
]

end_date = datetime.today().date() - timedelta(days=1)
start_date = end_date - timedelta(days=365)

portfolio_value = 1_000_000
n_simulations = 50000

print(f"Using data from {start_date} to {end_date}")

# =====================
# Download price data
# =====================
price_series = []
valid_tickers = []
failed_tickers = []

for ticker in tickers:
    try:
        data = yf.download(
            ticker,
            start=str(start_date),
            end=str(end_date),
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            failed_tickers.append(ticker)
            continue

        if "Adj Close" in data.columns:
            series = data["Adj Close"].copy()
        else:
            series = data["Close"].copy()

        series.name = ticker
        price_series.append(series)
        valid_tickers.append(ticker)

    except Exception:
        failed_tickers.append(ticker)

if not price_series:
    raise ValueError("No ticker returned valid price data.")

prices = pd.concat(price_series, axis=1)
prices = prices.dropna()

if prices.empty:
    raise ValueError("Price dataset is empty after cleaning.")

returns = prices.pct_change(fill_method=None).dropna()

if returns.empty:
    raise ValueError("Returns dataset is empty after calculation.")

print("Valid tickers:", valid_tickers)
print("Failed tickers:", failed_tickers)

# =====================
# Summary statistics
# =====================
mean_returns = returns.mean()
cov_matrix = returns.cov()
n_assets = len(valid_tickers)

# =====================
# Simulate random portfolios
# =====================
np.random.seed(42)
weights_sim = np.random.dirichlet(np.ones(n_assets), size=n_simulations)

ret_sim = weights_sim @ mean_returns.values
vol_sim = np.sqrt(np.einsum('ij,jk,ik->i', weights_sim, cov_matrix.values, weights_sim))
sharpe_sim = np.divide(ret_sim, vol_sim, out=np.zeros_like(ret_sim), where=vol_sim != 0)

# =====================
# Optimization settings
# =====================
w0 = np.repeat(1 / n_assets, n_assets)
bounds = [(0, 1)] * n_assets
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# =====================
# Minimum variance portfolio
# =====================
res_min = minimize(
    lambda w: np.sqrt(w @ cov_matrix.values @ w),
    w0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

w_min = res_min.x
ret_min = w_min @ mean_returns.values
vol_min = np.sqrt(w_min @ cov_matrix.values @ w_min)

# =====================
# Maximum return portfolio
# =====================
res_max_ret = minimize(
    lambda w: -(w @ mean_returns.values),
    w0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

w_max_ret = res_max_ret.x
ret_max = w_max_ret @ mean_returns.values
vol_max = np.sqrt(w_max_ret @ cov_matrix.values @ w_max_ret)

# =====================
# Maximum Sharpe ratio portfolio
# =====================
def negative_sharpe(w):
    r = w @ mean_returns.values
    v = np.sqrt(w @ cov_matrix.values @ w)
    if v == 0:
        return 1e9
    return -r / v

res_sharpe = minimize(
    negative_sharpe,
    w0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

w_sharpe = res_sharpe.x
ret_sharpe = w_sharpe @ mean_returns.values
vol_sharpe = np.sqrt(w_sharpe @ cov_matrix.values @ w_sharpe)

# =====================
# Efficient frontier (upper branch)
# =====================
target_returns_efficient = np.linspace(ret_min, ret_max, 100)
frontier_vol_efficient = []
frontier_ret_efficient = []

w_start = w_min.copy()

for target in target_returns_efficient:
    cons_target = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, target=target: w @ mean_returns.values - target}
    ]

    result = minimize(
        lambda w: w @ cov_matrix.values @ w,
        w_start,
        method='SLSQP',
        bounds=bounds,
        constraints=cons_target
    )

    if result.success:
        w_opt = result.x
        vol_opt = np.sqrt(w_opt @ cov_matrix.values @ w_opt)
        ret_opt = w_opt @ mean_returns.values

        frontier_vol_efficient.append(vol_opt)
        frontier_ret_efficient.append(ret_opt)

        w_start = w_opt.copy()

# =====================
# Risk metrics
# =====================
z_95 = norm.ppf(0.05)

var_parametric = -z_95 * vol_min * portfolio_value

portfolio_returns_min = returns @ w_min
var_historical = portfolio_value * (1 - np.percentile(1 + portfolio_returns_min, 5))

simulated_returns = np.random.normal(loc=ret_min, scale=vol_min, size=10000)
simulated_values = portfolio_value * (1 + simulated_returns)

var_cutoff = np.percentile(simulated_values, 5)
var_monte_carlo = portfolio_value - var_cutoff
cvar_monte_carlo = portfolio_value - np.mean(simulated_values[simulated_values <= var_cutoff])

individual_vars = -z_95 * returns.std() * w_min * portfolio_value
var_non_diversified = individual_vars.sum()
diversification_effect = var_non_diversified - var_parametric

# =====================
# Output results
# =====================
print("\n===== MARKET RISK RESULTS =====")
print(f"Parametric VaR (95%): R$ {var_parametric:,.2f}")
print(f"Historical VaR (95%): R$ {var_historical:,.2f}")
print(f"Monte Carlo VaR (95%): R$ {var_monte_carlo:,.2f}")
print(f"Monte Carlo CVaR (95%): R$ {cvar_monte_carlo:,.2f}")
print(f"Non-diversified VaR: R$ {var_non_diversified:,.2f}")
print(f"Diversification effect: R$ {diversification_effect:,.2f}")

def show_portfolio(name, portfolio_return, portfolio_volatility, weights, labels):
    print(f"\n{name}")
    print(f"Expected return: {portfolio_return:.6f}")
    print(f"Volatility: {portfolio_volatility:.6f}")
    print("Weights:")
    for asset, weight in zip(labels, weights):
        if weight > 0.0001:
            print(f" - {asset}: {weight:.2%}")

show_portfolio("Minimum Variance Portfolio", ret_min, vol_min, w_min, valid_tickers)
show_portfolio("Maximum Return Portfolio", ret_max, vol_max, w_max_ret, valid_tickers)
show_portfolio("Maximum Sharpe Portfolio", ret_sharpe, vol_sharpe, w_sharpe, valid_tickers)

# =====================
# Visualization: Efficient Frontier
# =====================
plt.figure(figsize=(14, 9))
scatter = plt.scatter(vol_sim, ret_sim, c=sharpe_sim, cmap='viridis', alpha=0.35)
plt.plot(frontier_vol_efficient, frontier_ret_efficient, color='red', lw=2.5, label='Efficient Frontier')
plt.scatter(vol_min, ret_min, marker='*', color='blue', s=250, label='Minimum Variance')
plt.scatter(vol_sharpe, ret_sharpe, marker='*', color='orange', s=250, label='Maximum Sharpe')

plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Simulated Portfolios and Efficient Frontier (Last 12 Months)')
plt.colorbar(scatter, label='Sharpe Ratio')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# =====================
# Visualization: Monte Carlo
# =====================
plt.figure(figsize=(12, 7))
plt.hist(simulated_values, bins=100, edgecolor='black', alpha=0.75)
plt.axvline(var_cutoff, color='red', linestyle='--', linewidth=2, label='VaR (5%)')
plt.axvline(np.mean(simulated_values[simulated_values <= var_cutoff]),
            color='darkred', linestyle=':', linewidth=2, label='CVaR')

plt.title('Monte Carlo Simulation - Portfolio Value Distribution (Last 12 Months)')
plt.xlabel('Simulated Portfolio Value (R$)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
