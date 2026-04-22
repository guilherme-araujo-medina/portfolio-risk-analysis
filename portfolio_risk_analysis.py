import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# Parameters
tickers = [
    'ABEV3.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC4.SA', 'BRFS3.SA',
    'CSAN3.SA', 'ELET3.SA', 'GGBR4.SA', 'HAPV3.SA', 'ITUB4.SA',
    'KLBN11.SA', 'LREN3.SA', 'MGLU3.SA', 'PETR4.SA', 'RADL3.SA',
    'RENT3.SA', 'SUZB3.SA', 'VALE3.SA', 'VIVT3.SA', 'WEGE3.SA'
]

start_date = '2023-07-01'
end_date = '2024-07-01'
portfolio_value = 1_000_000
n_simulations = 50000

# Download data
prices = yf.download(tickers, start=start_date, end=end_date)['Close']
returns = prices.pct_change().dropna()

# Summary statistics
mean_returns = returns.mean()
cov_matrix = returns.cov()
n_assets = len(tickers)

# Simulate random portfolios
np.random.seed(42)
weights_sim = np.random.dirichlet(np.ones(n_assets), size=n_simulations)
ret_sim = weights_sim @ mean_returns.values
vol_sim = np.sqrt(np.einsum('ij,jk,ik->i', weights_sim, cov_matrix.values, weights_sim))
sharpe_sim = ret_sim / vol_sim

# Optimization settings
w0 = np.repeat(1 / n_assets, n_assets)
bounds = [(0, 1)] * n_assets
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# Minimum variance portfolio
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

# Maximum return portfolio
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

# Maximum Sharpe portfolio
def negative_sharpe(w):
    r = w @ mean_returns.values
    v = np.sqrt(w @ cov_matrix.values @ w)
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

# Efficient frontier
target_returns = np.linspace(ret_sim.min(), ret_sim.max(), 100)
frontier_vol = []

for target in target_returns:
    cons_target = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, target=target: w @ mean_returns.values - target}
    ]
    result = minimize(
        lambda w: np.sqrt(w @ cov_matrix.values @ w),
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons_target
    )
    frontier_vol.append(result.fun)

# Risk metrics
z_95 = norm.ppf(0.05)

var_parametric = -z_95 * vol_min * portfolio_value
var_historical = portfolio_value * (1 - np.percentile(1 + returns @ w_min, 5))

simulated_returns = np.random.normal(loc=ret_min, scale=vol_min, size=10000)
simulated_values = portfolio_value * (1 + simulated_returns)

var_monte_carlo = portfolio_value - np.percentile(simulated_values, 5)
cvar_monte_carlo = portfolio_value - np.mean(
    simulated_values[simulated_values <= np.percentile(simulated_values, 5)]
)

# Non-diversified VaR
individual_vars = -z_95 * returns.std() * w_min * portfolio_value
var_non_diversified = individual_vars.sum()
diversification_effect = var_non_diversified - var_parametric

# Plot simulated portfolios and efficient frontier
plt.figure(figsize=(14, 9))
plt.scatter(vol_sim, ret_sim, c=sharpe_sim, cmap='viridis', alpha=0.3)
plt.plot(frontier_vol, target_returns, color='red', lw=2.5, label='Efficient Frontier')
plt.scatter(vol_min, ret_min, marker='*', color='blue', s=250, label='Minimum Variance')
plt.scatter(vol_max, ret_max, marker='*', color='green', s=250, label='Maximum Return')
plt.scatter(vol_sharpe, ret_sharpe, marker='*', color='orange', s=250, label='Maximum Sharpe')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Simulated Portfolios and Efficient Frontier')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Monte Carlo distribution
plt.figure(figsize=(10, 6))
plt.hist(simulated_values, bins=100, edgecolor='black', alpha=0.7)
plt.axvline(np.percentile(simulated_values, 5), color='red', linestyle='--', label='VaR (5%)')
plt.axvline(
    np.mean(simulated_values[simulated_values <= np.percentile(simulated_values, 5)]),
    color='darkred',
    linestyle=':',
    label='CVaR'
)
plt.title('Monte Carlo Simulation - Portfolio Value Distribution')
plt.xlabel('Simulated Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Display results
print(f"Parametric VaR (95%): R$ {var_parametric:.2f}")
print(f"Historical VaR (95%): R$ {var_historical:.2f}")
print(f"Monte Carlo VaR (95%): R$ {var_monte_carlo:.2f}")
print(f"Monte Carlo CVaR (95%): R$ {cvar_monte_carlo:.2f}")
print(f"Non-diversified VaR: R$ {var_non_diversified:.2f}")
print(f"Diversification effect: R$ {diversification_effect:.2f}")

def show_portfolio(name, portfolio_return, portfolio_volatility, weights):
    print(f"\n{name}")
    print(f"Expected return: {portfolio_return:.5f}")
    print(f"Volatility: {portfolio_volatility:.5f}")
    print("Weights:")
    for asset, weight in zip(tickers, weights):
        if weight > 0.0001:
            print(f" - {asset}: {weight:.2%}")

show_portfolio("Minimum Variance Portfolio", ret_min, vol_min, w_min)
show_portfolio("Maximum Return Portfolio", ret_max, vol_max, w_max_ret)
show_portfolio("Maximum Sharpe Portfolio", ret_sharpe, vol_sharpe, w_sharpe)