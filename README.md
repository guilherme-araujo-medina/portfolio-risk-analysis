# Market Risk and Portfolio Optimization

This project presents a quantitative analysis of market risk using a portfolio of Brazilian equities, applying portfolio theory and risk management techniques.

## Objective

The goal is to evaluate the risk and return profile of a portfolio composed of 20 Brazilian stocks, using quantitative methods such as portfolio optimization, Value at Risk (VaR), Conditional Value at Risk (CVaR) and Monte Carlo simulation.

## Data

- 20 liquid stocks from B3
- Daily adjusted closing prices
- Period: July 2023 – July 2024
- Source: Yahoo Finance (yfinance)

## Methodology

The analysis includes:

- Calculation of daily returns, mean returns and covariance matrix
- Simulation of 50,000 random portfolios
- Identification of:
  - Minimum variance portfolio
  - Maximum return portfolio
  - Maximum Sharpe ratio portfolio
- Construction of the efficient frontier using numerical optimization (SLSQP)

Risk measures:

- Parametric VaR (Normal distribution)
- Historical VaR
- Monte Carlo VaR
- CVaR (Expected Shortfall)
- Non-diversified VaR for comparison

## Results

- Minimum variance portfolio achieved low volatility with diversified allocation
- Maximum return portfolio concentrated in a single asset, illustrating the risk-return tradeoff
- Maximum Sharpe portfolio balanced risk and return with concentration in a few key assets
- VaR estimates ranged around R$ 10k–11k depending on the method
- CVaR indicated more severe average losses in tail scenarios
- Diversification reduced portfolio risk substantially compared to a non-diversified allocation

## Key Insight

The project shows how diversification and covariance structure can materially reduce market risk, reinforcing the importance of portfolio construction in risk management.

## Tools

- Python
- numpy
- pandas
- matplotlib
- scipy
- yfinance

## Notes

This project was developed as part of a quantitative market risk study, combining financial theory with computational methods.