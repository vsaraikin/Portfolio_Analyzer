import matplotlib.pyplot as plt
import yfinance as yf
import plotter
import metrics
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

weights = [0.3, 0.1, 0.05, 0.05, 0.1, 0.15, 0.15]
tickers = ['QQQ', 'AMZN', 'NVDA', 'BTC-USD', 'AAPL', 'YNDX', 'EPAM']

assets = yf.download(tickers, start='2020-01-01')['Adj Close'].dropna()
portfolio = pd.Series(metrics.portfolio_return_series(assets, weights))
portfolio.name = 'Portfolio'
benchmark = pd.Series(np.log(metrics.benchmark(assets) / metrics.benchmark(assets).shift(1)).dropna().cumsum())
portfolio.index = assets.index[1:]
data = pd.concat([benchmark, portfolio], axis=1).dropna()

plotter.cumulative_returns_plot(data)
plotter.distribution(assets)
plotter.drawndowns(assets, weights)
plotter.effecient_frontier(assets, num_ports=1000)
plotter.heatmap(assets)
plt.show()
print(metrics.get_all_table(assets, weights))