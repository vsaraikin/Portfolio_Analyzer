import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import metrics

sns.set_theme()
sns.set_context("paper")


def effecient_frontier(assets, risk_free=0.015, num_ports=500):
    sns.set(rc={'figure.figsize': (24, 8)})
    all_weights = np.zeros((num_ports, len(assets.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for x in range(num_ports):
        # Weights
        weights = np.array(np.random.random(assets.shape[1]))
        weights = weights / np.sum(weights)

        all_weights[x, :] = weights
        ret_arr[x] = np.sum((metrics.log_returns(assets).mean() * weights * 252))
        vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(metrics.log_returns(assets).cov() * 252, weights)))
        sharpe_arr[x] = ret_arr[x] - risk_free / vol_arr[x]

    return sns.scatterplot(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis').set(title='Effecient frontier', xlabel='Volatility', ylabel='Return')


# Cumulative returns vs S&P
def cumulative_returns_plot(data):
    sns.set(rc={'figure.figsize': (24, 8)})
    return sns.lineplot(data=data+1, legend=True).set(title='Cumulative returns\nS&P500 vs Portflio')


# Drawdowns
def drawndowns(assets, weights):
    sns.set(rc={'figure.figsize': (24, 8)})
    return sns.lineplot(data=metrics.drawdown(assets, weights)).set(title='Historical drawdowns')

# Distribution
def distribution(assets):
    log_returns = metrics.log_returns(assets)

    log_returns.hist(bins=30, figsize=(15, 15))
    plt.style.use('seaborn')
    return plt.show()

# Correlation Heatmap
def heatmap(asset):
    sns.set(rc={'figure.figsize': (12, 12)})
    log_returns = metrics.log_returns(asset)
    return sns.heatmap(log_returns.corr(), annot=True, xticklabels=log_returns.columns.values, yticklabels=log_returns.columns.values)

plt.show()
