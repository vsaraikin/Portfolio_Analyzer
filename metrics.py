import pandas as pd
import numpy as np
import yfinance as yf


def simple_returns(assets):
    s_returns = (assets / assets.shift(1)) - 1
    return s_returns


def log_returns(assets):
    log_return = np.log(assets / assets.shift(1)).dropna()
    return log_return


def portfolio_return(assets, weights):
    """
    Return a sum of portfolio weighted log returns
    """

    returns = log_returns(assets).sum()
    port_ret = np.dot(returns, weights)
    return port_ret

def portfolio_return_series(assets, weights):
    """
    Return of portfolio weighted log returns
    """

    returns = log_returns(assets)
    port_ret = np.matmul(returns, weights)
    return port_ret.cumsum()


def portfolio_volatility(assets, weights):
    """
    Returns weighted portfolio volatility (without annual adjustment)
    """
    trading_days = assets.shape[0]
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(log_returns(assets).cov() * trading_days, weights)))
    return portfolio_vol


def covariance(assets):
    return log_returns(assets).cov()


def correlation(assets):
    return log_returns(assets).corr()


def volatility_annualized(assets, weights):
    """
    Returns annualized volatility with annual adjustment
    """
    returns = portfolio_return_series(assets, weights)
    period = assets.shape[1]
    return returns.std() * (period ** 0.5)


def returns_annualized(assets, weights):
    returns = pow(portfolio_return(assets, weights) + 1, 1/(assets.shape[0]/252)) - 1
    return returns


def sharpe_ratio(assets, weights, rf=0.015):
    """
    Returns sharpe ratio
    """

    sharpe = (portfolio_return(assets, weights) - rf) / (portfolio_volatility(assets, weights))
    return sharpe


def drawdown(asset, weights):
    """
    Returns negative outcomes of returns
    """
    returns = log_returns(asset)
    port_ret = np.matmul(returns, weights)
    data = 1000 * (1 + port_ret).cumprod()
    wealth_index = pd.Series(data)
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns


def semideviation(asset, weights):
    returns = portfolio_return_series(asset, weights)
    returns = returns[returns < 0]
    return returns.std(ddof=0)


def var_historic(asset, weights, level=5):
    """
    Returns 5th lowest perntile of returns
    """

    asset = portfolio_return_series(asset, weights)
    if isinstance(asset, pd.DataFrame):
        return asset.aggregate(var_historic, level=level)
    elif isinstance(asset, pd.Series):
        return -np.percentile(asset, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def sharpe_ratio_generator(assets, risk_free=0.015, number_port=5000):
    number_assets = len(assets.columns)

    weights_store = np.zeros((number_port, len(assets.columns)))
    returns_store = np.zeros(number_port)
    vol_store = np.zeros(number_port)
    sharpe_store = np.zeros(number_port)

    for x in range(number_port):
        # Weights
        weights = np.array(np.random.random(number_assets))
        weights /= np.sum(weights)
        # Save Weights
        weights_store[x, :] = weights

        # Save Expected Returns
        returns_store[x] = np.sum(weights * log_returns(assets).mean()) * 252

        # Save Expected Volatility
        vol_store[x] = np.sqrt(np.dot(weights.T, np.dot(log_returns(assets).cov() * 252, weights)))

        # Save Sharpe Ratio
        sharpe_store[x] = (returns_store[x] - risk_free) / vol_store[x]
    data = pd.DataFrame(data = [sharpe_store, vol_store, weights_store], index=['Sharpe Ratios', 'Volatility', 'Weights'])
    data = data.transpose()
    data = data.sort_values(by='Sharpe Ratios', ascending=False).head(10)
    data = data[data['Volatility'] == data['Volatility'].min()]
    print('Maximizing returns and minimizing volatility')
    return data



def sortino_ratio(assets, weights, risk_free=0.15):
    returns = portfolio_return(assets, weights)
    std = semideviation(assets, weights)
    return (returns - risk_free) / std


def benchmark(assets, ticker='^GSPC'):
    """
    Return a series of prices of benchmark
    S&P 500 as default
    """
    bench = yf.download(ticker, start=assets.index[0])['Adj Close']
    bench.name = 'S&P500'
    return bench


def portfolio_beta(assets, weights):
    """
    Returns covariance (portfolio, benchmark) divided by variance of benchmark

    If beta > 1, then portfolio is more volatile than market
    If beta < 1, then portfolio is less volatile than market

    """
    if isinstance(weights, list):
        returns = np.matmul(log_returns(assets), weights)
    else:
        returns = log_returns(assets)

    benchmark_returns = log_returns(benchmark(assets)).dropna()
    covariance = benchmark_returns.cov(returns)
    beta = covariance / benchmark_returns.var()
    return beta


def portfolio_alpha(assets, weights, risk_free=0.015):
    """
    The result shows that the investment in this example outperformed the benchmark index by XXX%
    """
    return_of_portfolio = portfolio_return(assets, weights)
    alpha = return_of_portfolio.sum() - risk_free - portfolio_beta(assets, weights) * (
        np.log(benchmark(assets)/benchmark(assets).shift(1)).dropna().sum() - risk_free)
    return alpha


def treynor_ratio(assets, weights, risk_free=0.015):
    """
    A higher Treynor ratio result means a portfolio is a more suitable investment.
    """
    return (portfolio_return(assets, weights) - risk_free) / portfolio_beta(assets, weights)


def jensen_ratio(assets, weights, risk_free=0.015):
    """
    How much of the portfolio's rate of return is attributable to the manager's
    ability to deliver above-average returns, adjusted for market risk.
    The higher the ratio, the better the risk-adjusted returns.
    """
    beta = portfolio_beta(assets, weights)
    return portfolio_return(assets, weights) - beta * (portfolio_return(assets, weights) - risk_free)


def get_all_table(assets, weights, risk_free=0.015):
    """
    Returns a dataframe with all the data
    """
    dictionary = {
        "Total Return": f'{100*portfolio_return(assets, weights).round(3)}%',
        "Annualized Return": f'{100*returns_annualized(assets, weights).round(3)}%',
        "Annualized Volatility": volatility_annualized(assets, weights).round(3),
        "Alpha": portfolio_alpha(assets, weights, risk_free).round(3),
        "Beta": portfolio_beta(assets, weights).round(3),
        "Historical VaR": var_historic(assets, weights).round(3),
        "Sharpe Ratio (highest possible)": sharpe_ratio(assets, weights, risk_free).round(3),
        "Sortino Ratio": sortino_ratio(assets, weights, risk_free).round(3),
        "Treynor Ratio": treynor_ratio(assets, weights, risk_free).round(3)
    }
    return pd.Series(dictionary)
