
# import modules
import pandas as pd
import pandas_datareader as pdr
import numpy as np
from sklearn.decomposition import PCA
import random

def daily_return(start: str, end: str, tickers: list, dropna=True):

    '''
    Description
        This function takes in a starting date, ending date, and a list of tickers.
        It then returns a dictionary containing the closing prices of the tickers in the
        given time horizon.

    start, end: String form of date MM-DD-YYYY
    tickers: List of stock tickers in string.
    '''

    # dictionary for storing
    historical_quotes = pd.DataFrame()

    # import data from yahoo
    data = pdr.get_data_yahoo(tickers,
                                    start,
                                    end)
    raw_length = len(data)

    # dropna values
    if dropna:
        data = data.dropna()
        print("Dropped", (raw_length - len(data)), "rows")

    # populate return dictionary
    for ticker in tickers:

        # close price
        close = data["Adj Close"][ticker]

        # returns
        daily_return = close.pct_change()
        daily_return = daily_return.dropna()

        historical_quotes[ticker] = daily_return


    return historical_quotes


def daily_close_price(start: str, end: str, tickers: list, dropna=True):

    '''
    Description
        This function takes in a starting date, ending date, and a list of tickers.
        It then returns a dictionary containing the closing prices of the tickers in the
        given time horizon.

    start, end: String form of date MM-DD-YYYY
    tickers: List of stock tickers in string.
    '''

    # dictionary for storing
    historical_quotes = pd.DataFrame()

    # import data from yahoo
    data = pdr.get_data_yahoo(tickers,
                                    start,
                                    end)
    raw_length = len(data)

    # dropna values
    if dropna:
        data = data.dropna()
        print("Dropped", (raw_length - len(data)), "rows")

    # populate return dictionary
    for ticker in tickers:

        historical_quotes[ticker] = data["Adj Close"][ticker]

    return historical_quotes


def equal_weighted_portfolio(returns: np.ndarray):

    '''
    Description
        Takes in array where rows are returns per date
        and columns is stocks. Then outputs and equal 
        weighted portfolio.

    Parameters
        returns: Numpy array

    Output
        Array of equal weighted returns.
    '''
    if len(returns[0]) > 1:
        weigth = 1 / len(returns[0])
        eq_return = np.sum(returns * weigth, axis=1)
    else:
        weigth = 1 / len(returns)
        eq_return = np.sum(returns * weigth)

    return eq_return


def pc_weights(returns: np.ndarray, num_components=False):

    if num_components==False:
        n = len(returns[0])
    else:
        n = num_components 

    # principal components
    pc_portfolios = PCA(n_components=n)

    # fitting
    pcs = pc_portfolios.fit_transform(returns)

    # components
    components = pc_portfolios.components_

    # create matrix of weights per PC
    weights = np.empty([len(components), n])

    # store weights for each PC
    idx = 0
    for pc in components:
        pc_weight = np.array(pc / sum(pc).T)
        weights[idx] = pc_weight
        idx += 1

    return weights

def get_pc_returns(returns: np.ndarray):

    weights = pc_weights(returns)

    # array for storing returns
    pc_returns = np.empty([len(returns[0]), len(returns)])

    idx = 0
    for pc in range(0, len(returns[0])):

        inst_pc_returns = np.matmul(returns, weights[pc])
        pc_returns[idx] = inst_pc_returns

        idx += 1

    return pc_returns.T


def sp500_monthly_returns(start, end):

    SP500 = pdr.DataReader(['^GSPC'], 'yahoo', start, end)

    df = pd.DataFrame({"close": SP500["Close"]["^GSPC"],
                  "Date": SP500["Close"]["^GSPC"].index})

    df2 = df.resample('M', on='Date').mean()
    df2 = df2.reset_index()
    df2['Date'] = df2['Date'].apply(lambda x: x.strftime('%Y-%m'))
    df2.set_index('Date', inplace=True)
    
    sp500_returns = df2["close"].pct_change()
    sp500_returns = sp500_returns.dropna()

    return sp500_returns

def mean_abs_error(observed, true):
    
    abs_error = np.abs(observed - true)
    mabs = np.mean(abs_error)
    
    return mabs

def simulate_asset_returns(mu: float, sigma: float, obs: int, n: int, seed=False):

    if seed!=False:
        np.random.seed(seed)

    ret = np.random.normal(mu, sigma, size=(obs, n))

    return ret



def correlated_returns_with_shocks(mu, sigma, nObs, size, sigmaF, sLength):
    
    # original uncorrelated assets
    orgSize = int(0.5 * size) # the number of original assets ~ half size of total number of assets
    x = np.random.normal(mu, sigma, size=(nObs, orgSize))

    # selecting random assets for correlation
    corrAssets = [random.randint(0, orgSize-1) for i in range(orgSize)]

    # creating new returns which are correlated with corrAssets
    y = x[:, corrAssets] + np.random.normal(0, sigma*sigmaF, size = (nObs, len(corrAssets)))

    # appending correlated returns to original returns
    x = np.append(x, y, axis=1)

    # random common shocks
    point = np.random.randint(sLength, nObs-1, size=2)
    x[np.ix_(point, [corrAssets[0], orgSize])] = np.array([[-.5, -.5], [2,2]])
    
    # random specific shock
    point = np.random.randint(sLength, nObs-1,size=2)
    x[point, corrAssets[-1]] = np.array([-.5,2])

    return x
