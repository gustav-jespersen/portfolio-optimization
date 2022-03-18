import numpy as np
from sklearn.covariance import LedoitWolf

def check_positive_semi_definite(matrix: np.ndarray): 
    
    try:
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def inv_covariance(return_series: np.ndarray, ledoit_wolf=False, shrinkage=0):
    
    variances = np.var(return_series, axis=0)
    corr = np.corrcoef(return_series.T) * (1 - shrinkage) + np.identity(len(return_series[0])) * shrinkage

    covariance = np.outer(variances, variances) * corr
    check = check_positive_semi_definite(covariance)

    if (check == True) & (ledoit_wolf == False):
        inv_cov = np.linalg.inv(covariance)
    elif (check == True) & (ledoit_wolf == True):
        cov = LedoitWolf().fit(return_series)
        inv_cov = np.linalg.inv(cov.covariance_)
    else:
        inv_cov = "covariance matrix non-invertible"
    
    return inv_cov

def auxiliary_constants(returns: np.ndarray, return_series: np.ndarray, ledoit_wolf=False, shrinkage=0):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf, shrinkage)

    # auxiliary constants
    a = np.dot(returns, np.matmul(inv_cov, returns))
    b = np.dot(np.ones(n), np.matmul(returns, inv_cov.T))
    c = np.dot(np.ones(n), np.matmul(np.ones(n), inv_cov.T))
    d = a * c - b**2
    
    # returning auxiliary constnats
    auxiliary_constants = {"a": a,
                           "b": b,
                           "c": c,
                           "d": d}

    return auxiliary_constants


def tangency_portfolio_weights(returns: np.ndarray, return_series: np.ndarray, rf=0, ledoit_wolf=False, shrinkage=0):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf, shrinkage)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf, shrinkage)
    b = aux["b"]
    c = aux["c"]

    # calculate weights
    nominator = np.matmul(inv_cov, returns - rf)
    denominator = b - c * rf

    tan_weights = nominator / denominator

    return tan_weights

def tangency_portfolio_return(returns: np.ndarray, return_series: np.ndarray, rf=0, ledoit_wolf=False, shrinkage=0):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf, shrinkage)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf, shrinkage)
    a = aux["a"]
    b = aux["b"]
    c = aux["c"]

    tan_port_return = (a - b * rf) / (b - c * rf)

    return tan_port_return

def tangency_portfolio_variance(returns: np.ndarray, return_series: np.ndarray, rf=0, ledoit_wolf=False, shrinkage=0):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf, shrinkage)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf, shrinkage)
    a = aux["a"]
    b = aux["b"]
    c = aux["c"]

    tan_port_variance = (a - 2 * b * rf + c * rf**2) / (b- c * rf)**2

    return tan_port_variance

def tangency_portfolio_sharpe_ratio(returns: np.ndarray, return_series: np.ndarray, rf=0, ledoit_wolf=False, shrinkage=0):

    ret = tangency_portfolio_return(returns, return_series, rf, ledoit_wolf, shrinkage)
    std = np.sqrt(tangency_portfolio_variance(returns, return_series, rf, ledoit_wolf, shrinkage))

    return (ret - rf) / std


def portfolio_variance(return_series, weights):
    
    # calculating covariance matrix
    covar = np.cov(return_series.T)
    
    # calculating portfolio variance
    port_var = np.dot(weights.T, np.dot(covar,weights))
    
    return port_var


def epo_realized_sharpe(ret_train: np.ndarray,
                                   ret_test: np.ndarray,
                                   rf = 0.0,
                                   ledoit_wolf = False,
                                   accuracy = 10):
    
    '''
    Parameters
    ---
    ret_train:
        All returns before the ret_train period.
        This is the return serios on which the inputs for MVO will be configured.
    
    ret_test:
        This is the return period proceeding the last period of ret_train.

    accuracy:
        This is the number of steps inbetween 0 and 1 for the shrinkage parameter.
        Default is 1000.


    Returns
    ---
    Returns a list with lenght accuracy containing sharpe ratio per shrinkage parameter.

    '''

    # list for storing sharpe ratios
    sharpe_ratios = []

    # vector of expected mean and variance
    mu_train = np.mean(ret_train, axis=0)

    for shrink in np.linspace(0, 1, accuracy):

        # tangency portfolio weights
        tan_weights = tangency_portfolio_weights(mu_train, ret_train, rf, ledoit_wolf, shrink)

        # realized return
        real_tan_ret = np.sum(tan_weights * ret_test)

        # realized variance
        real_tan_var = portfolio_variance(np.vstack((ret_train[-1], ret_test)), tan_weights)
        real_tan_std = np.sqrt(real_tan_var)

        # realized tangency sharpe ratio
        real_tan_sharpe = (real_tan_ret - rf) / real_tan_std

        sharpe_ratios.append(real_tan_sharpe)

    return sharpe_ratios


def epo_realized_return(ret_train: np.ndarray,
                           ret_test: np.ndarray,
                           rf = 0.0,
                           ledoit_wolf = False,
                           accuracy = 10):
    
    '''
    Parameters
    ---
    ret_train:
        All returns before the ret_train period.
        This is the return serios on which the inputs for MVO will be configured.
    
    ret_test:
        This is the return period proceeding the last period of ret_train.

    accuracy:
        This is the number of steps inbetween 0 and 1 for the shrinkage parameter.
        Default is 1000.


    Returns
    ---
    Returns a list with lenght accuracy containing sharpe ratio per shrinkage parameter.

    '''

    # list for storing sharpe ratios
    real_ret = []

    # vector of expected mean and variance
    mu_train = np.mean(ret_train, axis=0)

    for shrink in np.linspace(0, 1, accuracy):

        # tangency portfolio weights
        tan_weights = tangency_portfolio_weights(mu_train, ret_train, rf, ledoit_wolf, shrink)

        # realized return
        real_tan_ret = np.sum(tan_weights * ret_test)

        # realized variance
        real_tan_var = portfolio_variance(np.vstack((ret_train[-1], ret_test)), tan_weights)
        real_tan_std = np.sqrt(real_tan_var)

        # realized tangency sharpe ratio
        #real_tan_sharpe = (real_tan_ret - rf) / real_tan_std

        real_ret.append(real_tan_ret)

    return real_ret



def epo_realized_volatility(ret_train: np.ndarray,
                           ret_test: np.ndarray,
                           rf = 0.0,
                           ledoit_wolf = False,
                           accuracy = 10):
    
    '''
    Parameters
    ---
    ret_train:
        All returns before the ret_train period.
        This is the return serios on which the inputs for MVO will be configured.
    
    ret_test:
        This is the return period proceeding the last period of ret_train.

    accuracy:
        This is the number of steps inbetween 0 and 1 for the shrinkage parameter.
        Default is 1000.


    Returns
    ---
    Returns a list with lenght accuracy containing sharpe ratio per shrinkage parameter.

    '''

    # list for storing sharpe ratios
    real_vol = []

    # vector of expected mean and variance
    mu_train = np.mean(ret_train, axis=0)

    for shrink in np.linspace(0, 1, accuracy):

        # tangency portfolio weights
        tan_weights = tangency_portfolio_weights(mu_train, ret_train, rf, ledoit_wolf, shrink)

        # realized return
        real_tan_ret = np.sum(tan_weights * ret_test)

        # realized variance
        real_tan_var = portfolio_variance(np.vstack((ret_train[-1], ret_test)), tan_weights)
        real_tan_std = np.sqrt(real_tan_var)

        # realized tangency sharpe ratio
        #real_tan_sharpe = (real_tan_ret - rf) / real_tan_std

        real_vol.append(real_tan_std)

    return real_vol


def optimal_shrinkage_parameters(return_series: np.ndarray,
                                 rf = 0.0,
                                 ledoit_wolf = False,
                                 accuracy = 10,
                                 ratio = 0.5):

    # the number of windows, here set as a fraction of the total length
    num_windows = int(ratio * len(return_series))
    arr = np.empty([num_windows, accuracy])

    # looping through every period after initial return series
    idx = 0
    for window in range((len(return_series)-num_windows), len(return_series)):
        
        # define ret train and ret test
        ret_train = return_series[idx:window]
        ret_test = return_series[window]
        
        # appending sharpe ratios
        arr[idx] = different_shrinkage_parameters(ret_train, ret_test, rf, ledoit_wolf, accuracy=accuracy)
        
        idx += 1

    # selecting optimal shrinkage parameter for each period
    opt_shrink = []

    # looping through every line and selecting max sharpe
    for period in range(0, len(arr)):

        max_sharpe = np.amax(arr[period])

        idx_location_sharpe = np.where(arr==max_sharpe)[1]

        optimal_shrinkage = np.linspace(0,1,accuracy)[idx_location_sharpe]

        opt_shrink.append(optimal_shrinkage[0])

    return np.array(opt_shrink)
