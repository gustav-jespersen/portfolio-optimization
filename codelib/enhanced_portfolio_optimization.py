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
    corr = np.corrcoef(return_series.T) * (1 - shrinkage)

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
