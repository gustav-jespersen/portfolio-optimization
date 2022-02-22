import numpy as np
from sklearn.covariance import LedoitWolf

def check_positive_semi_definite(matrix: np.ndarray): 
    
    try:
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def inv_covariance(return_series: np.ndarray, ledoit_wolf=False):
    
    covariance = np.cov(return_series.T)
    check = check_positive_semi_definite(covariance)

    if (check == True) & (ledoit_wolf == False):
        inv_cov = np.linalg.inv(covariance)
    elif (check == True) & (ledoit_wolf == True):
        cov = LedoitWolf().fit(return_series)
        inv_cov = np.linalg.inv(cov.covariance_)
    else:
        inv_cov = "covariance matrix non-invertible"

    return inv_cov

def auxiliary_constants(returns: np.ndarray, return_series: np.ndarray, ledoit_wolf=False):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

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

def minimum_varaince_weights(returns: np.ndarray, return_series: np.ndarray, ledoit_wolf=False):
    
    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf)
    c = aux["c"]

    # minimum variance portfolio weights
    min_var_weights = 1 / c * np.matmul(inv_cov, np.ones(n))

    return min_var_weights

def minimum_variance_return(returns: np.ndarray, return_series: np.ndarray, ledoit_wolf=False):

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf)
    b = aux["b"]
    c = aux["c"]

    # minimum variance portfolio expected return
    min_var_ret = b / c
    
    return min_var_ret

def minimum_variance_varaince(returns: np.ndarray, return_series: np.ndarray, ledoit_wolf=False):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf)
    c = aux["c"]

    # minimum variance portfolio expected varaince
    min_var_variance = 1 / c

    return min_var_variance

def maximum_slope_weights(returns: np.ndarray, return_series: np.ndarray, ledoit_wolf=False):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf)
    b = aux["b"]

    # max slope portfolio weights
    max_slp_weights = np.dot((1 / b), np.matmul(inv_cov.T, returns))

    return max_slp_weights

def maximum_slope_return(returns: np.ndarray, return_series: np.ndarray, ledoit_wolf=False):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf)
    a = aux["a"]
    b = aux["b"]

    max_slp_return = a / b

    return max_slp_return

def maximum_slope_variance(returns: np.ndarray, return_series: np.ndarray, ledoit_wolf=False):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf)
    a = aux["a"]
    b = aux["b"]

    max_slp_variance = a / b**2

    return max_slp_variance

def tangency_portfolio_weights(returns: np.ndarray, return_series: np.ndarray, rf=0, ledoit_wolf=False):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf)
    b = aux["b"]
    c = aux["c"]

    # calculate weights
    nominator = np.matmul(inv_cov, returns - rf)
    denominator = b - c * rf

    tan_weights = nominator / denominator

    return tan_weights

def tangency_portfolio_return(returns: np.ndarray, return_series: np.ndarray, rf=0, ledoit_wolf=False):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf)
    a = aux["a"]
    b = aux["b"]
    c = aux["c"]

    tan_port_return = (a - b * rf) / (b - c * rf)

    return tan_port_return

def tangency_portfolio_variance(returns: np.ndarray, return_series: np.ndarray, rf=0, ledoit_wolf=False):

    # number of assets
    n = len(returns)

    # inverse covariance
    inv_cov = inv_covariance(return_series, ledoit_wolf)

    # auxiliary constants
    aux = auxiliary_constants(returns, return_series, ledoit_wolf)
    a = aux["a"]
    b = aux["b"]
    c = aux["c"]

    tan_port_variance = (a - 2 * b * rf + c * rf**2) / (b- c * rf)**2

    return tan_port_variance

def efficient_frontier(returns: np.ndarray, return_series: np.ndarray, rf=0, ledoit_wolf=False):
    
    # minimum variance portfolio
    min_ret = minimum_variance_return(returns, return_series, ledoit_wolf)
    min_var = minimum_variance_varaince(returns, return_series, ledoit_wolf)
    
    # maximum slope portfolio
    max_ret = maximum_slope_return(returns, return_series, ledoit_wolf)
    max_var = maximum_slope_variance(returns, return_series, ledoit_wolf)
    
    # tangency portfolio
    tan_ret = tangency_portfolio_return(returns, return_series, rf, ledoit_wolf)
    tan_var = tangency_portfolio_variance(returns, return_series, rf, ledoit_wolf)
    
    # weights of portfoios
    weight_A = np.linspace(-5,5,1000)
    weight_B = 1 - weight_A

    # return vector of effecient frontier
    returns = weight_A * min_ret + weight_B * max_ret
    
    # variance vector of efficient frontier
    variance = weight_A**2 * min_var + weight_B**2 * max_var + 2 * weight_A * weight_B * min_var
    std = np.sqrt(variance)
    
    # return of global efficient frontier
    returns_all = weight_A * tan_ret + weight_B * rf
    
    # variance vector of global efficient frontier
    variance_all = weight_A**2 * tan_var
    std_all = np.sqrt(variance_all)
    
    # storing values
    results = {"returns": returns,
               "std": std,
               "returns_global": returns_all,
               "std_global": std_all}

    return results