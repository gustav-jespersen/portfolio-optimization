import numpy as np
from sklearn.covariance import LedoitWolf
from typing import Union, List, Tuple


def check_positive_semi_definite(matrix: np.ndarray): 
    
    """
    Checks that a symmetric square matrix is positive semi-deinite. 
    
    See https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    
    Parameters
    ----------
    
    matrix: 
        Symmetric, Square matrix. 
        
    
    Returns
    -------
    bool
        Boolean indicating whether the matrix is positive semi-definite
        
    """
    
    try:
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


class mean_variance:
    
    def __init__(self, returns, variance, cov, raw_returns):
        self.returns = returns
        self.variance = variance
        self.cov = cov
        self.raw_returns = raw_returns

    def inv_cov_mat(self):
    
        '''
        calculate inverse of matrix
        '''
        cov_mat = self.cov
        
        # check positive semi definite
        check = check_positive_semi_definite(cov_mat)

        if check == True:
            self.inv_covar = np.linalg.inv(cov_mat)
        
        else:
            print("Non-invertible")
        
        return self.inv_covar
        
    def port_return_equal(self):
        '''
        caluclate the expected return of the equal weighted portfolio
        '''
        n = len(self.returns)
        self.weighted_returns = np.sum(self.returns * (1 / n))
        return self.weighted_returns

    def port_var_equal(self, equal_weighted=True):
        '''
        calculates variance of portfolio
        '''
        cov_mat = self.cov

        if equal_weighted:
            n = len(cov_mat)
            weights = np.ones(n) / n
            self.port_var = np.dot(weights.T,np.dot(cov_mat, weights))
        else:
            self.port_var =  "not equal weighted"
        
        return self.port_var


    def max_slope_weights(self):
        '''
        returns the weight vector of the meximum slope portfolio
        '''
        # inverse covariance matrix
        inv_covar = self.inv_cov_mat()

        n = len(inv_covar)

        # auxillairy constants
        b = np.dot(np.ones(n), np.matmul(self.returns, inv_covar.T))

        max_slope_weights = np.dot((1 / b), np.matmul(inv_covar.T, self.returns))

        return max_slope_weights


    def max_slope_return(self):

        # inverse covariance matrix
        inv_covar = self.inv_cov_mat()

        n = len(inv_covar)

        # auxiliary constants
        a = np.dot(self.returns, np.matmul(inv_covar, self.returns))
        b = np.dot(np.ones(n), np.matmul(self.returns, inv_covar.T))

        max_slope_return = a / b

        return max_slope_return


    def max_slope_variance(self):

        # inverse covariance matrix
        inv_covar = self.inv_cov_mat()

        n = len(inv_covar)

        # auxiliary constants
        a = np.dot(self.returns, np.matmul(inv_covar, self.returns))
        b = np.dot(np.ones(n), np.matmul(self.returns, inv_covar.T))

        max_slope_variance = a / b**2

        return max_slope_variance


    def min_var_weights(self):
    
        '''
        calculate the global minimum variance portfolio
        '''

        # inverse covariance matrix
        inv_covar = self.inv_cov_mat()

        n = len(inv_covar)

        # auxilliary constant
        c = np.dot(np.ones(n), np.matmul(np.ones(n), inv_covar.T))

        # minimum variance portfolio
        self.min_var = 1 / c * np.matmul(inv_covar, np.ones(n))

        return self.min_var
    
    def min_var_return(self):
        '''
        calculates expected return of the minimum variance portfolio
        '''
        # inverse covariance
        inv_covar = self.inv_cov_mat()
        
        # constants
        n = len(self.returns)

        # auxiliary constants
        b = np.dot(np.ones(n), np.matmul(self.returns, inv_covar.T))
        c = np.dot(np.ones(n), np.matmul(np.ones(n), inv_covar.T))
        
        # minimum vairance portfolio expected return
        self.exp_return = b / c

        return self.exp_return

    def min_var_variance(self):
        '''
        calculates the variance of the global minimum variance portfolio
        '''
        # inverse covariance
        inv_covar = self.inv_cov_mat()
        
        # constants
        n = len(self.returns)

        # auxiliary constants
        c = np.dot(np.ones(n), np.matmul(np.ones(n), inv_covar.T))
        
        # minimum vairance portfolio expected variance
        self.exp_variance = 1 / c

        return self.exp_variance


    def tan_port_weights(self, rf=0.0001):
        '''
        weights of the tangency portfolio
        '''

        #constants
        n = len(self.returns)

        # inverse covariance
        inv_covar = self.inv_cov_mat()

        # auxisiliary constants
        b = np.dot(np.ones(n), np.matmul(self.returns, inv_covar.T))
        c = np.dot(np.ones(n), np.matmul(np.ones(n), inv_covar.T))

        # calculate weights
        nominator = np.matmul(inv_covar, self.returns - rf)
        denominator = b - c * rf

        tan_weights = nominator / denominator

        return tan_weights


    def tan_port_return(self, rf=0.0001):
        '''
        expected return of the tangency portfolio
        '''

        # constants
        n = len(self.returns)

        # inverse covariance matrix
        inv_covar = self.inv_cov_mat()

        # auxiliary constants
        a = np.dot(self.returns, np.matmul(inv_covar, self.returns))
        b = np.dot(np.ones(n), np.matmul(self.returns, inv_covar.T))
        c = np.dot(np.ones(n), np.matmul(np.ones(n), inv_covar.T))

        tan_port_return = (a - b * rf) / (b - c * rf)

        return tan_port_return


    def tan_port_variance(self, rf=0.0001):
        '''
        variance of the expected return of tangency portfolio
        '''

        # constants
        n = len(self.returns)

        # inverse covariance matrix
        inv_covar = self.inv_cov_mat()

        # auxiliary constants
        a = np.dot(self.returns, np.matmul(inv_covar, self.returns))
        b = np.dot(np.ones(n), np.matmul(self.returns, inv_covar.T))
        c = np.dot(np.ones(n), np.matmul(np.ones(n), inv_covar.T))


        tan_port_variance = (a - 2 * b * rf + c * rf**2) / (b- c * rf)**2

        return tan_port_variance

    def tan_port_sharpe(self, rf=0.0001):
        '''
        sharpe ratio of tangency portfolio
        '''
        ret = self.tan_port_return()
        var = self.tan_port_variance()

        sr = (ret - rf) / np.sqrt(var)

        return sr


def efficient_frontier(ret, var, cov_mat, raw_returns, rf=0.0001):
    '''
    returns dictionary with mean and std of efficient frontier
    '''

    # portfolios
    opt = mean_variance(ret, var, cov_mat, raw_returns)

    min_var_ret = opt.min_var_return()
    min_var_var = opt.min_var_variance()

    max_slp_ret = opt.max_slope_return()
    max_slp_var = opt.max_slope_variance()

    tan_port_ret = opt.tan_port_return(rf)
    tan_port_var = opt.tan_port_variance(rf)

    # weights of portfoios
    weight_A = np.linspace(-5,5,1000)
    weight_B = 1 - weight_A

    # return vector of effecient frontier
    returns = weight_A * min_var_ret + weight_B * max_slp_ret

    # variance vector of efficient frontier
    variance = weight_A**2 * min_var_var + weight_B**2 * max_slp_var + 2 * weight_A * weight_B * min_var_var
    std = np.sqrt(variance)

    # return of global efficient frontier
    returns_all = weight_A * tan_port_ret + weight_B * rf

    # variance vector of global efficient frontier
    variance_all = weight_A**2 * tan_port_var
    std_all = np.sqrt(variance_all)

    # storing values
    results = {"returns": returns,
               "std": std,
               "returns_global": returns_all,
               "std_global": std_all}

    return results

