""" 
Functions for main file
"""
import numpy as np

def portfolio_return(weights, returns):
    """ Use weights and returns matrix to calculate annualized portfolio returns"""
    return np.sum(np.mean(returns, axis=1) * weights) * 252

def portfolio_volatility(weights, returns):
    """ Use weights and returns to calculate annualized portfolio volatility """
    return np.sqrt(np.dot(weights.T, np.dot(np.cov(returns), weights))) * np.sqrt(252)

def sharpe_target(weights, *args):
    """ Generate the Sharpe Ratio of a Portfolio """
    # get the asset's returns
    returns = args[0]
    return - portfolio_return(weights, returns) / portfolio_volatility(weights, returns)

def pit_cov_matrix(returns_df):
    """ Calculate Cov Matrix with at least 1 year of data """
    cov_matrix = {}
    for date in returns_df[252:].index:
        cov_matrix[date] = returns_df.loc[:date].cov() * 252 # Annualized Covariance Matrix
    return cov_matrix
