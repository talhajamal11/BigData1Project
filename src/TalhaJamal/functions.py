""" 
Functions for main file
"""

def pit_cov_matrix(returns_df):
    """ Calculate Cov Matrix with at least 1 year of data """
    cov_matrix = {}
    for date in returns_df[252:].index:
        cov_matrix[date] = returns_df.loc[:date].cov() * 252 # Annualized Covariance Matrix
    return cov_matrix
