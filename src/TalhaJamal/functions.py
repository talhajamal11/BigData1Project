""" 
Functions for main file
"""
import os
import pandas as pd
import numpy as np
import scipy.optimize as sc 

def portfolioPerformance(weights, meanReturns, covMatrix):
    """ Calculate Portfolio Performance"""
    annualizedReturns = np.sum(meanReturns*weights)*252
    annualizedStd = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(252)
    return annualizedReturns*100, annualizedStd*100

def negativeSR(weights, meanReturns, covMatrix,riskFreeRate = 0):
    """ Calculate Negative Sharpe so that Optimizer works on it and minimizes it -> essentially maximising it"""
    annualizedRet, annualizedStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return - ((annualizedRet - riskFreeRate)/annualizedStd)

def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintSet = (0, 1)):
    """ Minimize the negative Sharpe Ratio -> Maximize it"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type':'eq','fun': lambda x: np.sum(x) - 1 }) # Weights must sum up to 1
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets)) # For every asset have this bound
    result = sc.minimize(negativeSR, x0=numAssets*[1./numAssets], args=args, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioVariance(weights, meanReturns, covMatrix):
    """ Return Portfolio Variance """
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def minimizeVariance(meanReturns, covMatrix, riskFreeRate = 0, constraintSet = (0, 1)):
    """ Minimize the portfolio variance by altering the weights/allocations of assets in the portfolio"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq','fun': lambda x: np.sum(x) - 1 }) # Weights must sum up to 1
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets)) # For every asset have this bound
    result = sc.minimize(portfolioVariance, x0=numAssets*[1./numAssets], args=args, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def eqWeightPortfolioPerformance(meanReturns, covMatrix, returns):
    """ Return the Performance of an Equally Weighted Portfolio"""
    weights = np.array([1/100 for _ in returns])
    equalWeightedPortfolioReturns, equalWeightedPortfolioVolatility = portfolioPerformance(weights, meanReturns, covMatrix)
    equalWeightedPortfolioSR = (-1) * negativeSR(weights, meanReturns, covMatrix)
    print(f"Equal Weighted Portfolio Returns: {round(equalWeightedPortfolioReturns, 2)} %")
    print(f"Equal Weighted Portfolio Volatility: {round(equalWeightedPortfolioVolatility, 2)} %")
    print(f"Equal Weighted Portfolio SR: {round(equalWeightedPortfolioSR, 2)}")
    return equalWeightedPortfolioReturns, equalWeightedPortfolioVolatility, equalWeightedPortfolioSR
    
def maxSRPortfolioPerformance(meanReturns, covMatrix):
    """ Return the Performance of the Maximum SR Portfolio"""
    maxSRPortfolioPerformance = maxSR(meanReturns, covMatrix)
    maxSRPortfolioWeights = maxSRPortfolioPerformance.x
    maxSRPortfolioReturns, maxSRPortfolioVolatility = portfolioPerformance(maxSRPortfolioWeights, meanReturns, covMatrix)
    maxSRPortfolioSR = maxSRPortfolioPerformance.fun * (-1)
    print(f"Max SR Portfolio Returns: {round(maxSRPortfolioReturns, 2)} %")
    print(f"Max SR Portfolio Volatility: {round(maxSRPortfolioVolatility, 2)} %")
    print(f"Max SR Portfolio SR: {round(maxSRPortfolioSR, 2)}")
    return maxSRPortfolioReturns, maxSRPortfolioVolatility, maxSRPortfolioSR

def minVariancePortfolioPerformance(meanReturns, covMatrix):
    """ Return the Performance of the Minimum Variance Portfolio"""
    minVariancePortfolioPerformance = minimizeVariance(meanReturns, covMatrix)
    minVariancePortfolioWeights = minVariancePortfolioPerformance.x
    minVariancePortfolioReturns, minVariancePortfolioVolatility = portfolioPerformance(minVariancePortfolioWeights, meanReturns, covMatrix)
    minVariancePortfolioSR = (-1) * negativeSR(minVariancePortfolioWeights, meanReturns, covMatrix)
    print(f"Min Variance Portfolio Returns: {round(minVariancePortfolioReturns, 2)} %")
    print(f"Min Variance Portfolio Volatility: {round(minVariancePortfolioVolatility, 2)} %")
    print(f"Min Variance Portfolio SR: {round(minVariancePortfolioSR, 2)}")
    return minVariancePortfolioReturns, minVariancePortfolioVolatility, minVariancePortfolioSR