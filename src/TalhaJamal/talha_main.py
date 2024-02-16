""" Mean Variance Portfolio Optimizer"""

import os
import pandas as pd
import numpy as np
import functions as f

# Import data from files
os.chdir('/Users/talhajamal/Desktop/Code/BigData1Project')
data = pd.read_csv("data/Returns_Data.csv")
data['date'] = pd.to_datetime(data['date'], dayfirst=False)
characteristics = pd.read_csv("data/Stock_Characteristics_Data.csv")
dictionary = pd.read_excel("data/StockDataDictionary.xlsx")

# Create new dataframes
prices = data.pivot(index='date', columns='ticker', values='PRC')
volume = data.pivot(index='date', columns='ticker', values='VOL')
returns = data.pivot(index='date', columns='ticker', values='RET')
#returns = returns * 100 # Scale returns to percentage
# Summary of Returns
returns_summary = returns.describe()
shares_outstanding = data.pivot(index='date', columns='ticker', values='SHROUT')
value_weighted_returns = data.pivot(index='date', columns='ticker', values='vwretd')
equal_weighted_returns = data.pivot(index='date', columns='ticker', values='ewretd')
tickers = prices.columns # List of Tickers

meanReturns = returns.mean()
covMatrix = returns.cov()

equalWeightedPortfolioRet, equalWeightedPortfolioVol, equalWeightedPortfolioSR = f.eqWeightPortfolioPerformance(meanReturns, covMatrix, returns)

maxSRPortfolioRet, maxSRPortfolioVol, maxSRPortfolioSR = f.maxSRPortfolioPerformance(meanReturns, covMatrix)

minVarPortfolioRet, minVarPortfolioVol, minVarPortfolioSR = f.minVariancePortfolioPerformance(meanReturns, covMatrix)
