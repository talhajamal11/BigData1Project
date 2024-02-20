import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandasgui as pdg

from sklearn import linear_model
from sklearn.datasets import load_diabetes, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

import statsmodels.api as sm
from tqdm.notebook import tqdm

stock_returns = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Stock Returns and Characteristics Data\Returns_Data.csv")
stock_chars = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Stock Returns and Characteristics Data\Stock_Characteristics_Data.csv")
"""print(stock_returns)
print(stock_chars)"""

stock_returns["date"] = pd.to_datetime(stock_returns["date"], format = "%Y-%m-%d")
stock_returns = stock_returns.sort_values(by = "date")
stock_returns = stock_returns.rename(columns = {"date": "datadate"})
stock_chars["datadate"] = pd.to_datetime(stock_chars["datadate"], format = "%Y-%m-%d")
stock_chars = stock_chars.sort_values(by = "datadate")
"""print(stock_returns.dtypes)
print(stock_chars.dtypes)"""

stock_merged = pd.merge_asof(stock_returns, stock_chars, on = "datadate", by = "ticker", direction = "forward")
print(stock_merged.isna().sum())

FF_5F = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\FF_5_factor_daily.CSV")
FF_MOM = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\FF_Momentum_Factor_daily.CSV")
FF_STR = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\FF_ST_Reversal_Factor_daily.csv")
FF_LTR = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\FF_LT_Reversal_Factor_daily.csv")
FF_IP = pd.read_csv("E:\秘密基地\看一听\硕士研究生相关\Electives\Big Data in Finance I\Course Project\FF_Industry_Portfolios_Daily.csv", nrows=25650)

FF_5F["datadate"] = pd.to_datetime(FF_5F["datadate"], format="%Y%m%d")
FF_MOM["datadate"] = pd.to_datetime(FF_MOM["datadate"], format="%Y%m%d")
FF_STR["datadate"] = pd.to_datetime(FF_STR["datadate"], format="%Y%m%d")
FF_LTR["datadate"] = pd.to_datetime(FF_LTR["datadate"], format="%Y%m%d")
FF_IP["datadate"] = pd.to_datetime(FF_IP["datadate"], format="%Y%m%d")

FF_5F = FF_5F.sort_values(by="datadate")
FF_MOM = FF_MOM.sort_values(by="datadate")
FF_STR = FF_STR.sort_values(by="datadate")
FF_LTR = FF_LTR.sort_values(by="datadate")
FF_IP = FF_IP.sort_values(by="datadate")

merged_1 = pd.merge(FF_5F, FF_MOM, on="datadate", how="inner")
merged_2 = pd.merge(merged_1, FF_STR, on="datadate", how="inner")
merged_3 = pd.merge(merged_2, FF_LTR, on="datadate", how="inner")
X = pd.merge(merged_3, FF_IP, on="datadate", how="inner")
X = X.sort_values(by="datadate")
X.loc[:, X.columns != "datadate"] /= 100
X.loc[:, ~X.columns.isin(["datadate", "Mkt-RF", "RF"])] = X.loc[:, ~X.columns.isin(["datadate", "Mkt-RF", "RF"])].sub(X["RF"], axis=0)

y = stock_returns.iloc[:, [0, 1, 4]]
X = X[(X["datadate"] >= y["datadate"].min()) & (X["datadate"] <= y["datadate"].max())]
X = X.set_index("datadate")
y = y.set_index("datadate")
y["RET"] -= X["RF"]
X = X.reset_index()
y = y.reset_index()

missing_values = X.isin([-99.99, -999]).any().any()
print(missing_values)

"""print(X)
print(y)"""

X_adjusted = pd.merge(y, X, on="datadate", how="left")
print(X_adjusted)

X = X_adjusted.drop(["datadate", "RET", "ticker"], axis=1)
y = y.drop(["datadate", "ticker"], axis=1)
print(X)
#print(y)

#Principal Component Analysis (PCA) and standardisation
X_pca = pd.DataFrame(PCA().fit_transform(X), columns=X.columns)
X_pca_std = pd.DataFrame(StandardScaler().fit_transform(X_pca), columns=X_pca.columns)
print(X_pca_std.head())


#Standardisation (only)
"""X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
print(X_std)"""

#Training dataset and testing dataset
train_size = int(len(X) * 0.8)
X_train, X_test = X_pca_std.iloc[:train_size, :], X_pca_std.iloc[train_size:, :]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:, :]
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
#print(X_train)
#print(y_test)

#Lasso regression with time-series cross-validation
lasso_cv = LassoCV(cv=TimeSeriesSplit(n_splits=5)).fit(X_train, y_train)
print(lasso_cv.alpha_)
print(lasso_cv.coef_)

#Cross-sectional expected returns
er_cs = lasso_cv.predict(X_test)
mse_cs = mean_squared_error(y_test, er_cs)
rmse_cs = root_mean_squared_error(y_test, er_cs)
r2_cs = r2_score(y_test, er_cs)
print(er_cs)
print(mse_cs)
print(rmse_cs)
print(r2_cs)


"""n_alphas = 2000
alphas = np.logspace(4.5, -5, n_alphas)
alphas = alphas.tolist()"""

"""def lasso_reg(X, y, alphas):
    lasso_models = []
    for a in alphas:
        lasso = linear_model.Lasso(alpha = a).fit(X, y)
        lasso_models.append(lasso)
    return lasso_models"""

"""def ridge_reg(X, y, alphas):
    ridge_models = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha = a).fit(X, y)
        ridge_models.append(ridge)
    return ridge_models"""


#pdg.show(stock_returns)
#pdg.show(stock_chars)
#pdg.show(stock_merged)

