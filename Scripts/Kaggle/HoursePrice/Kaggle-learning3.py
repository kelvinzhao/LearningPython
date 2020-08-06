# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# %% 引入模块

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

plt.style.use(style="ggplot")
sns.set(color_codes=True)
print("引入必要模块，完成!")

# %%  直观观察
# 发现四个变量与目标值关系密切
# OverallQual
# YearBuilt
# TotalBsmtSF
# GrLivArea

train_data = pd.read_csv('train.csv')
figure = plt.figure()
sns.pairplot(x_vars=['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF'], y_vars=[
             'SalePrice'], data=train_data, dropna=True)
plt.show()


# %% 观察变量相关性

corrmat = train_data.corr()
# plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.show()

# saleprice correlation matrix
k = 10  # number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# 分析这10个变量
# GarageCars 和 GarageArea 相似，取 GarageCars
# TotalBsmtSF 和 1stFloor 相关，取 TotalBsmtSF
# ToRmsAbvGrd 和 GrLivArea 相关，取 GrLivArea

# scatterplot，两两相关性分析，慎重执行，费CPU，结果见'7-features-scatter.pdf'
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea',
#         'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(train_data[cols], size=2.5)
# plt.show()

# %% 处理缺失值
# missing data
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# 缺失值超过15%的应该删掉该变量，所以'PoolQC', 'MiscFeature' and 'FireplaceQu' 应该可以被删掉。
# GarageX 系列变量丢失相同的数据，并且 GarageCars 已经表示了这一套变量的含义，所以，其他的这些GarageX变量可以删除。
# BsmtX同理
# MasVnrArea 和 MasVnrType 并不是必须的，相关含义已经可以通过YearBuilt和OverallQual所代表。因此这两个变量也可以删除。
# Electrical 变量只有一个NA值，这条记录删除即可。
# In summary, to handle missing data, we'll delete all the variables with missing data, except the variable 'Electrical'. In 'Electrical' we'll just delete the observation with missing data.
# dealing with missing data
train_data = train_data.drop(
    (missing_data[missing_data['Total'] > 1]).index, 1)
train_data = train_data.drop(
    train_data.loc[train_data['Electrical'].isnull()].index)
# just checking that there's no missing data missing...
train_data.isnull().sum().max()

# %% 处理 OutLiars 异常值
# 单变量分析 Univariate analysis
# The primary concern here is to establish a threshold that defines an observation as an outlier. To do so, we'll standardize the data. In this context, data standardization means converting data values to have mean of 0 and a standard deviation of 1.
# standardizing data
saleprice_scaled = StandardScaler().fit_transform(
    train_data['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# outer range最后两个大于7的断定为异常值，删掉
train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (
    train_data['SalePrice'] < 200000)].index, inplace=True)

# 二元变量分析 bivariate analysis saleprice/grlivarea
plt.scatter(x=train_data['TotalBsmtSF'], y=train_data['SalePrice'])
plt.ylim(0, 800000)

# 另一种画图方式：用pandas里的plot
var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# %% 深入了解 SalePrice, Who is 'SalePrice'?
# 应该验证四个假设：
# 1. 正态性-当谈论正态性时，我们的意思是数据看起来应该像正态分布。 这很重要，因为几个统计检验都依赖于此（例如t统计）。 在本练习中，我们将仅检查“ SalePrice”的单变量正态性（这是一种有限的方法）。 请记住，单变量正态性不能确保多元正态性（这是我们希望拥有的），但可以提供帮助。 要考虑的另一个细节是，在大样本（> 200个观测值）中，正态性不是这样的问题。 但是，如果我们解决正态性，就可以避免很多其他问题（例如，异方差性），这就是我们进行此分析的主要原因。
# 2. 同方差性 - 我只希望我写的是正确的。 同方差性是指“假设因变量在预测变量范围内表现出相等的方差水平”。 同方差性是理想的，因为我们希望误差项在自变量的所有值上都相同。
# 3. 线性-评估线性的最常用方法是检查散点图并搜索线性模式。 如果模式不是线性的，则探索数据转换是值得的。 但是，由于我们所看到的大多数散点图似乎都具有线性关系，因此我们不会对此进行讨论。
# 4. 缺少相关错误-正如定义所暗示的，相关错误发生在一个错误与另一个错误相关时。 例如，如果一个正误差系统地产生一个负误差，则意味着这些变量之间存在关联。 这通常发生在时间序列中，其中某些模式与时间相关。 我们也不会涉及到这一点。 但是，如果检测到某些东西，请尝试添加一个变量，该变量可以解释所获得的效果。 这是相关错误的最常见解决方案。
# 这里的重点是要以非常精简的方式测试“ SalePrice”。 我们将注意以下事项：
# 1. 直方图-峰度和偏度。
# 2. 正态概率图-数据分布应紧跟代表正态分布的对角线。

# histogram and normal probability plot
sns.distplot(train_data['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
# 结论：
# 'SalePrice' is not normal. It shows 'peakedness', positive skewness and does not follow the diagonal line(对角线).
# 对于正偏度，通过取log来纠正
train_data['SalePrice'] = np.log(train_data['SalePrice'])
# transformed histogram and normal probability plot
sns.distplot(train_data['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
# Done!

# %% 处理 GrLivArea
sns.distplot(train_data['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['GrLivArea'], plot=plt)
# log
train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
# transformed histogram and normal probability plot
sns.distplot(train_data['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['GrLivArea'], plot=plt)

# %%timeit
# 处理 TotalBsmtSF
sns.distplot(train_data['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['TotalBsmtSF'], plot=plt)
# 对于没有bsmt的数据不能取对数
train_data['TotalBsmtSF'] = train_data['TotalBsmtSF'].apply(
    lambda x: np.log(x) if x != 0 else x)


sns.distplot(train_data[train_data['TotalBsmtSF'] > 0]
             ['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(
    train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
