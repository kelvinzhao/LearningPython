# 参考文章：https://zhuanlan.zhihu.com/p/39429689
# %% 引入模块

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

# 发现四张图中有异常值，去掉
train_data.drop(train_data[(train_data['OverallQual'] < 5) & (
    train_data['SalePrice'] > 200000)].index, inplace=True)
train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (
    train_data['SalePrice'] < 200000)].index, inplace=True)
train_data.drop(train_data[(train_data['YearBuilt'] < 1900) & (
    train_data['SalePrice'] > 400000)].index, inplace=True)
train_data.drop(train_data[(train_data['TotalBsmtSF'] > 6000) & (
    train_data['SalePrice'] < 200000)].index, inplace=True)
train_data.reset_index(drop=True, inplace=True)

# %% 观察变量相关性

corrmat = train_data.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()


# %% 数据合并
# 这里我们先将训练数据集和测试数据集合并为一个数据集，
# 这样做除了方便之后可以同时对训练数据集和测试数据集进行数据清洗
# 和特征工程，此外，也考虑在之后对类别型变量（category variable）
# 需要进行标签编码（LabelEncoder）和独热编码(OneHotEncoder），
# 而标签编码和独热编码主要是基于类别变量的特征值进行编码，为了避
# 免测试集的类别变量存在训练集所不具有的特征值，而影响模型的性能，
# 因此这里先将两个数据集进行合并，在最后对模型进行训练时再将合并的
# 数据集按照索引重新分割为训练集和测试集。

test_data = pd.read_csv('test.csv')
my_data = pd.concat([train_data, test_data], axis=0)
my_data.reset_index(drop=True, inplace=True)
train_index = train_data.index
test_index = list(set(my_data.index).difference(set(train_data.index)))

# %% 处理缺失数据
# 对于缺失数据的处理，通常会有以下几种做法
# 1. 如果缺失的数据过多，可以考虑删除该列特征
# 2. 用平均值、中值、分位数、众数、随机值等替代。但是效果一般，因为等于人为增加了噪声
# 3. 用插值法进行拟合
# 4. 用其他变量做预测模型来算出缺失变量。效果比方法1略好。有一个根本缺陷，如果其他变量和缺失变量无关，则预测的结果无意义
# 5. 最精确的做法，把变量映射到高维空间。比如性别，有男、女、缺失三种情况，则映射成3个变量：是否男、是否女、是否缺失。缺点就是计算量会加大。
al_data = pd.concat([train_data, test_data])
count = al_data.isnull().sum().sort_values(ascending=False)
ratio = count/len(al_data)
nulldata = pd.concat([count, ratio], axis=1, keys=['count', 'ratio'])
nulldata[nulldata['count'] > 0]

# %% 插补缺失数据
#


def fill_missings(res):
    res['PoolQC'] = res['PoolQC'].fillna(res['PoolQC'].mode()[0])  # 游泳池质量
    res['MiscFeature'] = res['MiscFeature'].fillna('missing')  # 其他类别未涵盖的其他功能
    res['Alley'] = res['Alley'].fillna('missing')  # 小路
    res['Fence'] = res['Fence'].fillna('missing')  # 篱笆
    res['FireplaceQu'] = res['FireplaceQu'].fillna(
        res['FireplaceQu'].mode()[0])  # 火炉质量
    res['GarageQual'] = res['GarageQual'].fillna(
        res['GarageQual'].mode()[0])  # 车库质量
    res['GarageFinish'] = res['GarageFinish'].fillna(
        res['GarageFinish'].mode()[0])  # 车库内部装修
    res['GarageCond'] = res['GarageCond'].fillna('missing')  # 车库条件
    res['GarageType'] = res['GarageType'].fillna('missing')  # 车库类型
    res['BsmtExposure'] = res['BsmtExposure'].fillna(
        res['BsmtExposure'].mode()[0])
    res['BsmtCond'] = res['BsmtCond'].fillna(
        res['BsmtCond'].mode()[0])  # 地下室条件
    res['BsmtQual'] = res['BsmtQual'].fillna(
        res['BsmtQual'].mode()[0])  # 地下室质量
    res['BsmtFinType2'] = res['BsmtFinType2'].fillna('missing')
    res['BsmtFinType1'] = res['BsmtFinType1'].fillna('missing')
    res['MasVnrType'] = res['MasVnrType'].fillna('None')  # 砖石饰面类型
    res['MSZoning'] = res['MSZoning'].fillna(res['MSZoning'].mode()[0])
    res['Utilities'] = res['Utilities'].fillna('missing')
    res["Functional"] = res["Functional"].fillna("Typ")
    res['Exterior1st'] = res['Exterior1st'].fillna(
        res['Exterior1st'].mode()[0])
    res['Electrical'] = res['Electrical'].fillna(res['Electrical'].mode()[0])
    res['KitchenQual'] = res['KitchenQual'].fillna(
        res['KitchenQual'].mode()[0])
    res['SaleType'] = res['SaleType'].fillna(res['SaleType'].mode()[0])
    res['Exterior2nd'] = res['Exterior2nd'].fillna(
        res['Exterior2nd'].mode()[0])
    res['Street'] = res['Street'].fillna('missing')
    res['LotShape'] = res['LotShape'].fillna('missing')
    res['LandContour'] = res['LandContour'].fillna('missing')
    res['CentralAir'] = res['CentralAir'].fillna('missing')
    res['SaleCondition'] = res['SaleCondition'].fillna('missing')
    # 数值型变量的空值先用0值替换
    flist = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
             'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'GarageYrBlt']
    for fl in flist:
        res[fl] = res[fl].fillna(0)
    # 0值替换
    res['TotalBsmtSF'] = res['TotalBsmtSF'].apply(
        lambda x: np.exp(6) if x <= 0.0 else x)
    res['2ndFlrSF'] = res['2ndFlrSF'].apply(
        lambda x: np.exp(6.5) if x <= 0.0 else x)
    res['GarageArea'] = res['GarageArea'].apply(
        lambda x: np.exp(6) if x <= 0.0 else x)
    res['GarageCars'] = res['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
    res['LotFrontage'] = res['LotFrontage'].apply(
        lambda x: np.exp(4.2) if x <= 0.0 else x)
    res['MasVnrArea'] = res['MasVnrArea'].apply(
        lambda x: np.exp(4) if x <= 0.0 else x)
    res['BsmtFinSF1'] = res['BsmtFinSF1'].apply(
        lambda x: np.exp(6.5) if x <= 0.0 else x)
    return res


mydata = fill_missings(my_data)
# 结果
#           	count	ratio
# SalePrice	    1459	0.500515
# GarageYrBlt	159	    0.054545

# %% 特征工程
# 一些特征其被表示成数值特征缺乏意义，例如年份还有类别，这里将其转换为字符串，即类别型变量。
mydata['MSSubClass'] = mydata['MSSubClass'].apply(str)
mydata['YrSold'] = mydata['YrSold'].astype(str)
mydata['MoSold'] = mydata['MoSold'].astype(str)
mydata['OverallCond'] = mydata['OverallCond'].astype(str)
