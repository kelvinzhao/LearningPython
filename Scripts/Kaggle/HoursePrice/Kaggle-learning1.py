# %%
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

# %%
# 生成 random forest 参数矩阵
# Number of trees in random forest


def gen_random_grid():
    n_estimators = [int(x) for x in np.linspace(start=50, stop=200, num=15)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    return random_grid


print("构造随机参数矩阵方法，完成!")


def random_trainer(in_X, in_y):
    rf = RandomForestRegressor()
    random_grid = gen_random_grid()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=3, verbose=2, random_state=1, n_jobs=-1)
    # Fit the random search model with all data
    rf_random.fit(in_X, in_y)
    return(rf_random.best_estimator_)


print("构造最佳参数随机森林方法，完成!")

# %%
# 读取训练数据
# Path of the file to read. We changed the directory structure to simplify submitting to a competition

home_data = pd.read_csv('train.csv')
print("读取 train.csv 数据，完成!")

# %%
# 数据直觉观察，去除异常值
# home_data.shape
# home_data.info()
# 查看缺失值
# home_data.isnull().sum()
# -----------------
#plt.figure(figsize=(15, 8))
#sns.boxplot(home_data['YearBuilt'], home_data['SalePrice'])
# -----------------
# plt.figure(figsize=(12, 6))
# plt.scatter(x=home_data['GrLivArea'], y=home_data['SalePrice'])
# plt.ylim(0, 800000)
# plt.xlabel("GrLivArea", fontsize=13)
# plt.ylabel("SalePrice", fontsize=13)
# -----------------
# 可以看到GrlivArea>4000 且售价低于30000 有两个异常值，去掉它们。
home_data.drop(home_data[(home_data['GrLivArea'] > 4000) & (
    home_data['SalePrice'] < 300000)].index, inplace=True)
# -----------------
# plt.figure(figsize=(12, 6))
# plt.scatter(x=home_data['LotFrontage'], y=home_data['SalePrice'])
# plt.ylim(0, 800000)
# plt.xlabel("LotFrontage", fontsize=13)
# plt.ylabel("SalePrice", fontsize=13)
# -----------------
# 发现LotFrontage大雨300的一个异常值，去掉？
home_data.drop(home_data[home_data['LotFrontage'] > 300].index, inplace=True)

print("去除异常值，完成!")
# 看房价分布
# home_data['SalePrice'].describe()
# plt.figure(figsize=(10, 5))
# print("skew: ", home_data['SalePrice'].skew())  # 求偏度，
# sns.distplot(home_data['SalePrice'], color="red")
# 我们可以看到目标变量呈现偏态分布。当使用线性回归方法时，如果目标变量出现偏斜，则有必要对目标变量进行对数变换（log-transform）。通过对数变换，可以改善数据的线性度。

# %%
# 数据清洗
# nullsum = home_data.isnull().sum()
# hasnull_column = nullsum[nullsum != 0].index.tolist()
# print("以下列中包含空数据：")
# print(hasnull_column)

# home_data[hasnull_column] = home_data[hasnull_column].fillna(
#     home_data[hasnull_column].mean())
# print("数值型空值以平均值填充，完成!")

# ------------------------------------------------
# 似乎直接填充空值太武断了，尝试如下，
# 源自知乎：https://zhuanlan.zhihu.com/p/34904202
# ------------------------------------------------
# aa = home_data.isnull().sum()
# aa[aa > 0].sort_values(ascending=False)

# 有空值的列及数量如下：
# PoolQC          1453
# MiscFeature     1406
# Alley           1369
# Fence           1179
# FireplaceQu      690
# LotFrontage      259
# GarageYrBlt       81
# GarageType        81
# GarageFinish      81
# GarageQual        81
# GarageCond        81
# BsmtFinType2      38
# BsmtExposure      38
# BsmtFinType1      37
# BsmtCond          37
# BsmtQual          37
# MasVnrArea         8
# MasVnrType         8
# Electrical         1

# 为了方便对 test 数据也进行相同的清洗，这里定义为方法


def DataClean(fulldata):

    # 有些字段为空，表示房子没有该设施，则空值用"None"填充
    cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
             "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
    for col in cols1:
        fulldata[col].fillna("None", inplace=True)

    # 下面的这些特征多为表示XX面积或数量，比如 TotalBsmtSF 表示地下室的面积，如果一个房子本身没有地下室，则缺失值就用0来填补。
    cols = ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
            "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]
    for col in cols:
        fulldata[col].fillna(0, inplace=True)

    # LotFrontage 这个特征与 LotArea 和 Neighborhood 有比较大的关系，所以这里用这两个特征分组后的中位数进行插补。
    fulldata['LotFrontage'] = fulldata.groupby(
        ['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # Electrical 缺的这个值没找到原因，就用最通用的 'SBrkr' 填充
    fulldata['Electrical'].fillna('SBrkr', inplace=True)

    # 其他字段处理，这些字段应该取选项值最多的那个，或者设为none？
    fulldata['MSZoning'].fillna("RL", inplace=True)
    fulldata['Utilities'].fillna("AllPub", inplace=True)
    fulldata['Exterior1st'].fillna("None", inplace=True)
    fulldata['Exterior2nd'].fillna("None", inplace=True)
    fulldata['KitchenQual'].fillna("TA", inplace=True)
    fulldata['Functional'].fillna("Typ", inplace=True)
    fulldata['SaleType'].fillna("Oth", inplace=True)
    return "空值处理完成!"


DataClean(home_data)

# %%
# 找一些离散数据，先转成str类型，再map。


def map_Values(fulldata):
    NumStr = ['MSSubClass', 'BedroomAbvGr', 'MoSold', 'MSZoning',
              'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig']
    for col in NumStr:
        fulldata[col] = home_data[col].astype(str)

    fulldata['oMSSubClass'] = fulldata['MSSubClass'].map({
        '180': 1,
        '30': 2, '45': 2,
        '190': 3, '50': 3, '90': 3,
        '85': 4, '160': 4, '40': 4,
        '70': 5, '20': 5, '75': 5, '80': 5,
        '120': 6, '60': 6
    })
    fulldata['oBedroomAbvGr'] = fulldata['BedroomAbvGr'].map({
        '2': 1,
        '1': 2, '6': 2,
        '3': 3, '5': 3,
        '0': 4, '4': 4, '8': 4
    })
    fulldata['oMoSold'] = fulldata['MoSold'].map({
        '1': 1, '4': 1,
        '5': 2, '10': 2,
        '3': 3, '6': 3, '7': 3,
        '2': 4, '8': 4, '9': 4, '11': 4, '12': 4
    })
    fulldata['oMSZoning'] = fulldata['MSZoning'].map({
        'C (all)': 1,
        'RH': 2, 'RM': 2,
        'RL': 3,
        'FV': 4
    })
    fulldata['oStreet'] = fulldata['Street'].map({
        'Grvl': 1,
        'Pave': 2
    })
    fulldata['oLotShape'] = fulldata['LotShape'].map({
        'Reg': 1,
        'IR1': 2,
        'IR2': 3, 'IR3': 3
    })
    fulldata['oLandContour'] = fulldata['LandContour'].map({
        'Bnk': 1,
        'Lvl': 2,
        'Low': 3,
        'HLS': 3
    })
    fulldata['oUtilities'] = fulldata['Utilities'].map({
        'NoSeWa': 1,
        'AllPub': 2
    })
    fulldata['oLotConfig'] = fulldata['LotConfig'].map({
        'Inside': 1, 'Corner': 1, 'FR2': 1,
        'CulDSac': 2, 'FR3': 2
    })
    fulldata.drop(NumStr, axis=1, inplace=True)
    return "map 离散数据完成！"


home_data_bak = home_data.copy()
map_Values(home_data)

# %%
# 区分开数字特性和文字特性
all_dtypes = home_data.dtypes
object_features = all_dtypes[all_dtypes == 'object'].index.tolist()
num_features = all_dtypes[all_dtypes != 'object'].index.tolist()
num_features.remove('Id')


# %%
# 挑选features
prefeatures = num_features
corrMat = home_data[prefeatures].corr()
# mask = np.array(corrMat)
# mask[np.tril_indices_from(mask)] = False
# plt.subplots(figsize=(20, 10))
# plt.xticks(rotation=60)
# sns.heatmap(corrMat, mask=mask, vmax=.8, square=True, annot=True)

print(corrMat["SalePrice"].sort_values(ascending=False))
corr = corrMat["SalePrice"]
features = corr[corr >= 0.4].index.tolist()
features.remove('SalePrice')
print("筛选关联度大于 0.4 的参数列表，得到 features 如下：")
print(features)
home_data[features].to_csv('features-data.csv', index=False)


# %%
# 生成训练数据集
X = home_data[features]
y = home_data['SalePrice']
y2 = np.log(home_data['SalePrice'])
print("生成 X、y 、y2完成！")

# 归一化 X
X_copy = X[:]
scaler = MinMaxScaler()
X_transformed = scaler.fit_transform(X_copy)
print("生成归一化 X_transformed 完成！")

# %%
# 生成训练数据和校验数据
train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=1)
print("数据分割为 train 和 val 完成！")

# 生成归一化的训练数据和校验数据
train_X2, val_X2, train_y2, val_y2 = train_test_split(
    X_transformed, y2, random_state=1)
print("归一化数据分割为 train 和 val 完成！")

# %%
# 随机森林模型，不用归一化

rf_best_model = random_trainer(train_X, train_y)
rf_bst_predictions = rf_best_model.predict(val_X)
rf_bst_mae = mean_absolute_error(rf_bst_predictions, val_y)
print(
    "验证最佳随机森林模型的 MAE 值是：", rf_bst_mae)

# 线性回归模型，归一化
lr_model = LinearRegression()
lr_model.fit(train_X2, train_y2)
lr_val_predictions = lr_model.predict(val_X2)
lr_val_mae = mean_absolute_error(np.exp(lr_val_predictions), np.exp(val_y2))
print("验证线性回归模型的 MAE 值是：", lr_val_mae)
print("线性回归模型得分为：", lr_model.score(val_X2, val_y2))


# %%
# 使用 test 数据集进行预测

test_data = pd.read_csv('test.csv')
print("读取测试数据集，完成！")
# nullsum_test = test_data.isnull().sum()
# hasnull_column_test = nullsum_test[nullsum_test != 0].index.tolist()
# print("以下列中包含空数据：")
# print(hasnull_column_test)
# test_data[hasnull_column_test] = test_data[hasnull_column_test].fillna(
#     test_data[hasnull_column_test].mean())
# print("数值型空值以平均值填充，完成!")
DataClean(test_data)

test_data_bak = test_data.copy()
map_Values(test_data)

test_X = test_data[features]

# 使用随机森林对 test 数据进行预测，生成预测值
# rf_test_model = random_trainer(X, y)
# print("使用全部数据重新训练模型，完成！")
# 随机森林模型不需要对数据归一化处理，直接使用 test_X 输入
rf_test_preds = rf_best_model.predict(test_X)
print("随机森林预测完成！")

# 使用线性回归对 test 数据进行预测，生成预测值
# lr_model.fit(X_transformed, y2)
# print("使用全部归一化数据重新训练模型，完成！")
# 归一化 test_X
test_X_copy = test_X[:]
test_X_transformed = scaler.fit_transform(test_X_copy)
print("test_X 泛化完成！")
lr_test_preds = np.exp(lr_model.predict(test_X_transformed))
print("线性回归预测完成！")

# %%
# 生成提交文件

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': rf_test_preds})
output.to_csv('submission-rf.csv', index=False)
print("生成 submission-rf.csv 完成！")

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': lr_test_preds})
output.to_csv('submission-lr.csv', index=False)
print("生成 submission-lr.csv 完成！")
