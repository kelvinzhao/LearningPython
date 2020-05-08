# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
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

# 找到含有空值的列
nullsum = home_data.isnull().sum()
hasnull_column = nullsum[nullsum != 0].index.tolist()
print("以下列中包含空数据：")
print(hasnull_column)

home_data[hasnull_column] = home_data[hasnull_column].fillna(
    home_data[hasnull_column].mean())
print("数值型空值以平均值填充，完成!")

# outputtest = pd.DataFrame(home_data)
# outputtest.to_csv('home-data.csv')

# %%
# 数据直觉观察
# home_data.shape
# home_data.info()
# 查看缺失值
# home_data.isnull().sum()
# -----------------
# plt.figure(figsize=(15,8))
# sns.boxplot(home_data['LotArea'], home_data['SalePrice'])
# sns.boxplot(home_data['YearBuilt'], home_data['SalePrice'])

# 看房价分布
# home_data['SalePrice'].describe()
# plt.figure(figsize=(10, 5))
# print("skew: ", home_data['SalePrice'].skew())  # 求偏度，
# sns.distplot(home_data['SalePrice'], color="red")

# 我们可以看到目标变量呈现偏态分布。当使用线性回归方法时，如果目标变量出现偏斜，则有必要对目标变量进行对数变换（log-transform）。通过对数变换，可以改善数据的线性度。

# %%
# 区分开数字特性和文字特性
all_dtypes = home_data.dtypes
object_features = all_dtypes[all_dtypes == 'object'].index.tolist()
num_features = all_dtypes[all_dtypes != 'object'].index.tolist()
num_features.remove('Id')

print("区分数字特性和文字特性，完成!")

# %%
# 检测数值特征和目标变量之间的相关性
prefeatures = num_features
corrMat = home_data[prefeatures].corr()
# mask = np.array(corrMat)
# mask[np.tril_indices_from(mask)] = False
# plt.subplots(figsize=(20, 10))
# plt.xticks(rotation=60)
# sns.heatmap(corrMat, mask=mask, vmax=.8, square=True, annot=True)

print(corrMat["SalePrice"].sort_values(ascending=False))
corr = corrMat["SalePrice"]
features = corr[corr >= 0.1].index.tolist()
features.remove('SalePrice')
print("筛选关联度大于 0.1 的参数列表，得到 features 如下：")
print(features)

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
nullsum_test = test_data.isnull().sum()
hasnull_column_test = nullsum_test[nullsum_test != 0].index.tolist()
print("以下列中包含空数据：")
print(hasnull_column_test)
test_data[hasnull_column_test] = test_data[hasnull_column_test].fillna(
    test_data[hasnull_column_test].mean())
print("数值型空值以平均值填充，完成!")
test_X = test_data[features]

# 使用随机森林对 test 数据进行预测，生成预测值
rf_test_model = random_trainer(X, y)
print("使用全部数据重新训练模型，完成！")
# 随机森林模型不需要对数据归一化处理，直接使用 test_X 输入
rf_test_preds = rf_test_model.predict(test_X)
print("随机森林预测完成！")

# 使用线性回归对 test 数据进行预测，生成预测值
lr_model.fit(X_transformed, y2)
print("使用全部归一化数据重新训练模型，完成！")
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
