#!/usr/bin/env python
# coding: utf-8

# # OVERVIEW
# 
# ![](https://si.wsj.net/public/resources/images/B3-DM067_RIGHTS_IM_20190319162958.jpg)
# 
# Here I made an ULTIMATE Top 5% House Pricing Guide to making a great House Pricing model. 
# This guide consists of many explanations and simple actions in order to efficiently process data and setup a model for submission. Data Visualization is used a lot in this notebook in order to understand the data clearly and perform necessary actions.
# 
# I will constantly update this notebook if I find a more efficient way to doing something or if lots of support is shown for this notebook so please show your support and like this notebook in order to motivate me to updating this notebook more often. If you have questions or any tips please comment below 😁. VERSION LOGS are below.
# 
# 
# # VERSION LOGS:
# ### - Version 1: 
# - RELEASED
# 
# <br>
# 
# ### - Version 2 [CURRENT]: 
# - Added Data Science Workflow map for a general idea of the process
# - New subsection 'SalePrice Distribution' which visualizes SalePrice before and after log-transformation
# - We decide to log-transform SalePrice and inverse-transform it during modelling
# 
# 
# # TABLE OF CONTENTS:
# ### [1) IMPORTING LIBRARIES](#1)
# ### [2) READING DATA AND COMPREHENDING DATA](#2)
# ### [3) DATA VISUALIZATION](#3)
# ### [4) DATA PROCESSING](#4)
# ### [5) FEATURE ENGINEERING](#5)
# ### [6) MODELLING](#6)
# ### [7) SUBMISSION](#7)
# 
# # DATA SCIENCE WORKFLOW:
# ![](https://miro.medium.com/max/2000/1*3FQbrDoP1w1oibNPj9YeDw.png)

# # PLEASE <font color='red'><b>U</b></font><font color='orange'><b>P</b></font><font color='yellow'><b>V</b></font><font color='green'><b>O</b></font><font color='blue'><b>T</b></font><font color='purple'><b>E</b></font>👍 if you found HELPFUL!  
# 

# <a id="1"></a> <br>
# # 1) IMPORTING LIBRARIES

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LinearRegression
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id="2"></a> <br>
# # 2) READING DATA AND COMPREHENDING DATA
# In this section:
# - 2.1 Reading Data
# - 2.2 Understanding the Data
# - 2.3 Checking for missing values

# ### 2.1 Reading Data

# In[ ]:


home = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col='Id')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col='Id')


# ### 2.2 Understanding the Data

# In[ ]:


home.head()


# In[ ]:


test.head()


# In[ ]:


home.shape


# In[ ]:


test.shape


# In[ ]:


home.info()


# In[ ]:


test.info()


# ### 2.3 Checking for Missing Values

# In[ ]:


#missing values
missing = home.isnull().sum()
missing = missing[missing>0]
missing.sort_values(inplace=True)
missing.plot.bar()


# <a id="3"></a> <br>
# # 3) DATA VISUALIZATION
# In this section:
# 
# - 3.1 Viewing Columns
# 
# - 3.2 Distribution of Data
# 
# - 3.3 Univariate Analysis of Data
# 
# - 3.4 Bivariate Analysis of Data

# ### 3.1 Viewing Columns

# In[ ]:


numerical_features = home.select_dtypes(exclude=['object']).drop(['SalePrice'], axis=1).copy()
print(numerical_features.columns)


# In[ ]:


categorical_features = home.select_dtypes(include=['object']).copy()
print(categorical_features.columns)


# ### 3.2 Distribution of Data

# In[ ]:


fig = plt.figure(figsize=(12,18))
for i in range(len(numerical_features.columns)):
    fig.add_subplot(9,4,i+1)
    sns.distplot(numerical_features.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})
    plt.xlabel(numerical_features.columns[i])
plt.tight_layout()
plt.show()


# ### 3.3 Univariate Analysis

# In[ ]:


fig = plt.figure(figsize=(12,18))
for i in range(len(numerical_features.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=numerical_features.iloc[:,i])

plt.tight_layout()
plt.show()


# ### 3.4 Bivariate Analysis

# In[ ]:


fig = plt.figure(figsize=(12,18))
for i in range(len(numerical_features.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.scatterplot(numerical_features.iloc[:, i],home['SalePrice'])
plt.tight_layout()
plt.show()


# <a id="4"></a> <br>
# # 4) DATA PROCESSING
# In this section:
# - 4.1 Outliers
# - 4.2 Removing Certain Features
# - 4.3 Filling Numerical Missing Values
# - 4.4 Filling Categorical Missing Values
# - 4.5 Filling Missing Values in 'LotFrontage'

# ### 4.1 Outliers

# Notes on Outliers:
# According to the plots above, these are the features which appear to have outliers:
# - LotFrontage
# - LotArea
# - MasVnrArea
# - BsmtFinSF1
# - TotalBsmtSF
# - GrLivArea
# - 1stFlrSF
# - EnclosedPorch
# - MiscVal
# - LowQualFinSF
# 
# Let's take a closer look at these features...

# In[ ]:


figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows=5, ncols=2)
figure.set_size_inches(16,28)
_ = sns.regplot(home['LotFrontage'], home['SalePrice'], ax=ax1)
_ = sns.regplot(home['LotArea'], home['SalePrice'], ax=ax2)
_ = sns.regplot(home['MasVnrArea'], home['SalePrice'], ax=ax3)
_ = sns.regplot(home['BsmtFinSF1'], home['SalePrice'], ax=ax4)
_ = sns.regplot(home['TotalBsmtSF'], home['SalePrice'], ax=ax5)
_ = sns.regplot(home['GrLivArea'], home['SalePrice'], ax=ax6)
_ = sns.regplot(home['1stFlrSF'], home['SalePrice'], ax=ax7)
_ = sns.regplot(home['EnclosedPorch'], home['SalePrice'], ax=ax8)
_ = sns.regplot(home['MiscVal'], home['SalePrice'], ax=ax9)
_ = sns.regplot(home['LowQualFinSF'], home['SalePrice'], ax=ax10)


# In[ ]:


home.shape


# From these regplots we have confirmed there are outliers, so we decide to remove them.

# In[ ]:


home = home.drop(home[home['LotFrontage']>200].index)
home = home.drop(home[home['LotArea']>100000].index)
home = home.drop(home[home['MasVnrArea']>1200].index)
home = home.drop(home[home['BsmtFinSF1']>4000].index)
home = home.drop(home[home['TotalBsmtSF']>4000].index)
home = home.drop(home[(home['GrLivArea']>4000) & (home['SalePrice']<300000)].index)
home = home.drop(home[home['1stFlrSF']>4000].index)
home = home.drop(home[home['EnclosedPorch']>500].index)
home = home.drop(home[home['MiscVal']>5000].index)
home = home.drop(home[(home['LowQualFinSF']>600) & (home['SalePrice']>400000)].index)


# ### 4.2 Removing Certain Features

# - Find the highly-correlated (correlations higher than 0.8)

# In[ ]:


num_correlation = home.select_dtypes(exclude='object').corr()
plt.figure(figsize=(20,20))
plt.title('High Correlation')
sns.heatmap(num_correlation > 0.8, annot=True, square=True)


# Highly-Correlated Features:
# - YearBuilt vs GarageYrBlt
# - 1stFlrSF vs TotalBsmtSF
# - GrLivArea vs TotRmsAbvGrd
# - GarageCars vs GarageArea

# In[ ]:


corr = num_correlation.corr()
print(corr['SalePrice'].sort_values(ascending=False))


# Drop column with a lower correlation to SalePrice of the pair
# 
# (Ex: GarageCars(0.88) & GarageArea(0.876))
# 
# **DROP GarageArea**

# In[ ]:


home.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True) 
test.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True)


# We also have useless features so we also decide to drop the features below

# In[ ]:


# Useless Columns...
home=home.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating']) 
test=test.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating']) 


# Lets see which features have the most missing values...

# In[ ]:


home.isnull().mean().sort_values(ascending=False).head(3)


# We can clearly see features 'Alley', 'MiscFeature' and 'PoolQC' are missing over 90% of their values. So we decide to remove them.
# 'PoolArea' is pretty much a useless column because 99.6% of 'PoolQC' is missing so we also drop this feature.

# In[ ]:


home.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea'], axis=1, inplace=True)
test.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea'], axis=1, inplace=True)


# In[ ]:


test.isnull().mean().sort_values(ascending=False).head(3)


# Test data doesn't have any features that have over 90% of missing values. So we don't drop any features.

# ### 4.3 Filling Numerical Missing Values

# In[ ]:


# Checking Home and Test data missing value percentage
null = pd.DataFrame(data={'Home Null Percentage': home.isnull().sum()[home.isnull().sum() > 0], 'Test Null Percentage': test.isnull().sum()[test.isnull().sum() > 0]})
null = (null/len(home)) * 100

null.index.name='Feature'
null


# In[ ]:


home.isnull().sum().sort_values(ascending=False)[:50]


# In[ ]:


home_num_features = home.select_dtypes(exclude='object').isnull().mean()
test_num_features = test.select_dtypes(exclude='object').isnull().mean()

num_null_features = pd.DataFrame(data={'Missing Num Home Percentage: ': home_num_features[home_num_features>0], 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]})
num_null_features.index.name = 'Numerical Features'
num_null_features


# In[ ]:


for df in [home, test]:
    for col in ('GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
                'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotalBsmtSF',
                'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'MoSold', 'YrSold',
                'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea'):
                    df[col] = df[col].fillna(0)


# In[ ]:


_=sns.regplot(home['LotFrontage'],home['SalePrice'])


# In[ ]:


home_num_features = home.select_dtypes(exclude='object').isnull().mean()
test_num_features = test.select_dtypes(exclude='object').isnull().mean()

num_null_features = pd.DataFrame(data={'Missing Num Home Percentage: ': home_num_features[home_num_features>0], 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]})
num_null_features.index.name = 'Numerical Features'
num_null_features


# - We will deal with 'LotFrontage' later because it is an important feature

# ### 4.4 Filling Categorical Missing Values

# In[ ]:


cat_col = home.select_dtypes(include='object').columns
print(cat_col)


# In[ ]:


home_cat_features = home.select_dtypes(include='object').isnull().mean()
test_cat_features = test.select_dtypes(include='object').isnull().mean()

cat_null_features = pd.DataFrame(data={'Missing Cat Home Percentage: ': home_cat_features[home_cat_features>0], 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]})
cat_null_features.index.name = 'Categorical Features'
cat_null_features


# In[ ]:


cat_col = home.select_dtypes(include='object').columns

columns = len(cat_col)/4+1

fg, ax = plt.subplots(figsize=(20, 30))

for i, col in enumerate(cat_col):
    fg.add_subplot(columns, 4, i+1)
    sns.countplot(home[col])
    plt.xlabel(col)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# In[ ]:


var = home['KitchenQual']
f, ax = plt.subplots(figsize=(10,6))
sns.boxplot(y=home.SalePrice, x=var)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(12,8))
sns.boxplot(y=home.SalePrice, x=home.Neighborhood)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


## Count of categories within Neighborhood attribute
fig = plt.figure(figsize=(12.5,4))
sns.countplot(x='Neighborhood', data=home)
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.show()


# In[ ]:


for df in [home, test]:
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                  'BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu', 'Fence'):
        df[col] = df[col].fillna('None')


# In[ ]:


for df in [home, test]:
    for col in ('LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Condition1', 'RoofStyle',
                  'Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'ExterQual', 'ExterCond',
                  'Foundation', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition'):
        df[col] = df[col].fillna(df[col].mode()[0])


# In[ ]:


home_cat_features = home.select_dtypes(include='object').isnull().mean()
test_cat_features = test.select_dtypes(include='object').isnull().mean()

cat_null_features = pd.DataFrame(data={'Missing Cat Home Percentage: ': home_cat_features[home_cat_features>0], 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]})
cat_null_features.index.name = 'Categorical Features'
cat_null_features


# ### 4.5 Filling Missing Values in 'LotFrontage'

# In[ ]:


_=sns.regplot(home['LotFrontage'],home['SalePrice'])


# LotFrontage is correlated to Neighborhood, so we fill in the median based off of Neighborhood feature

# In[ ]:


home['LotFrontage'] = home.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))


# In[ ]:


home.corr()['SalePrice'].sort_values(ascending=False)


# In[ ]:


home.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# For the remaining missing values we will impute them with 'SimpleImputer()' when modelling

# <a id="5"></a> <br>
# # 5) FEATURE ENGINEERING
# In this section:
# - 5.1 Create 'TotalSF' feature
# - 5.2 Create 'TotalBath' feature
# - 5.3 Create 'YrBuiltAndRemod' feature
# - 5.4 Create 'PorchSF' feature
# - 5.5 Creating extra features and changing types
# - 5.6 SalePrice Distribution Visualization

# In[ ]:


list(home.select_dtypes(exclude='object').columns)


# ### 5.1 Create 'TotalSF' feature

# In[ ]:


figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(20,10)
_ = sns.regplot(home['TotalBsmtSF'], home['SalePrice'], ax=ax1)
_ = sns.regplot(home['2ndFlrSF'], home['SalePrice'], ax=ax2)
_ = sns.regplot(home['TotalBsmtSF'] + home['2ndFlrSF'], home['SalePrice'], ax=ax3)


# In[ ]:


home['TotalSF']=home['TotalBsmtSF']  + home['2ndFlrSF']
test['TotalSF']=test['TotalBsmtSF']  + test['2ndFlrSF']


# ### 5.2 Create 'TotalBath' feature

# In[ ]:


figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(14,10)
_ = sns.barplot(home['BsmtFullBath'], home['SalePrice'], ax=ax1)
_ = sns.barplot(home['FullBath'], home['SalePrice'], ax=ax2)
_ = sns.barplot(home['BsmtHalfBath'], home['SalePrice'], ax=ax3)
_ = sns.barplot(home['BsmtFullBath'] + home['FullBath'] + home['BsmtHalfBath'] + home['HalfBath'], home['SalePrice'], ax=ax4)


# In[ ]:


home['TotalBath']=home['BsmtFullBath'] + home['FullBath'] + (0.5*home['BsmtHalfBath']) + (0.5*home['HalfBath'])
test['TotalBath']=test['BsmtFullBath'] + test['FullBath'] + test['BsmtHalfBath'] + test['HalfBath']


# ### 5.3 Create 'YrBuiltAndRemod' feature

# In[ ]:


figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18,8)
_ = sns.regplot(home['YearBuilt'], home['SalePrice'], ax=ax1)
_ = sns.regplot(home['YearRemodAdd'], home['SalePrice'], ax=ax2)
_ = sns.regplot((home['YearBuilt']+home['YearRemodAdd'])/2, home['SalePrice'], ax=ax3)


# In[ ]:


home['YrBltAndRemod']=home['YearBuilt']+(home['YearRemodAdd']/2)
test['YrBltAndRemod']=test['YearBuilt']+(test['YearRemodAdd']/2)


# ### 5.4 Create 'PorchSF' feature

# In[ ]:


figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(20,10)
_ = sns.regplot(home['OpenPorchSF'], home['SalePrice'], ax=ax1)
_ = sns.regplot(home['3SsnPorch'], home['SalePrice'], ax=ax2)
_ = sns.regplot(home['EnclosedPorch'], home['SalePrice'], ax=ax3)
_ = sns.regplot(home['ScreenPorch'], home['SalePrice'], ax=ax4)
_ = sns.regplot(home['WoodDeckSF'], home['SalePrice'], ax=ax5)
_ = sns.regplot((home['OpenPorchSF']+home['3SsnPorch']+home['EnclosedPorch']+home['ScreenPorch']+home['WoodDeckSF']), home['SalePrice'], ax=ax6)


# In[ ]:


home['Porch_SF'] = (home['OpenPorchSF'] + home['3SsnPorch'] + home['EnclosedPorch'] + home['ScreenPorch'] + home['WoodDeckSF'])
test['Porch_SF'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])


# ### 5.5 Creating extra features and changing types

# We create extra features in order to categorize data with and without a feature
# 
# For example:
# - 'HasPool' is 1 if you have pool and 0 if you don't have a pool

# In[ ]:


home['Has2ndfloor'] = home['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
home['HasBsmt'] = home['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
home['HasFirePlace'] = home['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
home['Has2ndFlr']=home['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
home['HasBsmt']=home['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

test['Has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasBsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasFirePlace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
test['Has2ndFlr']=test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasBsmt']=test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)


# Some features are the wrong type so we convert them to the right type

# In[ ]:


home['MSSubClass'] = home['MSSubClass'].apply(str)
home['MoSold']=home['MoSold'].astype(str)
home['YrSold']=home['YrSold'].astype(str)
home['LotArea'] = home['LotArea'].astype(np.int64)

test['MSSubClass'] = test['MSSubClass'].apply(str)
test['MoSold']=test['MoSold'].astype(str)
test['YrSold']=test['YrSold'].astype(str)
test['LotArea'] = test['LotArea'].astype(np.int64)


# ### 5.6 SalePrice Distribution Visualization

# In[ ]:


fig = plt.figure(figsize=(11,11))

print ("Skew of SalePrice:", home.SalePrice.skew())
plt.hist(home.SalePrice, normed=1, color='red')
plt.show()


# The graph shows that SalePrice is skewed to the right and must be modified

# In[ ]:


fig = plt.figure(figsize=(11,11))

print ("Skew of Log-Transformed SalePrice:", np.log1p(home.SalePrice).skew())
plt.hist(np.log1p(home.SalePrice), color='green')
plt.show()


# As we can see the skew improved from approximately 1.88 to approximately 0.12 so we will log-transform SalePrice in the next section

# <a id="6"></a> <br>
# # 6) MODELLING
# In this section:
# 
# - 6.1 Dealing with Data for Modelling
# - 6.2 Finding the Best Model
# - 6.3 Setting up Final Model for Submission

# ### 6.1 Dealing with Data for Modelling

# In[ ]:


X = home.drop(['SalePrice'], axis=1)
y = np.log1p(home['SalePrice'])


# Split X and y into train and valid data for model testing

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=2)


# In[ ]:


test.head()


# We select every numerical column from X and the categorical columns with unique values under 30

# In[ ]:


categorical_cols = [cname for cname in X.columns if
                    X[cname].nunique() <= 30 and
                    X[cname].dtype == "object"] 
                


numerical_cols = [cname for cname in X.columns if
                 X[cname].dtype in ['int64','float64']]


my_cols = numerical_cols + categorical_cols

X_train = X_train[my_cols].copy()
X_valid = X_valid[my_cols].copy()
X_test = test[my_cols].copy()


# Here we create a 'num_transformer' and a 'cat_transformer' for imputing and hot-encoding numerical and categorical values. We then store these transformers into a preprocessor column transformer

# In[ ]:


num_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='constant'))
    ])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),       
        ('cat',cat_transformer,categorical_cols),
        ])


# ### 6.2 Finding the Best Model

# We test three models: 'XGBoost', 'Lasso', and 'Gradient' and see which one performs the best

# In[ ]:


# Reversing log-transform on y
def inv_y(transformed_y):
    return np.exp(transformed_y)



scores=[]

n_folds = 10

model_names = ['XGBoost','Lasso','Gradient']
models =[XGBRegressor(learning_rate=0.01, n_estimators=3460,
                     max_depth=3, min_child_weight=0,
                     gamma=0, subsample=0.7,
                     colsample_bytree=0.7,
                     objective='reg:squarederror', nthread=-1,
                     scale_pos_weight=1, seed=27,
                     reg_alpha=0.00006), 
         LassoCV(max_iter=1e7,  random_state=14, cv=n_folds), 
         GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)]

for model in models:
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    
    clf.fit(X_train, y_train)
    preds = clf.predict(X_valid)
    score = mean_absolute_error(inv_y(y_valid), inv_y(preds))
    scores.append(score)
new_models_data_frame = pd.DataFrame({'Score': scores}, index=model_names)
new_models_data_frame


# As we can see XGBoost performed the best so we will be using this for our final model

# ### 6.3 Setting up Final Model for Submission

# We use the best model XGBoost and combine it with preprocessor which imputes and hot-encodes missing data. We then train and predict the combination.

# In[ ]:


model = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                     max_depth=3, min_child_weight=0,
                     gamma=0, subsample=0.7,
                     colsample_bytree=0.7,
                     objective='reg:squarederror', nthread=-1,
                     scale_pos_weight=1, seed=27,
                     reg_alpha=0.00006)

final_model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)


# <a id="7"></a> <br>
# # 7) SUBMISSION

# In[ ]:


output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': inv_y(final_predictions)})

output.to_csv('submission.csv', index=False)

