# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import os
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import numpy as np


# %%
# 生成 random forest 参数矩阵
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=80, stop=150, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
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
pprint(random_grid)


def random_trainer(in_X, in_y):
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=3, verbose=2, random_state=1, n_jobs=-1)
    # Fit the random search model with all data
    rf_random.fit(in_X, in_y)
    return(rf_random.best_estimator_)


# %%
# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = os.path.dirname(__file__) + 'train.csv'

home_data = pd.read_csv(iowa_file_path)
home_data = home_data.fillna(method='ffill')

# Create target object and call it y
y = home_data.SalePrice
# Create X
# features = ['LotArea', 'YearBuilt', '1stFlrSF',
#             '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',
            'BedroomAbvGr', 'TotRmsAbvGrd',
            'OverallQual', 'OverallCond', 'MSSubClass', 'YearBuilt', 'GrLivArea',
            'Fireplaces', 'KitchenAbvGr', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
            'YearRemodAdd', 'BsmtUnfSF']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print(
    "Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print(
    "Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(
    n_estimators=130, criterion="mae", random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

rf_best_model = random_trainer(train_X, train_y)
rf_bst_predictions = rf_best_model.predict(val_X)
rf_bst_mae = mean_absolute_error(rf_bst_predictions, val_y)
print(
    "Validation MAE for the Best Random Forest Model: {:,.0f}".format(rf_bst_mae))

# %%
# use best random forest model to generate submission.csv file.
#
# path to file you will use for predictions
test_data_path = os.path.dirname(__file__) + 'test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
test_data = test_data.fillna(method='ffill')
# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]


# make predictions which we will submit.
# test_preds = rf_model_on_full_data.predict(test_X)
test_model = random_trainer(X, y)
test_preds = test_model.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
