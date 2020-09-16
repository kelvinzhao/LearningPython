# %%
# 数据准备
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import os

melbourne_file_path = os.path.dirname(__file__) + '/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom',
                      'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

print("Set up Done!")

# %%
# 模型1，决策树


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# for max_leaf_nodes in [5, 50, 500, 5000]:
#     my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %
#           (max_leaf_nodes, my_mae))

[(x, get_mae(x, train_X, val_X, train_y, val_y)) for x in [5, 50, 500, 5000]]

# %%
