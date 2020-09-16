# %% import modules
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# %% prepare the data
music_data = pd.read_csv("music.csv")
X = music_data.drop(columns='genre')
y = music_data['genre']

# %% train the model
# algorithm - decision tree
model = DecisionTreeClassifier()
model.fit(X, y)

tree.export_graphviz(model, out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     filled=True,
                     rounded=True)
