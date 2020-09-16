# %% import modules
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# %% prepare the data
music_data = pd.read_csv("music.csv")
X = music_data.drop(columns='genre')
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # %% train the model
# # algorithm - decision tree
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# # dump the model into physic file
# joblib.dump(model, 'music-recommender.joblib')
# # after that, you would no longer need to train again, just load it to predict
model = joblib.load('music-recommender.joblib')

# %% predict and caculate the accuracy
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
score
