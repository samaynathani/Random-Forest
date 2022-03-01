import copy
import json

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample

def get_learner(X,y,max_depth=10):
    return DecisionTreeRegressor(max_depth=max_depth).fit(X,y)


def make_trees(X,y,ntrees=100,max_depth=10):
    trees = []
    for i in range(ntrees):
        X_sample, y_sample = resample(X, y)
        trees.append(get_learner(X_sample, y_sample, max_depth))
    return trees


def make_prediction(trees,X):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j]
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).mean().values.flat)


def vote(trees,X):
    votes = np.zeros((len(X),len(trees)))
    for i,tree in enumerate(trees):
        votes[:,i] = tree.predict(X)
    y = pd.DataFrame(votes,index=X.index).mode(axis=1).iloc[:,0].astype(int)
    return y