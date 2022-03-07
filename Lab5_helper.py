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

def make_trees_boost(Xtrain, Xval, ytrain, yval, max_ntrees=100,max_depth=2):
    #Xtrain, Xval, ytrain, yval = train_test_split(X,y,test_size=val_frac,shuffle=True)
    trees = []
    yval_pred = None
    ytrain_pred = None
    train_RMSEs = [] # the root mean square errors for the validation dataset
    val_RMSEs = [] # the root mean square errors for the validation dataset
    ytrain_orig = copy.deepcopy(ytrain)
    for i in range(max_ntrees):
        if yval_pred is None:
            tree = get_learner(Xtrain, ytrain, max_depth)
            ytrain_pred = tree.predict(Xtrain)
            yval_pred = tree.predict(Xval)
        else:
            residuals = ytrain - ytrain_pred
            tree = get_learner(Xtrain, residuals, max_depth)
            ytrain_pred += tree.predict(Xtrain)
            yval_pred += tree.predict(Xval)
        trees.append(tree)
        train_RMSEs.append(np.sqrt(((ytrain_pred-ytrain)**2).sum()/len(ytrain)))
        val_RMSEs.append(np.sqrt(((yval_pred-yval)**2).sum()/len(yval)))
    
    return trees,train_RMSEs,val_RMSEs

def cut_trees(trees,val_RMSEs):
    # Your solution here that finds the minimum validation score and uses only the trees up to that
    idx = val_RMSEs.index(min(val_RMSEs))
    trees = trees[:idx]
    return trees

def make_prediction_boost(trees,X):
    tree_predictions = []
    for tree in trees:
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).sum().values.flat)

