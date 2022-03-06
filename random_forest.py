import copy
import json
import random

from math import log

import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.base import BaseEstimator, RegressorMixin

class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self):
        pass

    def _entropy(self, y):
        e = 0
        val_counts = y.value_counts()
        for i in list(val_counts.index):
            py = val_counts[i] / len(y)
            log2_py = log(py, 2)
            res = -1*py*log2_py
            e+=res
        return e

    def _gain(self, y,x):
        g = 0
        val_counts = x.value_counts()
        for i in list(val_counts.index):
            g += ((val_counts[i] / len(y)) * self._entropy(y[x == i]))
        return self._entropy(y) - g

    def _gain_ratio(self, y,x):
        g = self._gain(y, x)
        return g/self._entropy(y)

    def _select_split(self,X,y):
        col = None
        gr = float("-inf")
        for c in X.columns:
            g = self._gain_ratio(y, X[c])
            if g > gr:
                gr, col = g, c
        return col,gr

    def _make_tree(self,X,y, n_features):
        tree = {}
        if len(X.columns) == 0:
            return y.value_counts().idxmax()
        elif len(y.unique()) == 1:
            return y.unique()[0]
        if n_features is None or n_features >= len(X.columns):
            n_features = len(X.columns)
        feature_index = random.sample(range(len(X.columns)), n_features)
        features = []
        for i in range(len(X.columns)):
            if i in feature_index:
                features.append(X.columns[i])
        X2 = X.loc[:,features]
        col, gr = self._select_split(X2, y)
        if gr < 0.00001:
            return y.value_counts().idxmax()
        tree[col] = {}
        for ux in X[col].unique():
            tree[col][ux] = {}
            y2 = y[X[col] == ux]
            X2 = X2[X2[col] == ux]
            X3 = X[X[col] == ux]
            X3 = X3.drop(col, axis=1)
            tree[col][ux] = self._make_tree(X3, y2, n_features)
        return tree
    
    def fit(self, X, y, n_features=None):
        self._tree = self._make_tree(X, y, n_features)
        self._rules = self._generate_rules()
        self._default = y.mode().get(0)
        return self
    
    def _generate_rules(self):
        rules = []
        rule = []
        self._generate_rules_helper(self._tree, rule,  rules)
        return rules
    
    def _generate_rules_helper(self, tree, rule, rules):
        if not isinstance(tree, dict):
            rules.append(rule + [tree])
        else:
            for k, v in tree.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        self._generate_rules_helper(v2, rule + [(k, k2)], rules)
                    
    def _make_prediction(self, rules, x, default):
    
        predict = {}
        for idx in x.index:
            predict[idx] = [idx]

        for rule in rules:
            i = 0
            for r in rule[:-1]:
                if "<" in r[0]:
                    tokens = r[0].split("<")
                    if tokens[0] in predict:
                        if predict[tokens[0]] < float(tokens[1]) and r[1] == "True":
                            i+=1
                        if predict[tokens[0]] >= float(tokens[1]) and r[1] == "False":
                            i+=1
                    if tokens[0] not in predict:
                            i+=1
                else:
                    if r[0] in predict and r[1] == predict[r[0]]:
                        i+=1
                    if r[0] not in predict:
                        i+=1
            if i == len(rule)-1:
                return rule[-1]
        return(default)
    
    def predict(self, X):
        predict = {}
        for idx in X.index:
            predict[idx] = X.loc[idx]
        
        for rule in self._rules:
            i = 0
            for r in rule[:-1]:
                if "<" in r[0]:
                    tokens = r[0].split("<")
                    if tokens[0] in predict:
                        if predict[tokens[0]] < float(tokens[1]) and r[1] == "True":
                            i+=1
                        if predict[tokens[0]] >= float(tokens[1]) and r[1] == "False":
                            i+=1
                    if tokens[0] not in predict:
                            i+=1
                else:
                    if r[0] in predict and r[1] == predict[r[0]]:
                        i+=1
                    if r[0] not in predict:
                        i+=1
            if i == len(rule)-1:
                return rule[-1]
        return(self._default)

    
    def score(self, X):
        assert type(X) == pd.DataFrame
        X.apply(lambda x: self._make_prediction(self._rules, x, self._default))
        

        


def get_learner(X,y,max_depth=1):
    return DecisionTreeRegressor().fit(X,y)


def make_trees(X, y, n_features=None, ntrees=100, max_depth=10):
    if n_features is None:
        n_features = len(X.columns)
    trees = []
    for i in range(ntrees):
        X_sample, y_sample = resample(X, y)
        trees.append(get_learner(X_sample, y_sample, max_depth))
    return trees


def make_prediction(trees,X):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j][0]
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).mean().values.flat)


def vote(trees,X):
    votes = np.zeros((len(X),len(trees)))
    for i,tree in enumerate(trees):
        votes[:,i] = tree.predict(X)
    y = pd.DataFrame(votes,index=X.index).mode(axis=1).iloc[:,0].astype(int)
    return y