import copy
import json

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from math import log

def entropy(y):
    e = 0
    val_counts = y.value_counts()
    for i in list(val_counts.index):
        py = val_counts[i] / len(y)
        log2_py = log(py, 2)
        res = -1*py*log2_py
        e+=res
    return e

def gain(y,x):
    g = 0
    val_counts = x.value_counts()
    for i in list(val_counts.index):
        g += ((val_counts[i] / len(y)) * entropy(y[x == i]))
    return entropy(y) - g

def gain_ratio(y,x):
    g = gain(y, x)
    return g/entropy(y)

def select_split(X,y):
    col = None
    gr = float("-inf")
    for c in X.columns:
        g = gain_ratio(y, X[c])
        if g > gr:
            gr, col = g, c
    return col,gr

def make_tree(X,y):
    tree = {}
    if len(X.columns) == 0:
        return y.value_counts().idxmax()
    elif len(y.unique()) == 1:
        return y.unique()[0]
    col, gr = select_split(X, y)
    if gr < 0.00001:
        return y.value_counts().idxmax()
    tree[col] = {}
    for ux in X[col].unique():
        tree[col][ux] = {}
        y2 = y[X[col] == ux]
        X2 = X.copy()
        X2 = X2[X2[col] == ux]
        X2 = X2.drop(col, axis=1)
        tree[col][ux] = make_tree(X2, y2)
    return tree

# if you want to print like me :)
def print_tree(tree):
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            if type(tree) == np.int64:
                return int(tree)
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))


def _generate_rules(tree, rule, rules):
    if not isinstance(tree, dict):
        rules.append(rule + [tree])
    else:
        for k, v in tree.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    _generate_rules(v2, rule + [(k, k2)], rules)


def generate_rules(tree):
    rules = []
    rule = []
    _generate_rules(tree, rule,  rules)
    return rules

def split_col(x, y):
    x2 = list(x.copy())
    save_x = x.copy()
    x2.sort()
    splits = []
    for i in range(0, len(x2)-1):
        splits.append((x2[i] + x2[i+1]) / 2)
    for split in splits:
        x = x.apply(lambda x: True if x < split else False)
        g = gain_ratio(y, x)
        if g > gr:
            gr, col = g, c
            
        x = save_x.copy()
    return g, c

def select_split2(X,y):
    col = None
    gr = float("-inf")
    best_split = None
    for c in X.columns:
        if X[c].dtype in ['int64','float64']:
            
            save_x = X[c].copy()
            x2 = list(X[c].copy())
            x2.sort()
            
            splits = []
            for i in range(0, len(x2)-1):
                splits.append((x2[i] + x2[i+1]) / 2)
                
            for split in splits:
                save_x = save_x.apply(lambda x: 'True' if x < split else 'False')
                g = gain_ratio(y, save_x)
                if g > gr:
                    gr, col, best_split = g, c, split
                save_x = X[c].copy()
        else:
            g = gain_ratio(y, X[c])
            if g > gr:
                gr, col, best_split = g, c, None
    newname = col
    if best_split:
        newname = '%s<%.2f'%(col, best_split)
        X.rename(columns={col:newname}, inplace=True)
        X[newname] = X[newname].apply(lambda x: "True" if x < best_split else "False")
    return newname ,gr

def make_tree2(X,y,min_split_count=5):
    tree = {}
    if len(X.columns) == 0:
        return y.value_counts().idxmax()
    if len(y.unique()) == 1:
        return y.unique()[0]
    if len(X) < min_split_count:
        return y.value_counts().idxmax()
    col, gr = select_split2(X, y)
    if gr < 0.001:
        return y.value_counts().idxmax()
    tree[col] = {}
    for ux in X[col].unique():
        tree[col][ux] = {}
        y2 = y[X[col] == ux]
        X2 = X.copy()
        X2 = X2[X2[col] == ux]
        X2 = X2.drop(col, axis=1)
        tree[col][ux] = make_tree2(X2, y2, min_split_count)
    return tree


def make_prediction(rules,x,default):
    
    predict = {}
    for idx in x.index:
        predict[idx] = x[idx]
    
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
