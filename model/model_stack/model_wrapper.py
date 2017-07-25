#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-24 下午1:30
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import xgboost as xgb
from abc import ABCMeta, abstractmethod
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

R2 = make_scorer(r2_score, greater_is_better=True)

def RMSLE_(y_val, y_val_pred):
    return np.sqrt(np.mean((np.log(y_val+1)-np.log(y_val_pred+1))**2))
RMSLE = make_scorer(RMSLE_, greater_is_better=False) 

def RMSE_(y_val, y_val_pred):
    return np.sqrt(np.mean((y_val-y_val_pred)**2))
RMSE = make_scorer(RMSE_, greater_is_better=False)


class BaseWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        return None


class SklearnWrapper(BaseWrapper):
    def __init__(self, clf, seed=None, params={}):
        if (seed):
            params['random_state'] = seed
        if (len(params)):
            self.clf = clf(**params)
        else:
            self.clf = clf

    def train(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        return self.clf.predict(x)


class XgbWrapper(BaseWrapper):
    def __init__(self, seed=0, params={}, cv_fold=5):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)
        self.score = 0
        self.cv_fold = cv_fold

    def train(self, x, y, plotImp=False):
        best_nrounds, cv_mean, cv_std = self.cv_train(x, y, nfold=self.cv_fold)
        self.nrounds = best_nrounds
        #print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
        dtrain = xgb.DMatrix(x, label=y)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)
        if (plotImp):
            fig, ax = plt.subplots(1, 1, figsize=(8, 13))
            xgb.plot_importance(self.gbdt, max_num_features=50, height=0.5, ax=ax)

    def cv_train(self, x, y, num_boost_round=1000, nfold=5, early_stopping_rounds=40):
        dtrain = xgb.DMatrix(x, label=y)
        res = xgb.cv(self.param, dtrain, num_boost_round=num_boost_round, nfold=nfold,
                     early_stopping_rounds=early_stopping_rounds, verbose_eval=None, show_stdv=True)

        best_nrounds = res.shape[0] - 1
        cv_mean = res.iloc[-1, 0]
        cv_std = res.iloc[-1, 1]
        self.score = cv_mean
        #print("score={}".format(self.score ))
        self.score = 1 - self.score**2/np.var(y)
        return best_nrounds, cv_mean, cv_std

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))
    def getScore(self):
        return self.score
    
class GridCVWrapper(BaseWrapper):
    def __init__(self, clf, seed=0, cv_fold=5, params=None, scoring=R2, param_grid = {
            'alpha': [1e-3,5e-3,1e-2,5e-2,1e-1,0.2,0.3,0.4,0.5,0.8,1e0,3,5,7,1e1],
        }):
        if (not params):
            params = {}
        params['random_state'] = seed
        self.grid = GridSearchCV(estimator=clf(**params), param_grid=param_grid, n_jobs=-1, cv=cv_fold, scoring=scoring)
        self.score = 0

    def train(self, x, y):
        self.grid.fit(x, y)
        self.score = self.grid.best_score_

    def predict(self, x):
        return self.grid.predict(x)
    def getScore(self):
        return self.score
