#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-24 下午12:01
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
import numpy as np
# remove warnings
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import Imputer

from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount

from xgboost import XGBRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LassoCV,LassoLarsCV, ElasticNet, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from model_stack.model_wrapper import XgbWrapper, SklearnWrapper, GridCVWrapper
from model_stack.model_stack import TwoLevelModelStacking, ThreeLevelModelStacking

# my own module
from utils import data_util
from conf.configure import Configure
from CV import DeepCV, PrintImportance, StackingEstimator

from sklearn.model_selection import train_test_split
import xgboost as xgb

# my own module
from conf.configure import Configure
from utils import data_util

import gc
import time
from sklearn.model_selection import train_test_split
from model_stack.model_wrapper import XgbWrapper, SklearnWrapper, GridCVWrapper
from utils.time_util import tick_tock

import lightgbm as lgb
import sklearn.metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.sparse import dok_matrix, coo_matrix
from sklearn.utils.multiclass import  type_of_target

from sklearn.metrics import log_loss, make_scorer
Log_loss = make_scorer(log_loss, greater_is_better=False)
F1 = make_scorer(f1_score, greater_is_better=True)

print 'load datas...'
train, test = data_util.load_dataset()

train = train.replace([np.inf, -np.inf], np.nan)
train = train.dropna()
train.loc[:, 'reordered'] = train.reordered.fillna(0)
#train = train[~pd.isnull(train.reordered)]
print 'train:', train.shape, ', test:', test.shape
y_train = train['reordered']
X_train = train.drop(['order_id', 'reordered'], axis=1)
X_test = test.drop(['order_id'], axis=1)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.7, random_state=42)    
# del X_val, y_val

product_id_test = test.product_id
order_id_test = test.order_id
del train, test
gc.collect()

#############################
#nTest = 500
#train = train[:nTest]
#y_train = y_train[:nTest]

#############################

X_train = np.array(X_train)
X_test = np.array(X_test)

rf_params1 = {'max_depth':16, 'n_estimators':250, 'min_samples_leaf':20, 'min_samples_split':60,
              'max_features':.4, 'random_state':5, 'n_jobs':-1} #LB 0.55597

rf_params2 = {'n_estimators': 1000, 'criterion': 'mse', 'max_depth': 8, 'min_samples_split': 2, 'min_samples_leaf':1, 
              'min_weight_fraction_leaf': 0.0, 'max_features': 'auto', 'n_jobs': -1} #LB Score: 0.55199

et_params1 = {'max_depth':13, 'n_estimators':200, 'min_samples_leaf':10,
              'min_samples_split':20, 'max_features':.7, 'random_state':10, 'n_jobs':-1}# 0.55793

et_params2 = {'min_samples_leaf':2, 'max_depth':12, 'n_jobs':-1, 'n_estimators':100, 'max_features':.5}

gb_params1 = {'learning_rate':0.05, 'n_estimators':50, 'min_samples_leaf':60, 'min_samples_split':20, 'subsample':0.8, 'max_features':.4,    
              'max_depth':5}#LB: 0.55387

gb_params2 = {'learning_rate':0.001, 'loss':"huber", 'max_depth':3, 'max_features':0.55, 
              'min_samples_leaf':18, 'min_samples_split':14, 'subsample':0.7}

xgb_params = {'eta': .0917, 'max_depth': 6, 'subsample': 0.76, 'objective': 'reg:logistic',
               'min_child_weight':10, 'colsample_bytree': 0.95, #'n_trees': 950,
               'eval_metric': 'logloss', 'silent': 1}#average validate r2 score = 0.561513621097

lcv_params = {'alphas' : [1, 0.1, 0.001, 0.0005]}#[.1, 1, 10, 100, 1000]}#

llcv_params = {'fit_intercept': True, 'verbose': False, 'max_iter': 500, 'normalize': True, 'precompute': 'auto', 'cv':5,
               'max_n_alphas': 1000, 'n_jobs': -1, 'eps': 2.2204460492503131e-16, 'copy_X': True, 'positive': False}#0.55827

rd_params = {'alpha': .5}#LB: 0.52903

ls_params = {'alpha':  .01}#.0001} #LB: 0.54676

eln_params = {'l1_ratio': 1, 'alpha': 0.00452445, 'max_iter': 32, 'fit_intercept': True, 'normalize': True}

knr_params1 = {'n_neighbors' : 5}

knr_params2 = {'n_neighbors' : 10}

knr_params3 = {'n_neighbors' : 15}

knr_params4 = {'n_neighbors' : 25}

SEED = 0

level_1_models = [XgbWrapper(seed=SEED, params=xgb_params, cv_fold=4), 
                 ]
                
# level_1_models = level_1_models + [SklearnWrapper(clf=KNeighborsRegressor,  params=knr_params1),
#                  SklearnWrapper(clf=KNeighborsRegressor,  params=knr_params2),
#                  SklearnWrapper(clf=KNeighborsRegressor,  params=knr_params3),
#                  SklearnWrapper(clf=KNeighborsRegressor,  params=knr_params4)]

level_1_models = level_1_models + [SklearnWrapper(make_pipeline( ZeroCount(), LassoLarsCV(normalize=True))),#LB 0.55797
                 SklearnWrapper(make_pipeline(StackingEstimator(estimator=LassoLarsCV(normalize=True)),
                 StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001,
                 loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18,
                 min_samples_split=14, subsample=0.7)), LassoLarsCV()))
                                  ]

params_list = [rf_params1, rf_params2, et_params1, et_params2, gb_params1, #gb_params2, 
               rd_params, ls_params, 
               eln_params, 
               lcv_params,
               llcv_params
               ]
   

func_list = [RandomForestRegressor, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesRegressor, 
             GradientBoostingRegressor, #GradientBoostingRegressor, 
             Ridge, Lasso, 
             ElasticNet, 
             LassoCV,
             LassoLarsCV
            ]
level_1_models = level_1_models + \
    list(map(lambda x: SklearnWrapper(clf=x[1], seed=SEED, params=x[0]), zip(params_list, func_list)))

#level_1_models = level_1_models [16:]
et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

rd_params = {
    'alpha': .5
}

ls_params = {
    'alpha': .01#0.005
}

xg = XgbWrapper(seed=SEED, params=xgb_params, cv_fold=4)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)

#level_2_models = [SklearnWrapper(clf=ExtraTreesRegressor,seed=SEED,params={}),
#                 XgbWrapper(seed=SEED, params=xgb_params1)]
# level_2_models = [xg, et, rf, rd, ls,
#                   XgbWrapper(seed=SEED, params=xgb_params3, cv_fold=4)
                 ]

# level_2_models = level_2_models + [SklearnWrapper(make_pipeline( ZeroCount(), LassoLarsCV(normalize=True))),
#                  SklearnWrapper(make_pipeline(StackingEstimator(estimator=LassoLarsCV(normalize=True)),
#                  StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001,
#                  loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18,
#                  min_samples_split=14, subsample=0.7)), LassoLarsCV()))
#                                   ]
    
# xgb_params = {
#     'eta': 0.05,
#     'max_depth': 5,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     'silent': 1
# }

#stacking_model = XgbWrapper(seed=SEED, params=xgb_params, cv_fold=5)
stacking_model = GridCVWrapper(Ridge, seed=SEED, cv_fold=5, params={}, scoring=F1, param_grid = {
           'alpha': [1e-3,5e-3,1e-2,5e-2,1e-1,0.2,0.3,0.4,0.5,0.8,1e0,3,5,7,1e1]})

model_stack = TwoLevelModelStacking(X_train, y_train, X_test, level_2_models, stacking_model=stacking_model, stacking_with_pre_features=False, n_folds=5, random_seed=0, 
                                    scorer = F1)

# model_stack = ThreeLevelModelStacking(train, y_train, test, level_1_models, level_2_models, 
# stacking_model=stacking_model, stacking_with_pre_features=True, n_folds=10, random_seed=0)

predicts, score= model_stack.run_stack_predict()

df_sub = pd.DataFrame({'ID': id_test, 'y': predicts})
df_sub.to_csv(Configure.submission_path+str(score)+'.csv', index=False)


#tmp_predicts, tmp_score= model_stack.run_stack_predict()
#try:
#    predicts = np.add(predicts, tmp_predicts)
#    score = score + tmp_score
#except:
#    predicts = tmp_predicts
#    score = tmp_score
#predicts = predicts/len(seedList)
#score = score/len(seedList)
    
    
    
