#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-24 下午1:32
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

class TwoLevelModelStacking(object):
    """两层的 model stacking"""

    def __init__(self, train, y_train, test,
                 models, stacking_model, 
                 scorer,
                 stacking_with_pre_features=True, n_folds=5, random_seed=0):
        self.train = train
        self.y_train = y_train
        self.test = test
        self.n_folds = n_folds
        self.models = models
        self.stacking_model = stacking_model

        # stacking_with_pre_features 指定第二层 stacking 是否使用原始的特征
        self.stacking_with_pre_features = stacking_with_pre_features

        self.ntrain = train.shape[0]
        self.ntest = test.shape[0]
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        self.scorer = scorer

    def run_out_of_folds(self, clf):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.n_folds, self.ntest))

        for i, (train_index, test_index) in enumerate(self.kfold.split(self.train)):
            #print 'fold-{}: train: {}, test: {}'.format(i, train_index, test_index)
            x_tr = self.train[train_index]
            y_tr = self.y_train[train_index]
            x_te = self.train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(self.test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def run_stack_predict(self):
        if self.stacking_with_pre_features:
            x_train = self.train
            x_test = self.test
        #else:
        #    x_train = np.empty((self.ntrain, self.train.shape[1]))
        #    x_test = np.empty((self.ntest, self.test.shape[1]))

        # run level-1 out-of-folds
        for model in self.models:
            oof_train, oof_test = self.run_out_of_folds(model)
            print("{}-1stCV: {}".format(model, self.scorer(self.y_train, oof_train)))
            try:
                x_train = np.concatenate((x_train, oof_train), axis=1)
                x_test = np.concatenate((x_test, oof_test), axis=1)
            except:
                x_train = oof_train
                x_test = oof_test

        # run level-2 stacking
        # best_nrounds, cv_mean, cv_std = self.stacking_model.cv_train(x_train, self.y_train)
        # self.stacking_model.nrounds = best_nrounds
        # print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
            
        self.stacking_model.train(x_train, self.y_train)
       
        # stacking predict
        predicts = self.stacking_model.predict(x_test)
        score = self.stacking_model.getScore()
        print("stackingCV: {}".format(score))
        return predicts, score

    
class ThreeLevelModelStacking(object):
    """three layer model stacking"""

    def __init__(self, train, y_train, test,
                 firstLevelModels, secondLevelModels, stacking_model, 
                 stacking_with_pre_features=True, n_folds=5, random_seed=0):
        self.train = train
        self.y_train = y_train
        self.test = test
        self.n_folds = n_folds
        self.models = firstLevelModels
        self.secondLevelModels = secondLevelModels
        self.stacking_model = stacking_model

        # stacking_with_pre_features 指定第二层 stacking 是否使用原始的特征
        self.stacking_with_pre_features = stacking_with_pre_features

        self.ntrain = train.shape[0]
        self.ntest = test.shape[0]
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    def run_out_of_folds(self, clf):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.n_folds, self.ntest))

        for i, (train_index, test_index) in enumerate(self.kfold.split(self.train)):
            #print 'fold-{}: train: {}, test: {}'.format(i, train_index, test_index)
            x_tr = self.train[train_index]
            y_tr = self.y_train[train_index]
            x_te = self.train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(self.test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def run_stack_predict(self):
        if self.stacking_with_pre_features:
            x_train = self.train
            x_test = self.test
        #else:
        #    x_train = np.empty((self.ntrain, self.train.shape[1]))
        #    x_test = np.empty((self.ntest, self.test.shape[1]))

        # run level-1 out-of-folds
        for model in self.models:
            oof_train, oof_test = self.run_out_of_folds(model)
            #print(oof_train)
            print("{}-1stCV: {}".format(model, r2_score(self.y_train, oof_train)))
            try:
                x_train = np.concatenate((x_train, oof_train), axis=1)
                x_test = np.concatenate((x_test, oof_test), axis=1)
            except:
                x_train = oof_train
                x_test = oof_test
        
        train_to_save = x_train
        test_to_save = x_test
        
        # run level-2 out-of-folds
        self.train = x_train
        self.test = x_test
        
        #x_train = np.empty((self.ntrain, self.train.shape[1]))
        #x_test = np.empty((self.ntest, self.test.shape[1]))
            
        for model in self.secondLevelModels:
            oof_train, oof_test = self.run_out_of_folds(model)
            print("{}-2ndCV: {}".format(model, r2_score(self.y_train, oof_train)))
            try:
                x_train = np.concatenate((x_train, oof_train), axis=1)
                x_test = np.concatenate((x_test, oof_test), axis=1)
            except:
                x_train = oof_train
                x_test = oof_test
            
        # run lesvel-3 stacking
        #xgbWrapper only
        # best_nrounds, cv_mean, cv_std = self.stacking_model.cv_train(x_train, self.y_train)
        # self.stacking_model.nrounds = best_nrounds
        # print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
        #xgbWrapper only
        
        self.stacking_model.train(x_train, self.y_train)
       

        # stacking predict
        predicts = self.stacking_model.predict(x_test)
        score = self.stacking_model.getScore()
        
        # print("score={}".format(score))
        # score = 1 - score**2/self.y_train.var()
        
        #pd.DataFrame(train_to_save).to_csv("1stLayerX_train_{}.csv".format(score))
        #pd.DataFrame(test_to_save).to_csv("1stLayerX_test_{}.csv".format(score))
        
        print("stackingCV: {}".format(score))
        return predicts, score

class TwoLevelModelStacking_new(object):
    """两层的 model stacking"""

    def __init__(self, train, y_train, test,
                 models, stacking_model, 
                 stacking_with_pre_features=True, n_folds=5, random_seed=0, newFeat_train=pd.DataFrame(),
                 newFeat_test=pd.DataFrame(), stacking_with_new_features=True):
        self.train = train
        self.y_train = y_train
        self.test = test
        self.n_folds = n_folds
        self.models = models
        self.stacking_model = stacking_model
        self.newFeat_train = newFeat_train
        self.newFeat_test = newFeat_test

        # stacking_with_pre_features 指定第二层 stacking 是否使用原始的特征
        self.stacking_with_pre_features = stacking_with_pre_features
        self.stacking_with_new_features = stacking_with_new_features

        self.ntrain = train.shape[0]
        self.ntest = test.shape[0]
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    def run_out_of_folds(self, clf):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.n_folds, self.ntest))

        for i, (train_index, test_index) in enumerate(self.kfold.split(self.train)):
            #print 'fold-{}: train: {}, test: {}'.format(i, train_index, test_index)
            x_tr = self.train[train_index]
            y_tr = self.y_train[train_index]
            x_te = self.train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(self.test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def run_stack_predict(self):
        if self.stacking_with_pre_features:
            x_train = self.train
            x_test = self.test
        if self.stacking_with_new_features:
            try:
                x_train = np.concatenate((x_train, self.newFeat_train), axis=1)
                x_test = np.concatenate((x_test, self.newFeat_test), axis=1)
            except:
                x_train = self.newFeat_train
                x_test = self.newFeat_test

        # run level-1 out-of-folds
        for model in self.models:
            oof_train, oof_test = self.run_out_of_folds(model)
            print("{}-1stCV: {}".format(model, r2_score(self.y_train, oof_train)))
            try:
                x_train = np.concatenate((x_train, oof_train), axis=1)
                x_test = np.concatenate((x_test, oof_test), axis=1)
            except:
                x_train = oof_train
                x_test = oof_test

        # # run level-2 stacking
        best_nrounds, cv_mean, cv_std = self.stacking_model.cv_train(x_train, self.y_train)
        self.stacking_model.nrounds = best_nrounds
        print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
            
        self.stacking_model.train(x_train, self.y_train)
       
        # stacking predict
        predicts = self.stacking_model.predict(x_test)
        score = self.stacking_model.getScore()
        print("stackingCV: {}".format(score))
        return predicts, score

class ThreeLevelModelStacking_new(object):
    """three layer model stacking"""

    def __init__(self, train, y_train, test,
                 firstLevelModels, secondLevelModels, stacking_model, 
                 stacking_with_pre_features=True, n_folds=5, random_seed=0, newFeat_train=pd.DataFrame(),
                 newFeat_test=pd.DataFrame(), stacking_with_new_features=True):
        self.train = train
        self.y_train = y_train
        self.test = test
        self.n_folds = n_folds
        self.models = firstLevelModels
        self.secondLevelModels = secondLevelModels
        self.stacking_model = stacking_model
        self.newFeat_train = newFeat_train
        self.newFeat_test = newFeat_test

        # stacking_with_pre_features 指定第二层 stacking 是否使用原始的特征
        self.stacking_with_pre_features = stacking_with_pre_features
        self.stacking_with_new_features = stacking_with_new_features

        self.ntrain = train.shape[0]
        self.ntest = test.shape[0]
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    def run_out_of_folds(self, clf):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.n_folds, self.ntest))

        for i, (train_index, test_index) in enumerate(self.kfold.split(self.train)):
            #print 'fold-{}: train: {}, test: {}'.format(i, train_index, test_index)
            x_tr = self.train[train_index]
            y_tr = self.y_train[train_index]
            x_te = self.train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(self.test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def run_stack_predict(self):
        if self.stacking_with_pre_features:
            x_train = self.train
            x_test = self.test
       
        if self.stacking_with_new_features:
            try:
                x_train = np.concatenate((x_train, self.newFeat_train), axis=1)
                x_test = np.concatenate((x_test, self.newFeat_test), axis=1)
            except:
                x_train = self.newFeat_train
                x_test = self.newFeat_test

        # run level-1 out-of-folds
        for model in self.models:
            oof_train, oof_test = self.run_out_of_folds(model)
            #print(oof_train)
            print("{}-1stCV: {}".format(model, r2_score(self.y_train, oof_train)))
            try:
                x_train = np.concatenate((x_train, oof_train), axis=1)
                x_test = np.concatenate((x_test, oof_test), axis=1)
            except:
                x_train = oof_train
                x_test = oof_test
        
        train_to_save = x_train
        test_to_save = x_test
        
        # run level-2 out-of-folds
        self.train = x_train
        self.test = x_test
        
        #x_train = np.empty((self.ntrain, self.train.shape[1]))
        #x_test = np.empty((self.ntest, self.test.shape[1]))
            
        for model in self.secondLevelModels:
            oof_train, oof_test = self.run_out_of_folds(model)
            print("{}-2ndCV: {}".format(model, r2_score(self.y_train, oof_train)))
            try:
                x_train = np.concatenate((x_train, oof_train), axis=1)
                x_test = np.concatenate((x_test, oof_test), axis=1)
            except:
                x_train = oof_train
                x_test = oof_test
            
        # run level-3 stacking
        #xgbWrapper only
        best_nrounds, cv_mean, cv_std = self.stacking_model.cv_train(x_train, self.y_train)
        self.stacking_model.nrounds = best_nrounds
        print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))
        #xgbWrapper only
        
        self.stacking_model.train(x_train, self.y_train)
       

        # stacking predict
        predicts = self.stacking_model.predict(x_test)
        score = self.stacking_model.getScore()
        
        #pd.DataFrame(train_to_save).to_csv("1stLayerX_train_{}.csv".format(score))
        #pd.DataFrame(test_to_save).to_csv("1stLayerX_test_{}.csv".format(score))
        
        print("stackingCV: {}".format(score))
        return predicts, score