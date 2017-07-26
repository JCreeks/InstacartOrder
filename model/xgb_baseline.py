import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# my own module
from conf.configure import Configure
from utils import data_util

import gc
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from model_stack.model_wrapper import XgbWrapper, SklearnWrapper, GridCVWrapper

def main():
    print 'load datas...'
    train, test = data_util.load_dataset()
    
    #train.loc[:, 'reordered'] = train.reordered.fillna(0)
    train = train[~pd.isnull(train.reordered)]
    print 'train:', train.shape, ', test:', test.shape
    y_train = train['reordered']
    X_train = train.drop(['eval_set', 'user_id', 'product_id', 'order_id', 'reordered'], axis=1)
    X_test = test.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id'], axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.8, random_state=42)
    
    id_test = test.product_id
    del train, test

    xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
    ,"silent"           :1
    }
    
    SEED = 10
    model = XgbWrapper(seed=SEED, params=xgb_params, cv_fold=4)
    model.train(X_train, y_train, cv_train=True)
    
    y_predict = model.predict(X_test)
    X_test.loc[:,'reordered'] = (y_predict > 0.21).astype(int)
    X_test.loc[:, 'product_id'] = id_test.astype(str)
    submit = ka_add_groupby_features_n_vs_1(X_test[X_test.reordered == 1], 
                                                   group_columns_list=['order_id'],
                                                   target_columns_list= ['product_id'],
                                                   methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
    submit.columns = sample_submission.columns.tolist()
    submit_final = sample_submission[['order_id']].merge(submit, how='left').fillna('None')
    submit_final.to_csv("../result/jul26_1.csv", index=False)


if __name__ == '__main__':
    main()
