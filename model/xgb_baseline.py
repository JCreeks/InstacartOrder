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
    print 'train:', train.shape, ', test:', test.shape
    
    train.loc[:, 'reordered'] = train.reordered.fillna(0)
    y_train = train['reordered']
    X_train = train.drop('reordered', axis=1)
    X_test = test
    
    del train, test

    xgb_params = {
        'eta': 0.005,
        'max_depth': 4,
        'subsample': 0.93,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    SEED = 10
    model = XgbWrapper(seed=SEED, params=xgb_params, cv_train=True, cv_fold=4)
    model.train(X_train, y_train)
    
    y_predict = model.predict(X_test)
    X_test.loc[:,'reordered'] = (y_predict > 0.21).astype(int)
    X_test.loc[:, 'product_id'] = X_test.product_id.astype(str)
    submit = ka_add_groupby_features_n_vs_1(X_test[X_test.reordered == 1], 
                                                   group_columns_list=['order_id'],
                                                   target_columns_list= ['product_id'],
                                                   methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
    submit.columns = sample_submission.columns.tolist()
    submit_final = sample_submission[['order_id']].merge(submit, how='left').fillna('None')
    submit_final.to_csv("../result/jul25_1.csv", index=False)


if __name__ == '__main__':
    main()
