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
from utils.time_util import tick_tock

def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict, only_new_feature=True):
    with tick_tock("add stats features"):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError("group_columns_list" + "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict)
        the_stats.columns = the_stats.columns.droplevel(0)
        the_stats.reset_index(inplace=True)
        if only_new_feature:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')

    return df_new

def ka_add_groupby_features_n_vs_1(df, group_columns_list, target_columns_list, methods_list, keep_only_stats=True, verbose=1):
    with tick_tock("add stats features", verbose):
        dicts = {"group_columns_list": group_columns_list , "target_columns_list": target_columns_list, "methods_list" :methods_list}

        for k, v in dicts.items():
            try:
                if type(v) == list:
                    pass
                else:
                    raise TypeError(k + "should be a list")
            except TypeError as e:
                print(e)
                raise

        grouped_name = ''.join(group_columns_list)
        target_name = ''.join(target_columns_list)
        combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped[target_name].agg(methods_list).reset_index()
        the_stats.columns = [grouped_name] +                             ['_%s_%s_by_%s' % (grouped_name, method_name, target_name)                              for (grouped_name, method_name, target_name) in combine_name]
        if keep_only_stats:
            return the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
        return df_new


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()
    
    train.loc[:, 'reordered'] = train.reordered.fillna(0)
    #train = train[~pd.isnull(train.reordered)]
    print 'train:', train.shape, ', test:', test.shape
    y_train = train['reordered']
    X_train = train.drop(['eval_set', 'user_id', 'product_id', 'order_id', 'reordered'], axis=1)
    X_test = test.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id'], axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.7, random_state=42)
    
    product_id_test = test.product_id
    order_id_test = test.order_id
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
    X_test.loc[:, 'product_id'] = product_id_test.astype(str)
    X_test.loc[:, 'order_id'] = order_id_test.astype(str)
    submit = ka_add_groupby_features_n_vs_1(X_test[X_test.reordered == 1], 
                                                   group_columns_list=['order_id'],
                                                   target_columns_list= ['product_id'],
                                                   methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
    sample_submission = pd.read_csv("../input/sample_submission.csv")
    submit.columns = sample_submission.columns.tolist()
    submit_final = sample_submission[['order_id']].merge(submit, how='left').fillna('None')
    submit_final.to_csv("../result/jul26_1.csv", index=False)


if __name__ == '__main__':
    main()
