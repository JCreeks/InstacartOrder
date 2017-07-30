import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor

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
from meanF1 import mean_F_score
from sklearn.metrics import make_scorer
#from model_stack.CV import DeepCV

F1 = make_scorer(mean_F_score, greater_is_better=False)

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

def compare_results(df_gt, df_preds):
    
    df_gt_cut = df_gt.loc[df_preds.index]
    
    f1 = []
    for gt, pred in zip(df_gt_cut.sort_index().products, df_preds.sort_index().products):
        lgt = gt.replace("None", "-1").split(' ')
        lpred = pred.replace("None", "-1").split(' ')

        rr = (np.intersect1d(lgt, lpred))
        precision = np.float(len(rr)) / len(lpred)
        recall = np.float(len(rr)) / len(lgt)

        denom = precision + recall
        f1.append(((2 * precision * recall) / denom) if denom > 0 else 0)

    #print(np.mean(f1))
    return(np.mean(f1))

def DeepCV(df_train_gt, train, model, n_folds = 4): 
    train_scores = []
    val_scores = []
    num_boost_roundses = []
    df_columns = train.columns.values
    for fold in range(n_folds):
        train_subset = train[train.user_id % 4 != fold]
        valid_subset = train[train.user_id % 4 == fold]

        X_train = train_subset.drop(['eval_set', 'user_id', 'product_id', 'order_id', 'reordered'], axis=1)
        y_train = train_subset.reordered

        X_val = valid_subset.drop('reordered', axis=1)
        y_val = valid_subset.reordered
        model.train(X_train, y_train, cv_train=False, nrounds=80)

        val_out = X_val[['user_id', 'product_id', 'order_id']]

        lim = .202
        
        y_predict = model.predict(X_val.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis=1))
        val_out.loc[:,'reordered'] = (y_predict > lim).astype(int)
        val_out.loc[:, 'product_id'] = val_out.product_id.astype(str)
        presubmit = ka_add_groupby_features_n_vs_1(val_out[val_out.reordered == 1], 
                                                       group_columns_list=['order_id'],
                                                       target_columns_list= ['product_id'],
                                                       methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)

        presubmit = presubmit.set_index('order_id')
        presubmit.columns = ['products']

        fullfold = pd.DataFrame(index = val_out.order_id.unique())

        fullfold.index.name = 'order_id'
        fullfold['products'] = ['None'] * len(fullfold)

        fullfold.loc[presubmit.index, 'products'] = presubmit.products

        del presubmit

        #print(fold, compare_results(df_train_gt, fullfold))

        #train_score = metric(y_train, model.predict(X_train))
        val_score = compare_results(df_train_gt, fullfold)
        print 'perform {} cross-validate: validate score = {}'.format(fold + 1, val_score)
        #train_scores.append(train_score)
        val_scores.append(val_score)

    print '\n average validate score = {}'.format(
        sum(val_scores) / len(val_scores))


def main():
    print 'load datas...'
    train, test = data_util.load_dataset()
    train = train.sample(frac=.5)
    
    train.loc[:, 'reordered'] = train.reordered.fillna(0)
    #train = train[~pd.isnull(train.reordered)]
    print 'train:', train.shape, ', test:', test.shape
    del test

    # y_train = train['reordered']
    # X_train = train.drop(['eval_set', 'user_id', 'product_id', 'order_id', 'reordered'], axis=1)
    # X_test = test.drop(['eval_set', 'user_id', 'order_id', 'reordered', 'product_id'], axis=1)
    
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.007, random_state=42)
    
    # product_id_test = test.product_id
    # order_id_test = test.order_id
    # del train, test, X_val, y_val

    path_data = "../input/"
    df_train_gt = pd.read_csv(path_data+'train.csv', index_col='order_id')

    xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "error" #"logloss" #"
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
    #model = XGBRegressor(**xgb_params)
    model = XgbWrapper(seed=SEED, params=xgb_params, cv_fold=4)
    # # model.cv_train(X_train, y_train, num_boost_round=2000, nfold=5, early_stopping_rounds=20)

    # #print(model.getScore())
    DeepCV(df_train_gt, train, model, n_folds = 4)


if __name__ == '__main__':
    main()
