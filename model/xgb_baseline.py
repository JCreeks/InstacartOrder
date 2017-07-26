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



def main():
    print 'load datas...'
    train, test = data_util.load_dataset()

    y_train_all = train['y']
    del train['ID']
    del train['y']
    id_test = test['ID']
    del test['ID']

    print 'train:', train.shape, ', test:', test.shape

    train_r2_scores = []
    val_r2_scores = []
    num_boost_roundses = []

    X_test = test
    df_columns = train.columns.values
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    xgb_params = {
        'eta': 0.005,
        'max_depth': 4,
        'subsample': 0.93,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    for i in range(0, 5):
        random_state = 42 + i
        X_train, X_val, y_train, y_val = train_test_split(train, y_train_all, test_size=0.25, random_state=random_state)

        dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
        dval = xgb.DMatrix(X_val, y_val, feature_names=df_columns)

        y_mean = np.mean(y_train)

        cv_result = xgb.cv(dict(xgb_params, base_score=y_mean),  # base prediction = mean(target)
                           dtrain,
                           num_boost_round=2000,  # increase to have better results (~700)
                           early_stopping_rounds=50,
                           )

        num_boost_rounds = len(cv_result)
        num_boost_roundses.append(num_boost_rounds)
        model = xgb.train(dict(xgb_params, base_score=y_mean), dtrain, num_boost_round=num_boost_rounds)
        train_r2_score = r2_score(dtrain.get_label(), model.predict(dtrain))
        val_r2_score = r2_score(dval.get_label(), model.predict(dval))
        print 'perform {} cross-validate: train r2 score = {}, validate r2 score = {}'.format(i + 1, train_r2_score,
                                                                                              val_r2_score)
        train_r2_scores.append(train_r2_score)
        val_r2_scores.append(val_r2_score)

    print '\naverage train r2 score = {}, average validate r2 score = {}'.format(
        sum(train_r2_scores) / len(train_r2_scores),
        sum(val_r2_scores) / len(val_r2_scores))

    best_num_boost_rounds = sum(num_boost_roundses) // len(num_boost_roundses)
    print 'best_num_boost_rounds =', best_num_boost_rounds
    # train model
    print 'training on total training data...'
    dtrain_all = xgb.DMatrix(train, y_train_all, feature_names=df_columns)
    model = xgb.train(dict(xgb_params, base_score=np.mean(y_train_all)), dtrain_all,
                      num_boost_round=best_num_boost_rounds)

    print 'predict submit...'
    y_pred = model.predict(dtest)
    df_sub = pd.DataFrame({'ID': id_test, 'y': y_pred})
    df_sub.to_csv(Configure.submission_path, index=False)


if __name__ == '__main__':
    main()
