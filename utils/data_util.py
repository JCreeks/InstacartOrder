#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-1 下午1:36
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import cPickle
import pandas as pd
import numpy as np
from conf.configure import Configure


def load_dataset():
    if not os.path.exists(Configure.processed_train_path):
        train = pd.read_csv(Configure.original_train_path)
    else:
        with open(Configure.processed_train_path, "rb") as f:
            train = cPickle.load(f)

    if not os.path.exists(Configure.processed_test_path):
        test = pd.read_csv(Configure.original_test_path)
    else:
        with open(Configure.processed_test_path, "rb") as f:
            test = cPickle.load(f)
    return train, test


def save_dataset(train, test):
    if train is not None:
        with open(Configure.processed_train_path, "wb") as f:
            cPickle.dump(train, f, -1)

    if test is not None:
        with open(Configure.processed_test_path, "wb") as f:
            cPickle.dump(test, f, -1)

def load_data(path_data = '../input/'):
    '''
    --------------------------------order_product--------------------------------
    * Unique in order_id + product_id
    '''
    priors = pd.read_csv(path_data + 'order_products__prior.csv', 
                     dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    train = pd.read_csv(path_data + 'order_products__train.csv', 
                    dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    '''
    --------------------------------order--------------------------------
    * This file tells us which set (prior, train, test) an order belongs
    * Unique in order_id
    * order_id in train, prior, test has no intersection
    * this is the #order_number order of this user
    '''
    orders = pd.read_csv(path_data + 'orders.csv', 
                         dtype={
                                'order_id': np.int32,
                                'user_id': np.int64,
                                'eval_set': 'category',
                                'order_number': np.int16,
                                'order_dow': np.int8,
                                'order_hour_of_day': np.int8,
                                'days_since_prior_order': np.float32})

    #  order in prior, train, test has no duplicate
    #  order_ids_pri = priors.order_id.unique()
    #  order_ids_trn = train.order_id.unique()
    #  order_ids_tst = orders[orders.eval_set == 'test']['order_id'].unique()
    #  print(set(order_ids_pri).intersection(set(order_ids_trn)))
    #  print(set(order_ids_pri).intersection(set(order_ids_tst)))
    #  print(set(order_ids_trn).intersection(set(order_ids_tst)))

    '''
    --------------------------------product--------------------------------
    * Unique in product_id
    '''
    products = pd.read_csv(path_data + 'products.csv')
    aisles = pd.read_csv(path_data + "aisles.csv")
    departments = pd.read_csv(path_data + "departments.csv")
    sample_submission = pd.read_csv(path_data + "sample_submission.csv")
    order_streaks = pd.read_csv(path_data + "order_streaks.csv")
    
    return priors, train, orders, products, aisles, departments, sample_submission, order_streaks