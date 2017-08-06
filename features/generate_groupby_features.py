#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from sklearn.preprocessing import LabelEncoder
import cPickle
import pandas as pd
import numpy as np
# remove warnings
import warnings
import gc
warnings.filterwarnings('ignore')

from utils import data_util
from conf.configure import Configure
from utils.time_util import tick_tock
from utils.catgorize_util import transform_categorical_data

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

# def transform_categorical_data(data, categorical_list):                   
#     for f in categorical_list:
#         encoder = LabelEncoder()
#         encoder.fit(list(data[f])) 
#         data[f] = encoder.transform(data[f].ravel())


def main():
    print 'load datas...'
    priors, train, orders, products, aisles, departments, sample_submission, order_streaks = data_util.load_data()

    groupby_features_train = pd.DataFrame()
    groupby_features_test = pd.DataFrame()

    if (not os.path.exists(Configure.groupby_features_train_path)) or \
            (not os.path.exists(Configure.groupby_features_test_path)):

        # # Product part

        # Products information ----------------------------------------------------------------
        # add order information to priors set
        priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')

        # create new variables
        ## _user_buy_product_times: 用户是第几次购买该商品
        priors_orders_detail.loc[:,'_user_buy_product_times'] = priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1
        # _prod_tot_cnts: 该商品被购买的总次数,表明被喜欢的程度
        # _reorder_tot_cnts_of_this_prod: 这件商品被再次购买的总次数
        ### 我觉得下面两个很不好理解，考虑改变++++++++++++++++++++++++++
        # _prod_order_once: 该商品被购买一次的总次数
        # _prod_order_more_than_once: 该商品被购买一次以上的总次数
        agg_dict = {'user_id':{'_prod_tot_cnts':'count'}, 
                    'reordered':{'_prod_reorder_tot_cnts':'sum'}, 
                    '_user_buy_product_times': {'_prod_buy_first_time_total_cnt':lambda x: sum(x==1),
                                                '_prod_buy_second_time_total_cnt':lambda x: sum(x==2)}}
        prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'], agg_dict)

        # _prod_reorder_prob: 这个指标不好理解
        # _prod_reorder_ratio: 商品复购率
        prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
        prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
        prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt

        # # User Part

        # _user_total_orders: 用户的总订单数
        # 可以考虑加入其它统计指标++++++++++++++++++++++++++
        # _user_sum_days_since_prior_order: 距离上次购买时间(和),这个只能在orders表里面计算，priors_orders_detail不是在order level上面unique
        # _user_mean_days_since_prior_order: 距离上次购买时间(均值)
        agg_dict_2 = {'order_number':{'_user_total_orders':'max'},
                      'days_since_prior_order':{'_user_sum_days_since_prior_order':'sum', 
                                                '_user_mean_days_since_prior_order': 'mean'}}
        users = ka_add_groupby_features_1_vs_n(orders[orders.eval_set == 'prior'], ['user_id'], agg_dict_2)

        # _user_reorder_ratio: reorder的总次数 / 第一单后买后的总次数
        # _user_total_products: 用户购买的总商品数
        # _user_distinct_products: 用户购买的unique商品数
        # agg_dict_3 = {'reordered':
        #               {'_user_reorder_ratio': 
        #                lambda x: sum(priors_orders_detail.ix[x.index,'reordered']==1)/
        #                          sum(priors_orders_detail.ix[x.index,'order_number'] > 1)},
        #               'product_id':{'_user_total_products':'count', 
        #                             '_user_distinct_products': lambda x: x.nunique()}}
        # us = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['user_id'], agg_dict_3)

        us = pd.concat([
        priors_orders_detail.groupby('user_id')['product_id'].count().rename('_user_total_products'),
        priors_orders_detail.groupby('user_id')['product_id'].nunique().rename('_user_distinct_products'),
        (priors_orders_detail.groupby('user_id')['reordered'].sum() /
        priors_orders_detail[priors_orders_detail['order_number'] > 1].groupby('user_id')['order_number'].count()).rename('_user_reorder_ratio')
        ], axis=1).reset_index()
        users = users.merge(us, how='inner')

        # 平均每单的商品数
        # 每单中最多的商品数，最少的商品数++++++++++++++
        users['_user_average_basket'] = users._user_total_products / users._user_total_orders

        us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
        us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)

        users = users.merge(us, how='inner')


        # # Database Part

        # 这里应该还有很多变量可以被添加
        # _up_order_count: 用户购买该商品的次数
        # _up_first_order_number: 用户第一次购买该商品所处的订单数
        # _up_last_order_number: 用户最后一次购买该商品所处的订单数
        # _up_average_cart_position: 该商品被添加到购物篮中的平均位置
        agg_dict_4 = {'order_number':{'_up_order_count': 'count', 
                                      '_up_first_order_number': 'min', 
                                      '_up_last_order_number':'max'}, 
                      'add_to_cart_order':{'_up_average_cart_position': 'mean'}}

        data = ka_add_groupby_features_1_vs_n(df=priors_orders_detail, 
                                                              group_columns_list=['user_id', 'product_id'], 
                                                              agg_dict=agg_dict_4)

        data = data.merge(prd, how='inner', on='product_id').merge(users, how='inner', on='user_id')
        # 该商品购买次数 / 总的订单数
        # 最近一次购买商品 - 最后一次购买该商品
        # 该商品购买次数 / 第一次购买该商品到最后一次购买商品的的订单数
        data['_up_order_rate'] = data._up_order_count / data._user_total_orders
        data['_up_order_since_last_order'] = data._user_total_orders - data._up_last_order_number
        data['_up_order_rate_since_first_order'] = data._up_order_count / (data._user_total_orders - data._up_first_order_number + 1)

        # add user_id to train set
        train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')
        data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')
        data = pd.merge(data, products[['product_id', 'aisle_id', 'department_id']], how='left', on='product_id')
        transform_categorical_data(data, ['aisle_id', 'department_id'])
        data = data.merge(order_streaks[['user_id', 'product_id', 'order_streak']], on=['user_id', 'product_id'], how='left')


        # release Memory
        # del train, prd, users
        # gc.collect()
        # release Memory
        #del priors_orders_detail
        del orders, order_streaks
        gc.collect()

        starting_size = sys.getsizeof(data)
        i = 0
        for c, dtype in zip(data.columns, data.dtypes):
            if 'int' in str(dtype):
                if min(data[c]) >=0:
                    max_int =  max(data[c])
                    if max_int <= 255:
                        data[c] = data[c].astype(np.uint8)
                    elif max_int <= 65535:
                        data[c] = data[c].astype(np.uint16)
                    elif max_int <= 4294967295:
                        data[c] = data[c].astype(np.uint32)
                    i += 1
        print("Number of colums adjusted: {}\n".format(i))
        ## Changing known reorderd col to smaller int size
        data['reordered'] = np.nan_to_num(data['reordered']).astype(np.uint8)
        data['reordered'][data['reordered']==0] = np.nan
        print("Reduced size {:.2%}".format(float(sys.getsizeof(data))/float(starting_size)))


        # # Create Train / Test
        train = data.loc[data.eval_set == "train",:]
        #train.drop(['eval_set', 'user_id', 'product_id', 'order_id'], axis=1, inplace=True)
        #train.loc[:, 'reordered'] = train.reordered.fillna(0)

        test = data.loc[data.eval_set == "test",:]
        #test.drop(['eval_set', 'user_id', 'product_id', 'order_id', 'reordered'], axis=1, inplace=True)
        #groupby_features_train = train
        #groupby_features_test = test

        # with open(Configure.groupby_features_train_path, "wb") as f:
        #     cPickle.dump(groupby_features_train, f, -1)
        # with open(Configure.groupby_features_test_path, "wb") as f:
        #     cPickle.dump(groupby_features_test, f, -1)

        print 'train:', train.shape, ', test:', test.shape
        print("Save data...")
        data_util.save_dataset(train, test)
        
    else:
        with open(Configure.groupby_features_train_path, "rb") as f:
            groupby_features_train = cPickle.load(f)
        with open(Configure.groupby_features_test_path, "rb") as f:
            groupby_features_test = cPickle.load(f)

        # merge
        # train = pd.merge(train, groupby_features_train, how='left', on='ID')
        # test = pd.merge(test, groupby_features_test, how='left', on='ID')

    #train, test = data_util.load_dataset()
    # print 'train:', train.shape, ', test:', test.shape
    # print("Save data...")
    # data_util.save_dataset(train, test)


if __name__ == '__main__':
    print '========== perform groupby features =========='
    main()


# # Thanks for sharing the python version!

# # If you are looking to reduce the memory size, I would suggest adding the following after the data data-frame is complete.

# import sys
# starting_size = sys.getsizeof(data)
# i = 0
# for c, dtype in zip(data.columns, data.dtypes):
#     if 'int' in str(dtype):
#         if min(data[c]) >=0:
#             max_int =  max(data[c])
#             if max_int <= 255:
#                 data[c] = data[c].astype(np.uint8)
#             elif max_int <= 65535:
#                 data[c] = data[c].astype(np.uint16)
#             elif max_int <= 4294967295:
#                 data[c] = data[c].astype(np.uint32)
#             i += 1
# print("Number of colums adjusted: {}\n".format(i))
# ## Changing known reorderd col to smaller int size
# data['reordered'] = np.nan_to_num(data['reordered']).astype(np.uint8)
# data['reordered'][data['reordered']==0] = np.nan
# print("Reduced size {:.2%}".format(float(sys.getsizeof(data))/float(starting_size)))

# # Thanks for sharing!

# # You can improve performance of us dataframe calculation with following code:

# us = pd.concat([
#     priors_orders_detail.groupby('user_id')['product_id'].count().rename('_user_total_products'),
#     priors_orders_detail.groupby('user_id')['product_id'].nunique().rename('_user_distinct_products'),
#     (priors_orders_detail.groupby('user_id')['reordered'].sum() /
#         priors_orders_detail[priors_orders_detail['order_number'] > 1].groupby('user_id')['order_number'].count()).rename('_user_reorder_ratio')
# ], axis=1).reset_index()
# Runs about 12s on my machine instead of 780 using ka_add_groupby_features_1_vs_n

