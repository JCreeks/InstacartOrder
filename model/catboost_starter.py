
# coding: utf-8

# This script demonstrates basic use of catboost (CV 0.377750059602, LB 0.3802708)
# 
# I believe higher scores are possible - I got better early results without using aisle, so there may be an issue with how I set it up - maybe you can make it work better? ;)
# 
# The core of the script is based from "LB 0.3805009, Python Edition" by 🐳鲲(China), the catboost settings are taken from Fred Navruzov's posts, and the CV code is reused from my earlier notebook.
# 
# #### Original notes:
# 
# This script is translate from @Fabienvs's [R code](https://www.kaggle.com/fabienvs/instacart-xgboost-starter-lb-0-3791), I think it may help kagglers who do not use R.  
# I really appreciate @Fabienvs's great work, to be honest, I have no idea about how to handling this kind of problem(this is the first time I encounter recommendation problem- -)  
# here we go!! 
# below exist some very useful functions I write by my own, you can download from my [github repo](https://github.com/NickYi1990/Kaggle_Buddy.git)
# sorry for adding some Chinese in it, it would not affect the code
# ## The dataset is too big, you should run it on your desktop!

# In[1]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import gc
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:

import xgboost
import catboost


# In[3]:

def load_data(path_data):
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
    
    return priors, train, orders, products, aisles, departments, sample_submission

class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose
    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ......")
            self.begin_time = time.time()
    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + " end ......")
            print('time lapsing {0} s \n'.format(end_time - self.begin_time))
            
def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict, only_new_feature=True):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    '''
    with tick_tock("add stats features"):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(k + "should be a list")
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
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       ka_add_stats_features_n_vs_1(train, group_columns_list=['x0'], target_columns_list=['x10'])
    '''
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


# In[4]:

path_data = '../input/'
priors, train, orders, products, aisles, departments, sample_submission = load_data(path_data)


# ### Time-saving measure for local work - cache a local copy of the final dataframe 

# In[5]:

try:
    data = pd.read_pickle(path_data+'kernel38-data.pkl')
except:

    # Product part

    # Products information ----------------------------------------------------------------
    # add order information to priors set
    priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')

    # create new variables
    # _user_buy_product_times: 用户是第几次购买该商品
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

    prd.head()

    # User Part

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
    agg_dict_3 = {'reordered':
                  {'_user_reorder_ratio': 
                   lambda x: sum(priors_orders_detail.ix[x.index,'reordered']==1)/
                             sum(priors_orders_detail.ix[x.index,'order_number'] > 1)},
                  'product_id':{'_user_total_products':'count', 
                                '_user_distinct_products': lambda x: x.nunique()}}
    us = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['user_id'], agg_dict_3)
    users = users.merge(us, how='inner')

    # 平均每单的商品数
    # 每单中最多的商品数，最少的商品数++++++++++++++
    users['_user_average_basket'] = users._user_total_products / users._user_total_orders

    us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
    us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)

    users = users.merge(us, how='inner')

    users.head()

    # Database Part

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

    # release Memory
    # del train, prd, users
    # gc.collect()
    # release Memory
    #del priors_orders_detail, orders
    #gc.collect()

    #data.head()
    
    data.to_pickle(path_data+'kernel38-data.pkl')


# ## Validation code

# In[6]:

try:
    df_train_gt = pd.read_csv(path_data+'train.csv', index_col='order_id')
except:
    train_gtl = []

    op_train = pd.read_csv('../input/order_products__train.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})

    train_details = pd.merge(
                left=op_train,
                 right=orders, 
                 how='left', 
                 on='order_id'
                ).apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))
    for uid, subset in train_details.groupby('user_id'):
        subset1 = subset[subset.reordered == 1]
        oid = subset.order_id.values[0]

        if len(subset1) == 0:
            train_gtl.append((oid, 'None'))
            continue

        ostr = ' '.join([str(int(e)) for e in subset1.product_id.values])
        # .strip is needed because join can have a padding space at the end
        train_gtl.append((oid, ostr.strip()))

    df_train_gt = pd.DataFrame(train_gtl)

    df_train_gt.columns = ['order_id', 'products']
    df_train_gt.set_index('order_id', inplace=True)
    df_train_gt.sort_index(inplace=True)
    
    df_train_gt.to_csv(path_data+'train.csv')


# release Memory
del train, prd, users
gc.collect()
# release Memory
del priors_orders_detail, orders
gc.collect()

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


# # Create Train / Test

# ### Add aisle and dept id for catboost

# In[8]:

data = pd.merge(data, products.drop('product_name', axis=1), on='product_id')


# In[16]:

X_test = data.loc[data.eval_set == "test",:].copy()
X_test.drop(['eval_set'], axis=1, inplace=True)


# In[9]:

train = data.loc[data.eval_set == "train",:].copy()
train.drop(['eval_set'], axis=1, inplace=True)
train.loc[:, 'reordered'] = train.reordered.fillna(0)


# ### perform CV for catboost 

# In[10]:

def catboost_cv(X_train, y_train, X_val, y_val, features_to_use):
    cb = catboost.CatBoostClassifier(iterations=100, 
        thread_count=8, 
        verbose=True,
        random_seed=42,
        learning_rate=0.05,
        use_best_model=True,
        depth=8,
        fold_permutation_block_size=64,
        calc_feature_importance=True,
        leaf_estimation_method='Gradient')

    cb.fit(X=X_train[features_to_use].values, 
            y=y_train.values, 
            #cat_features=[c.get_loc('aisle_id'), c.get_loc('department_id')], 
            cat_features=[X_train[features_to_use].columns.get_loc('aisle_id')], 
            eval_set=[X_val[features_to_use].values, y_val.values],
            verbose=True,
            #plot=True
           )
    
    return cb


# In[ ]:

df_cvfolds = []
cb = []

for fold in range(4):
    train_subset = train[train.user_id % 4 != fold]
    valid_subset = train[train.user_id % 4 == fold]

    X_train = train_subset.drop('reordered', axis=1)
    y_train = train_subset.reordered

    X_val = valid_subset.drop('reordered', axis=1)
    y_val = valid_subset.reordered

    val_index = X_val[['user_id', 'product_id', 'order_id']]
    
    features_to_use = list(X_train.columns)
    features_to_use.remove('user_id')
    features_to_use.remove('product_id')
    features_to_use.remove('order_id')

    cb.append(catboost_cv(X_train, y_train, X_val, y_val, features_to_use))
    rawpreds = cb[-1].predict_proba(X_val[features_to_use].values)

    lim = .202
    val_out = val_index.copy()

    val_out.loc[:,'reordered'] = (rawpreds[:,1] > lim).astype(int)
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

    print(fold, compare_results(df_train_gt, fullfold))
    
    df_cvfolds.append(fullfold)


# In[15]:

df_cv = pd.concat(df_cvfolds)
print(compare_results(df_train_gt, df_cv))


# In[42]:

### now run model with 100% of train set for submission


# In[18]:

cb_full = catboost.CatBoostClassifier(iterations=100, 
    thread_count=8, 
    verbose=True,
    random_seed=42,
    learning_rate=0.05,
    depth=8,
    fold_permutation_block_size=64,
    calc_feature_importance=True,
    leaf_estimation_method='Gradient')

cb_full.fit(X=train[features_to_use].values, 
        y=train.reordered.values, 
        cat_features=[train[features_to_use].columns.get_loc('aisle_id')], 
        verbose=True,
        #plot=True
       )


# ### Prepare submission

# In[20]:

testpreds = X_test[['user_id', 'product_id', 'order_id']].copy()
testpreds['reordered'] = (cb_full.predict_proba(X_test[features_to_use].values)[:,1] > .202).astype(int)
testpreds.product_id = testpreds.product_id.astype(str)


# In[30]:

g = testpreds[testpreds.reordered == 1].groupby('order_id', sort=False)
df_testpreds = g[['product_id']].agg(lambda x: ' '.join(set(x)))

df_testpreds.head()


# In[34]:

# complete (but empty) test df
df_test = pd.DataFrame(index=X_test.order_id.unique())
df_test.index.name = 'order_id'
df_test['products'] = ['None'] * len(df_test)


# In[35]:

# yup, there are ~3345 order_id's with no orders
len(df_test), len(df_testpreds)


# In[37]:

# combine empty output df with predictions
df_test.loc[df_testpreds.index, 'products'] = df_testpreds.product_id
df_test.sort_index(inplace=True)
df_test.to_csv('../result/'+'preds_catboost.csv')

