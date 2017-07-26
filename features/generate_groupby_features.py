#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-2 下午8:55
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import cPickle
import pandas as pd
# remove warnings
import warnings
warnings.filterwarnings('ignore')

from utils import data_util
from conf.configure import Configure
from utils.time_util import tick_tock

# def main():
#     print 'load datas...'
#     train, test = data_util.load_dataset()

#     groupby_features_train = pd.DataFrame({'ID': train['ID']})
#     groupby_features_test = pd.DataFrame({'ID': test['ID']})

#     if (not os.path.exists(Configure.groupby_features_train_path)) or \
#             (not os.path.exists(Configure.groupby_features_test_path)):
#         groupby_features = []
#         for c in train.columns:
#             if ('label_encoder' in c) or ('tsne_cluster' in c):
#                 groupby_features.append(c)

#         for c in groupby_features:
#             print '>>>> perform groupby features, feature : {}...'.format(c)
#             groupby_df = train[[c, 'y']].groupby(c).aggregate('mean')['y'].reset_index()

#             def label_encoder_mean_map(data):
#                 values = groupby_df.loc[groupby_df[c] == data, 'y'].values
#                 if len(values) == 0:
#                     return sum(groupby_df.y) / groupby_df.shape[0]
#                 return values[0]

#             groupby_features_train[c + '_mean_y'] = train[c].map(label_encoder_mean_map)
#             groupby_features_test[c + '_mean_y'] = test[c].map(label_encoder_mean_map)

#             groupby_df = train[[c, 'y']].groupby(c).aggregate('median')['y'].reset_index()

#             def label_encoder_median_map(data):
#                 values = groupby_df.loc[groupby_df[c] == data, 'y'].values
#                 if len(values) == 0:
#                     return sum(groupby_df.y) / groupby_df.shape[0]
#                 return values[0]

#             groupby_features_train[c + '_median_y'] = train[c].map(label_encoder_median_map)
#             groupby_features_test[c + '_median_y'] = test[c].map(label_encoder_median_map)

#             groupby_df = train[[c, 'y']].groupby(c).aggregate('std')['y'].reset_index()
#             groupby_df.fillna(0, inplace=True)

#             def label_encoder_std_map(data):
#                 values = groupby_df.loc[groupby_df[c] == data, 'y'].values
#                 if len(values) == 0:
#                     return sum(groupby_df.y) / groupby_df.shape[0]
#                 return values[0]

#             groupby_features_train[c + '_std_y'] = train[c].map(label_encoder_std_map)
#             groupby_features_test[c + '_std_y'] = test[c].map(label_encoder_std_map)

#         with open(Configure.groupby_features_train_path, "wb") as f:
#             cPickle.dump(groupby_features_train, f, -1)
#         with open(Configure.groupby_features_test_path, "wb") as f:
#             cPickle.dump(groupby_features_test, f, -1)
#     else:
#         with open(Configure.groupby_features_train_path, "rb") as f:
#             groupby_features_train = cPickle.load(f)
#         with open(Configure.groupby_features_test_path, "rb") as f:
#             groupby_features_test = cPickle.load(f)

#     # merge
#     train = pd.merge(train, groupby_features_train, how='left', on='ID')
#     test = pd.merge(test, groupby_features_test, how='left', on='ID')

#     print 'train:', train.shape, ', test:', test.shape
#     print("Save data...")
#     data_util.save_dataset(train, test)


# if __name__ == '__main__':
#     print '========== perform groupby features =========='
#     main()
