#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-29 下午2:21
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

cmd = 'rm ../input/*_dataset.pkl'
os.system(cmd)

cmd = 'python train_test_preprocess.py'
os.system(cmd)

cmd = 'python perform_category_features.py'
os.system(cmd)

# cmd = 'python impute_missing_data.py'
# os.system(cmd)

cmd = 'python generate_decomposition_features.py'
os.system(cmd)

cmd = 'python perform_feature_discretize.py'
os.system(cmd)

cmd = 'python generate_feature_distance.py'
os.system(cmd)

# cmd = 'python generate_tsne_features.py'
# os.system(cmd)

cmd = 'python generate_groupby_features.py'
os.system(cmd)

cmd = 'python perform_other_features.py'
os.system(cmd)
