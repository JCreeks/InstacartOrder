#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-3 上午10:29
"""

import os
import sys
from sklearn.preprocessing import LabelEncoder

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import pandas as pd

def transform_categorical_data(data, categorical_list):                   
    for f in categorical_list:
        encoder = LabelEncoder()
        encoder.fit(list(data[f])) 
        data[f] = encoder.transform(data[f].ravel())