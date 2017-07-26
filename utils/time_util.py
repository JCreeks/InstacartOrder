#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-3 上午10:29
"""

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import pandas as pd

import time

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
