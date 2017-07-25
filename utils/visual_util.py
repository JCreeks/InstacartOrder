#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-7-3 上午10:29
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import pandas as pd


def boxplot_sorted(df, by, column, target='median', ascending=True, rot=0):
    """ 按照从小到大的顺序绘制 boxplot 盒型图
    :param df: 待统计的 DataFrame
    :param by: 所要 groupby 统计的特征
    :param column: boxplot 的目标值
    :param target: 所要绘制数据的属性, mean median
    :param ascending: boxplot 绘制的顺序
    :param rot: 图标显示的角度
    :return: axes
    
    plt.figure(figsize=(18,8))
    axes = boxplot_sorted(train, 'feature', 'target_y')
    axes.set_title("Boxplot of sepal width by iris species")
    axes.set_ylim([6,200])
    """
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col: vals[column] for col, vals in df.groupby(by)})
    sorded_df = None
    if target == 'median':
        # find and sort the median values in this new dataframe
        sorded_df = df2.median().sort_values(ascending=ascending)

    if target == 'mean':
        # find and sort the median values in this new dataframe
        sorded_df = df2.mean().sort_values(ascending=ascending)

    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    return df2[sorded_df.index].boxplot(rot=rot, return_type="axes")
