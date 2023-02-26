# -*- coding: utf-8 -*-
#@Time:2022/6/25 15:19
#@Author: FZY
#@File:predata.py
#@Software:Pycharm
#@Describe: 数据集组织形式
'''
    1.任务一
        (1)单文件: 单用户，或者多用户
        (2)多文件: 单文件夹，每个文件属于一个用户
    2.任务二
        (1)单文件: 单用户、多用户
        (2)多文件: 单文件夹，每个文件属于一个用户
    3.任务三
        (1)单文件: 单用户、多用户
        (2)多文件: 单文件夹，每个文件属于一个用户
    4.任务四
        (1)单文件: 只支持单用户的预测
        (2)多文件: 单文件夹，每个文件属于一个用户
    5.任务五
        同任务四
'''
import pandas as pd



if __name__ == '__main__':
    filepath = r'F:\study\com\softwareCup\文档\数据集\CER_Electricity_Gas\CER Smart Metering Project\CER Electricity Revised March 2012\File1.txt'
    df = pd.read_csv(filepath, sep=' ')
    print(df.columns)
