# -*- coding: utf-8 -*-
#@Time:2022/6/18 11:50
#@Author: FZY
#@File:session2.py
#@Software:Pycharm
#@Describe:
import datetime
import time

import pandas as pd
import numpy as np
import os

# 将DataFrame组织成Json表格格式的代码
def Dataframe2Json(df):
    columns = df.columns
    #### 组织列名的格式
    tmp_columns = []
    for i in range(len(columns)):
        tmp = dict()
        tmp['prop'] = i + 1
        tmp['label'] = columns[i]
        tmp_columns.append(tmp)
    content = df.values.tolist()
    # 组织数据内容的格式
    tmp_content = []
    for i in range(len(content)):
        tmp = dict()
        for j in range(len(columns)):
            tmp[str(j + 1)] = content[i][j]
        tmp_content.append(tmp)
    return {
        'columns': tmp_columns,
        'data': tmp_content
    }


# 先写一个计算那四类用户的函数(针对文件的)  前提：那个计算平均水平的表格已经存下来了(在session1中自动生成的)
def category(filepath, avg_money, avg_count, u_count, u_money):
    '''
    将文件中的所有用户分成了四类用户，并返回
    :param fileName:
    :param avg_money:
    :param avg_count:
    :return:
    {
        'code': 200,
        'columns': *,
        'data':*
    }
    '''
    ##### 读取文件
    if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
        df = pd.read_excel(str(filepath))
    elif str(filepath)[-4:] == '.csv':
        df = pd.read_csv(str(filepath))
    ##### 组织一下格式
    columns = list(df.columns)
    ##### 文件筛选成四类DataFrame
    user = []
    user_count = [0,0,0,0]
    count_list = list(df[u_count])
    money_list = list(df[u_money])
    for i in range(len(count_list)):
        if(count_list[i] >= avg_count and money_list[i] >= avg_money):
            user.append('高价值用户')
            user_count[0] += 1
        elif(count_list[i] >= avg_count and money_list[i] < avg_money):
            user.append('大众型用户')
            user_count[1] += 1
        elif (count_list[i] < avg_count and money_list[i] >= avg_money):
            user.append('潜力型用户')
            user_count[2] += 1
        elif (count_list[i] < avg_count and money_list[i] < avg_money):
            user.append('低价值用户')
            user_count[3] += 1
    df.insert(loc=len(df.columns), column='用户类型', value=user)
    # 组织保存路径
    savepath = os.path.join('./outputFiles', '居民客户的用电缴费习惯分析2.csv')
    if(os.path.exists(savepath)):
        pass
    else:
        os.makedirs(savepath)
    df.to_csv(savepath, index=False, encoding="utf_8_sig")
    return_dict = Dataframe2Json(df)
    return_dict['code'] = 200
    return_dict['user_count'] = user_count
    return_dict['filename'] = savepath.split('/')[-1].split('\\\\')[-1].split('\\')[-1]
    return_dict['savePath'] = os.path.abspath(savepath)
    return_dict['code'] = 200
    return return_dict
    


if __name__ == '__main__':
    basePath = r'..\\outputFiles\\01_output1.csv'
    category(basePath, 1000, 6)












