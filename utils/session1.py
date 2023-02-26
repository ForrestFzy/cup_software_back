# -*- coding: utf-8 -*-
# @Time:2022/5/5 1:12
# @Author: FZY
# @File:session1.py
# @Software:Pycharm
# @Describe:
import datetime
import time

import pandas as pd
import numpy as np
import os


# 读取表格文件的列名和内容
def getColumns(filePath):
    if (str(filePath)[-5:] == '.xlsx' or str(filePath)[-4:] == '.xls'):
        df = pd.read_excel(str(filePath))
    elif str(filePath)[-4:] == '.csv':
        df = pd.read_csv(str(filePath))
    else:
        return False, False
    return list(df.columns), df


# 读取文件夹的内容
def getDirs(filePath):
    '''
    如果文件夹内的文件列名不一样，直接返回False
    否则： 计算并返回column, msg
    :param filePath: 文件夹路径
    :return:
     - column: 列名
            column: ['标题1', '标题2', ...]
     - msg: 文件夹内所有文件的内容
            msg: [ [[]], [[]], [[]], ... ]
    '''
    # 保存表格内容的变量
    table_content = []
    columns = []
    try:
        for i in os.listdir(filePath):
            subpath = os.path.join(filePath, i)
            tmp_columns, tmp_content = getColumns(subpath)
            flag = tmp_columns
            break
        for i in os.listdir(filePath):
            subpath = os.path.join(filePath, i)
            tmp_columns, tmp_content = getColumns(subpath)
            table_content.append(tmp_content)
            for j in tmp_columns:
                columns.append(j)
        columns = set(columns)
        if (list(columns) == list(flag)):
            return False, False
        else:
            return list(columns), np.array(table_content)
    except:
        return False, False


# 点击开始分析按钮调用的内容
def file_analysis(filepath, u_name, u_data, u_money):
    '''
    日期：按年来计算（需要判断年）
    分析的是文件的：
        个人缴费平均缴费次数   个人平均缴费金额
        全部平均缴费次数      全部平均缴费金额
    :param u_msg: 二维数组(整个表格的内容)
         - u_name: str : 用户名
         - u_data(时间戳delta): datatime :
         - u_money: str
    :return:
        - table_columns: ['标题1', '标题2', ...]
        - table_content: [[0001,5, 200],[0002,10, 300],[0003,20, 500], [人民,平均缴费次数,平均缴费金额],[]]
        - t_avg_c: 全部的平均缴费次数
    '''
    # 读取指定的文件
    if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
        df = pd.read_excel(filepath)
    elif str(filepath)[-4:] == '.csv':
        df = pd.read_csv(filepath)
    # 列名转变为列表，方便后续查询
    u_columns = list(df.columns)
    name_index = u_columns.index(u_name)  # 0
    data_index = u_columns.index(u_data)  # 1
    money_index = u_columns.index(u_money)  # 2
    # 获取所有的用户名
    name_list = list(set(list(df[u_name])))
    # 获取每位用户缴费的总金额
    money_list = list(df.groupby(u_name)[u_columns[money_index]].sum())
    # 获取每位用户缴费次数
    count_list = list(df.groupby(u_name)[u_columns[data_index]].count())
    # 获取每位用户缴费年限
    years = []  # 保存用户年限的

    df_group = list(df.groupby(u_name))  # 对所有用户按照用户名分组

    for i in range(len(df_group)):
        tmp = list(df_group[i][1][u_data])  # 依次提取出每个用户的缴费日期
        if(type(tmp[-1])==str):
            tmp[-1] = datetime.datetime.strptime(tmp[-1], "%Y-%m-%d")
            tmp[0] = datetime.datetime.strptime(tmp[0], "%Y-%m-%d")
        years.append(round((tmp[-1] - tmp[0]).days / 365, 2))  # 计算每一个用户最后一次缴费和第一次缴费的日期差，转换为年（保留2位小数）
    # 以上，每一位用户的总金额在money_list，总缴费次数在count_list, 缴费日期差在years,用户名在name_list
    # 转换为 numpy数组，方便计算
    count_list = np.array(count_list)
    money_list = np.array(money_list)
    years = np.array(years)
    result = pd.DataFrame({u_name: name_list,
                           '总缴费次数': count_list,
                           '总' + u_money: money_list,
                           '平均缴费次数(年)': count_list / years,
                           '平均' + u_money: money_list / years})

    return result


# 文件夹点击开始分析按钮调用的内容
def dir_analysis(dirpath, u_name, u_data, u_money):
    '''
    1 先把三维数组变成二维数组
    2 file_analysis(): 分组; 每年平均次数和金额;
    :param u_msg: 三维数组
    :return:
    '''
    result = []
    for i in os.listdir(dirpath):
        file_path = os.path.join(dirpath, i)
        tmp = file_analysis(file_path, u_name, u_data, u_money)
        result.append(tmp)
    return result


# 这里是画图的函数，横轴:时间; 纵轴:金额,次数
def paint_time_to_count(filepath, u_name, u_data, u_money):
    # try:
    # 读取指定的文件
    if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
        df = pd.read_excel(filepath)
    elif str(filepath)[-4:] == '.csv':
        df = pd.read_csv(filepath)
    money_time = list(df.groupby(u_name)[u_money].sum())
    count_time = list(df.groupby(u_name)[u_money].count())
    # 获取所有日期
    time_list = sorted(list(set(df[u_name])))
    result = pd.DataFrame({
        u_data: time_list,
        '总'+u_money: money_time,
        '计数': count_time
    })
    # 组织一下返回格式
    columns = result.columns
    content = result.values.tolist()
    # 组织一下列名的返回格式
    # 组织列名的格式
    tmp_columns = []
    for i in range(len(columns)):
        tmp = dict()
        tmp['prop'] = i + 1
        tmp['label'] = columns[i]
        tmp_columns.append(tmp)
    # 组织数据内容的格式
    tmp_content = []
    for i in range(len(content)):
        tmp = dict()
        for j in range(len(columns)):
            tmp[str(j + 1)] = content[i][j]
        tmp_content.append(tmp)
    return_dict = {
        'code': 200,
        'columns': tmp_columns,
        'data': tmp_content
    }
    ### 计算总平均缴费次数和缴费金额，直接返回
    avg_money = round(sum(money_time) / len(money_time), 2)
    avg_count = round(sum(count_time) / len(count_time), 2)
    # for i in range(len(time_list)):
    #     time_list[i] = datetime.datetime.strftime(time_list[i], "%Y-%m-%d")
    return_dict['datalist'] = time_list
    return_dict['moneylist'] = money_time
    return_dict['countlist'] = count_time
    return_dict['sum_money'] = sum(money_time)
    return_dict['sum_count'] = sum(count_time)
    # except:
    #     return_dict = {'code': 500}
    return return_dict


if __name__ == '__main__':
    basePath = r'../file/第一问.xlsx'
    # columns, content = getColumns(basePath)
    # content = content.values.tolist()
    # print(columns, content)
    # print(columns)
    r = file_analysis(basePath, '用户编号', '缴费日期', '缴费金额（元）')
    print(r)
    # result = file_analysis(basePath, '用户编号', '缴费日期', '缴费金额（元）')
    # print(result.values.tolist())
