# -*- coding: utf-8 -*-
#@Time:2022/6/28 19:26
#@Author: FZY
#@File:DBSCAN.py
#@Software:Pycharm
#@Describe: 先暂时弄一个简简单单的k-means聚类吧
import os
import ftplib

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import math
from numpy import array
from operator import itemgetter
import pickle
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

##### 计算df的熵权法的权重
def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 标准化
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))

    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)

    lnf = [[None] * cols for i in range(rows)]

    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf

    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
        # 计算各样本的综合得分,用最原始的数据

    w = pd.DataFrame(w)
    return w

##### 读取数据
def read_data(filepath):
    # 读取数据
    if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
        df = pd.read_excel(str(filepath))
    elif str(filepath)[-4:] == '.csv':
        df = pd.read_csv(str(filepath))
    elif str(filepath)[-4:] == '.txt':
        df = pd.read_csv(str(filepath), sep=' ')
    return df
##### 将DataFrame组织成Json表格格式的代码
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
##### 返回一定比例下将要删除的列
def null_processing(filepath, del_percent=0.5):
    df  =read_data(filepath)
    # 统计缺失值
    df_null = df.isnull().sum()
    # 存在来将要删除的列
    del_columns = []
    for i in df_null.axes[0]:
        if (df_null[i] / len(df) > del_percent):
            del_columns.append(i)
    return df, del_columns

##### 返回缺失值的列表(没什么用)
def null_list(df):
    df_null = df.isnull().sum()
    cols = []
    for i in df_null.axes[0]:
        if(df_null[i] > 0):
            cols.append(i)
    return cols

##### 返回聚类后的结果
# (名字是kmeans，但是使用的是DBSCAN和Brich，懒得改了)
def kmeans(filepath, del_percent, func):
    ### 首先删除缺失值超过比例的数据
    df, del_columns = null_processing(filepath, del_percent)
    df = df.drop(del_columns, axis=1)
    ### 再填充缺失值
    df = df.fillna(method="bfill")
    ### 基于密度的聚类方法
    if(func=="brich"):
        db = Birch(n_clusters=4).fit(df)
    else:
        db = DBSCAN(eps=150, min_samples=4).fit(df)
    # 保存模型
    with open('./model/model_data/电力用户集群分析模型.mdl', 'wb') as f:
        pickle.dump(db, f)
    ### 将聚类后的数据返回
    df.insert(1, 'labels', db.labels_)
    ### 将这个表格df返回
    return df

# 返回的是经过处理的前端表格返回结果
def kmeans_data(filepath, del_percent, func):
    df = kmeans(filepath, del_percent, func)
    return Dataframe2Json(df[df.columns[0:5]][0:50])

##### 画图返回每一组的结果(单个属性的结果)
def paintOne(filepath, del_percent, prop_col, func):
    df = kmeans(filepath, del_percent, func)
    prop_content = []
    label = ['label']
    flag = 0
    testName = ['高价值用户', '潜力型用户', '大众性用户', '低价值用户']
    for i in df.groupby('labels'):
        label.append(testName[flag])
        flag += 1
        tmp_content = dict()
        c = list(i[1][prop_col])
        for j in range(len(c)):
            if(c[j] not in tmp_content.keys()):
                tmp_content[c[j]] = 1
            else:
                tmp_content[c[j]] += 1
        prop_content.append(tmp_content)
    content = []
    for i in range(len(prop_content)):
        tmp_dict = dict()
        for j in prop_content[i].keys():
            if(j not in tmp_dict.keys()):
                tmp_dict[j] = prop_content[i][j]
            else:
                tmp_dict[j] += prop_content[i][j]
        content.append(tmp_dict)
    tmp_one = []
    for i in range(len(content)):
        keys = list(content[i].keys())
        for j in range(len(keys)):
            if(str(keys[j]) not in tmp_one):
                tmp_one.append(str(keys[j]))
    return_content = []
    return_content.append(label)
    for i in range(len(tmp_one)):
        tmp = []
        tmp.append(tmp_one[i])
        for j in range(len(content)):
            try:
                tmp.append(content[j][float(tmp_one[i])])
            except:
                tmp.append(0)
        return_content.append(tmp)
    return return_content

##### 这一个返回的是所有用户的权重占比(使用熵权法)
def return_weight(filepath, del_percent, func):
    df = kmeans(filepath, del_percent, func)
    df_columns = df.columns
    return_dict = {
        "code": 200,
        "data": []
    }
    for i in df.groupby("labels"):
        tmp = i[1]
        tmp.drop('labels', axis=1, inplace=True)
        tmp_weight = cal_weight(tmp).values.tolist()
        index_weight = sorted(enumerate(tmp_weight), key=itemgetter(1), reverse=True)
        tmp_content = []
        if(len(index_weight)<10):
            index_max = len(index_weight)
        else:
            index_max = 10
        for i in index_weight[0:index_max]:
            tmp_dict = {}
            tmp_dict["name"] = df_columns[i[0]]
            tmp_dict["value"] = round(i[1][0], 2)
            tmp_content.append(tmp_dict)
        return_dict['data'].append(tmp_content)
    return return_dict

##### 这一个也是返回的是所有用户所占有的权重占比(全部权重返回)
def allWeight(filepath, del_percent, func):
    df = kmeans(filepath, del_percent, func)
    df_columns = list(df.columns)
    del df_columns[1]
    return_dict = {
        "code": 200,
        "data": [],
        "label": df_columns,
    }
    for i in df.groupby("labels"):
        tmp = i[1]
        tmp.drop('labels', axis=1, inplace=True)
        tmp_weight = cal_weight(tmp).values.tolist()
        tmp_weight = [x[0] for x in tmp_weight]
        return_dict['data'].append(tmp_weight)
    return return_dict

##### 画聚类的3D立体图像
def paint3d(filepath, del_percent, func):
    df = kmeans(filepath, del_percent, func)
    n_comp = 3
    pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)
    X = df
    X_pca = pca.fit_transform(X)   # 返回一个二维数组
    # print('解释方差: %.4f' % pca.explained_variance_ratio_.sum())
    # print('个体差异贡献:')
    # for j in range(n_comp):
    #     print(pca.explained_variance_ratio_[j])
    ### 画图
    x = list(X_pca[:,0])
    y = list(X_pca[:,1])
    z = list(X_pca[:,2])
    c = []
    label = df['labels'].values.tolist()
    colors = ['red', 'blue', 'black', 'orange']
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(x)):
        c.append(colors[label[i]])
    ax.scatter3D(x, y, z, c=c, s=200)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    ax.tick_params(labelsize=11)
    savepath = os.path.join("./outputFiles", "cluster.svg")
    if (os.path.exists(savepath)):
        pass
    else:
        os.makedirs(savepath)
    return savepath








if __name__ == '__main__':
    filepath = r"../file/用户问卷信息1.xlsx"
    del_percent = 0.5
    return_dict = return_weight(filepath, del_percent)
    print(return_dict)
    data_x = []
    data_y = []
    for i in return_dict['data']:
        tmp_x = []
        tmp_y = []
        for j in i:
            tmp_x.append(j['name'])
            tmp_y.append(j['value'])
        data_x.append(tmp_x)
        data_y.append(tmp_y)

    pass


