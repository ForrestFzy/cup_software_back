# -*- coding: utf-8 -*-
#@Time:2022/6/20 9:46
#@Author: FZY
#@File:NN.py
#@Software:Pycharm
#@Describe:

# -*- coding: utf-8 -*-
# @Time:2022/6/19 18:44
# @Author: FZY
# @File:GaussianNB.py
# @Software:Pycharm
# @Describe:
###### 导入所需要的第三方包
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # 拆分数据集
import category_encoders as ce  # 编码分类变量
from sklearn.tree import DecisionTreeClassifier


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

###### 读取数据
def load_data(filepath, x_cols, y_col):
    '''
    读取数据，返回自变量X和因变量Y
    :param filepath: 文件的路径，一般是任务二的输出文件
    :param x_cols: 自变量的列名
    :param y_col: 因变量的列名
    :return: 自变量 和 因变量 的数据
    '''
    ### 读取原始数据
    if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
        df = pd.read_excel(str(filepath))
    elif str(filepath)[-4:] == '.csv':
        df = pd.read_csv(str(filepath))
    Y = df[y_col]
    X = df[x_cols]
    return df, X, Y

###### 拆分数据集为训练集和测试集
def train_and_test(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    encoder = ce.OrdinalEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    feature_names = encoder.get_params()
    return X_train, X_test, y_train, y_test, encoder, feature_names

###### 构建决策树
def DT(X, x_train, y_train, x_test, y_test, criterion, splitter, max_depth):
    '''
    构建决策树对训练集和测试集进行预测，并对整个数据集进行预测
    :param X: 整个数据集的自变量
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    '''
    max_depth = int(max_depth)
    clf = DecisionTreeClassifier(criterion=criterion,
                                 splitter=splitter,
                                 max_depth=max_depth)
    clf = clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    y_pred = clf.predict(X)
    y_pred_prob = clf.predict_proba(X)
    return y_train, y_test, y_test_pred, y_train_pred, y_pred, y_pred_prob

###### 计算衡量指标
def calculate(y_train, y_test, y_test_pred, y_train_pred):
    test_score = accuracy_score(y_test, y_test_pred)  # 计算测试集上的得分，即准确率
    train_score = accuracy_score(y_train, y_train_pred)  # 计算训练集上的得分，即准确率
    return train_score, test_score

###### 只需要调用这一个函数即可（主函数）
def DTModel(filepath, x_cols, y_col, criterion, splitter, max_depth):
    if(type(x_cols)==str):
        x_cols = x_cols.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(',')
    # 获取自变量和因变量
    df, X, Y = load_data(filepath, x_cols, y_col)
    # 拆分数据集
    X_train, X_test, y_train, y_test, encoder, feature_names = train_and_test(X, Y)
    # 训练模型
    y_train, y_test, y_test_pred, y_train_pred, y_pred, y_pred_prob = DT(X, X_train, y_train, X_test, y_test, criterion, splitter, max_depth)
    # 预测的数字逆转码
    y_pred = pd.DataFrame({y_col: y_pred})
    y_pred = encoder.inverse_transform(y_pred)
    # 重新组织表格展示的部分(并将表格存储在outputFiles3中)
    df['预测' + y_col] = y_pred
    df['高价值用户概率'] = y_pred_prob[: ,1]
    savepath = os.path.join('./outputFiles', '居民客户的用电缴费习惯分析3.csv')
    if (os.path.exists(savepath)):
        pass
    else:
        os.makedirs(savepath)
    df.to_csv(savepath, index=False, encoding='utf_8_sig')
    # 得到衡量指标
    train_score, test_score = calculate(y_train, y_test, y_test_pred, y_train_pred)
    ###### 返回两个指标 和 表格的内容 和 下载的位置
    return train_score, test_score, Dataframe2Json(df), savepath

###### 基于决策树的画图的函数
def paintDT(filepath, x_cols, y_col):
    criterions = ['gini', 'entropy']
    max_depth = list(range(1, 11))
    dataList = max_depth
    gini_train = []
    gini_test = []
    entropy_train = []
    entropy_test = []
    log_train = []
    log_test = []
    for i in criterions:
        for j in max_depth:
            train_score, test_score, df, filename = DTModel(filepath, x_cols, y_col, criterion=i, max_depth=j, splitter='best')
            if(i=='gini'):
                gini_train.append(train_score)
                gini_test.append(test_score)
            elif(i=='entropy'):
                entropy_train.append(train_score)
                entropy_test.append(test_score)
            else:
                log_train.append(train_score)
                log_test.append(test_score)
    return dataList, gini_train, gini_test, entropy_train, entropy_test, log_train, log_test


###### 决策树的测试部分，仅供测试使用，对外提供接口，由flask调用
if __name__ == '__main__':
    filepath = r'../outputFiles/01_output2.csv'
    x_cols = "['平均缴费次数(年)', '平均缴费金额（元）', '总缴费次数', '总缴费金额（元）']"
    y_col = '用户类型'
    # train_score, test_score, df, savepath = DTModel(filepath, x_cols, y_col)
    # print(train_score, test_score, df, savepath)
    dataList, gini_train, gini_test, entropy_train, entropy_test, log_train, log_test = paintDT(filepath, x_cols, y_col)
    print(dataList, gini_train, gini_test, entropy_train, entropy_test, log_train, log_test )




