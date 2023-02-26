# -*- coding: utf-8 -*-
#@Time:2022/6/27 9:45
#@Author: FZY
#@File:lstm_pre.py
import math

import pandas as pd
import numpy
from tensorflow.python.keras.models import load_model
from sklearn.metrics import mean_squared_error   # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from sklearn.metrics import mean_absolute_percentage_error  # MAPE
from sklearn.metrics import r2_score             # R2
from sklearn.preprocessing import MinMaxScaler
import datetime
import os

##### 只返回日期的那一列数据，以列表的形式返回
def read_date(filepath, date_col):
    # 读取数据
    if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
        df = pd.read_excel(str(filepath))
    elif str(filepath)[-4:] == '.csv':
        df = pd.read_csv(str(filepath))
    elif str(filepath)[-4:] == '.txt':
        df = pd.read_csv(str(filepath), sep=' ')
    return list(df[date_col])

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

#### 加载数据 顺便 缩放数据
def read_data(filepath, usecols=None):
    # 读取数据
    if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
        df = pd.read_excel(str(filepath), usecols=usecols)
    elif str(filepath)[-4:] == '.csv':
        df = pd.read_csv(str(filepath), usecols=usecols)
    elif str(filepath)[-4:] == '.txt':
        df = pd.read_csv(str(filepath), sep=' ', usecols=usecols)
    # df = df.drop(df.columns[timeCol], axis=1)
    # 转换数据为float型，只有float型数据才能进行LSTM
    df = df.values
    df = df.astype('float32')
    # 缩放数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df)
    return df, scaler

### 数据集拆分成训练集和测试集，按照3:7的比例进行拆分
def train_test_split(dataset, look_back=3):
    # 分割2/3数据作为测试
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # 预测数据步长为3,三个预测一个，3->1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    return trainX, trainY, testX, testY

def main(filePath, y_col, look_back):
    df, scaler = read_data(filepath=filePath,
                           usecols=y_col)
    trainX, trainY, testX, testY = train_test_split(df, look_back=look_back)
    # 重构输入数据格式 [samples, time steps, features] = [93,1,3]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # 对训练数据的Y进行预测
    trainPredict = model.predict(trainX)
    # 对测试数据的Y进行预测
    testPredict = model.predict(testX)
    # 对数据进行逆缩放
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    rawData = list(trainY[0])+ list(testY[0])
    predictData = list(map(float, list(trainPredict) + list(testPredict)))
    # 计算RMSE误差 和 MSE误差
    trainScore = mean_squared_error(trainY[0], trainPredict[:, 0])
    testScore = mean_squared_error(testY[0], testPredict[:, 0])
    MSE = (trainScore + testScore) / 2
    RMSE = math.sqrt(MSE)
    # 计算 MAE
    trainScore = mean_absolute_error(trainY[0], trainPredict[:, 0])
    testScore = mean_absolute_error(testY[0], testPredict[:, 0])
    MAE = (trainScore + testScore) / 2
    # 计算 MAPE
    trainScore = mean_absolute_percentage_error(trainY[0], trainPredict[:, 0])
    testScore = mean_absolute_percentage_error(testY[0], testPredict[:, 0])
    MAPE = (trainScore + testScore) / 2
    # 计算R2
    trainScore = r2_score(trainY[0], trainPredict[:, 0])
    testScore = r2_score(testY[0], testPredict[:, 0])
    R2 = (trainScore + testScore) / 2
    return rawData, predictData, MSE, RMSE, MAE, MAPE, R2

model = load_model(r'./model/model_data/企业电力营销模型.h5')
# 一次只预测一步
def predict_one_step(date, data):
    data = numpy.array([data])
    data = numpy.reshape(data, (data.shape[0], 1, data.shape[1]))
    Predict = model.predict(data)
    date = datetime.datetime.strptime(str(date), "%Y-%m-%d %H:%M")
    tmp_day = (date + datetime.timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")
    tmp_day = str(tmp_day)
    return float(Predict) + sum(data)/len(data), tmp_day

# **************** 对文件夹内容进行预测 **********************
def read_dir(dirpath, date_col, load_col):
    data = []
    date = []
    subFile = []
    for i in os.listdir(dirpath):
        subFile.append(i)
        filepath = os.path.join(dirpath, i)
        # 读取数据
        if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
            df = pd.read_excel(str(filepath))
        elif str(filepath)[-4:] == '.csv':
            df = pd.read_csv(str(filepath))
        elif str(filepath)[-4:] == '.txt':
            df = pd.read_csv(str(filepath), sep=' ')
        df = df.sort_values(date_col)
        load_data = df.groupby(date_col)[load_col].sum()
        date_data = df.drop_duplicates(subset=date_col)
        data.append(load_data.values.tolist()[-2000:-1])
        date = date_data['时间'].values.tolist()[-2000:-1]
    data1 = numpy.array(data)
    all_data = []
    for i in range(len(data1[0])):
        all_data.append(float(sum(data1[:,i])))
    all_data = list(all_data)
    all_date = date_data['时间'].values.tolist()
    rawData, predictData, MSE, RMSE, MAE, MAPE, R2 = main(filepath, [load_col], 3)
    return data, date, all_data, all_date, MSE, RMSE, MAE, MAPE, R2, subFile

##### 这里返回的是任务四的报告的 三个单位 的预测
def pre_three(dirpath, date_col, load_col):
    data, date, all_data, MSE, RMSE, MAE, MAPE, R2, subFile = read_dir(dirpath, date_col, load_col)
    for k in range(3):
        for i in range(len(data)):
            tmp_data = data[i][-3:]
            tmp_date = date[-1]
            tmp1, tmp2 = predict_one_step(tmp_date, tmp_data)
            data[i].append(tmp1)
            date.append(tmp2)
    return_dict = {
        "code": 200,
        "columns": [],
        "data": [],
    }
    return_dict['columns'].append({"prop": '1', "label": "用户"})
    for i in range(3):
        # 首先组织以下列的内容
        tmp = {}
        tmp['prop'] = str(i + 2)
        tmp['label'] = date[i]
        return_dict['columns'].append(tmp)
    for i in range(len(data)):
        # 然后组织内容
        tmp = {}
        tmp[str(1)] = os.listdir(dirpath)[i].replace(".csv", "")
        tmp[str(2)] = data[i][-3]
        tmp[str(3)] = data[i][-2]
        tmp[str(4)] = data[i][-1]
        return_dict['data'].append(tmp)
    return return_dict









# *********************************************************


if __name__ == '__main__':
    ### TODO 这里控制所有的变量
    # filePath = r"../jupy_test/test_datasets1/1000.csv"
    # y_col = ['负荷']
    # look_back = 3
    # rawData, predictData, MSE, RMSE, MAE, MAPE, R2 = main(filePath, y_col, look_back)
    print(predict_one_step([0.2564,0.2366,0.6585], "2001-8-9 14:00"))
    # dirpath = r"../file\用户负荷测试数据1"
    # print(pre_three(dirpath, date_col="时间", load_col="负荷"))
    # print(rawData)