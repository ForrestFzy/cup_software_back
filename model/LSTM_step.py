# -*- coding: utf-8 -*-
#@Time:2022/6/26 21:47
#@Author: FZY
#@File:LSTM_step.py
#@Software:Pycharm
#@Describe: 基于多变量LSTM的移动窗口回归预测电力负荷

'''
 单变量的预测和多变量的预测是不同的，
 在这里我写两个文件，分别提供单变量预测和多变量预测的方法
'''

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

### 将数据截取成3个一组的监督学习格式
from tensorflow.python.keras.models import load_model


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

#### 加载数据 顺便 缩放数据
def read_data(filepath, timeCol, usecols=None):
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

### 构建 LSTM 网络
def LSTMModel():
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == '__main__':
    ### TODO 这里控制所有的变量
    filePath = r"../file\用户负荷测试数据1\1000.csv"
    time_cols = 1
    x_cols = [2]
    look_back = 3
    ### 这里是测试方法
    df, scaler = read_data(filepath=filePath,
                   usecols=x_cols,
                   timeCol = time_cols)
    trainX, trainY, testX, testY = train_test_split(df, look_back=look_back)

    # 重构输入数据格式 [samples, time steps, features] = [93,1,3]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # model = LSTMModel()
    # 模型的恢复
    model = load_model(r'model_data/企业电力营销模型.h5')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
    # 模型的保存
    model.save(r'model_data/企业电力营销模型.mdl')
    # 对训练数据的Y进行预测
    trainPredict = model.predict(trainX)
    # 对测试数据的Y进行预测
    testPredict = model.predict(testX)
    # 对数据进行逆缩放
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # 计算RMSE误差
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
