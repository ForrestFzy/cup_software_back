# -*- coding: utf-8 -*-
# @Time:2022/7/5 15:36
# @Author: FZY
# @File:LSTM.py
# @Software:Pycharm
# @Describe: 这里提供的是多变量多步预测的，另外堆叠网络以加以改进(注意过程需要防止过拟合)
import random
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import numpy as np


# 转成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # 预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # 拼接 put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除值为NAN的行 drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 返回数据
def return_map_data(name):
    nameMap = {
        'Carlow': 61931 * 17,
        'Cavan': 76176 * 17,
        'Clare': 118817 * 17,
        'Cork': 127419 * 17,
        'Donegal': 159192 * 17,
        'Dublin': 1345402 * 17,
        'Galway': 66000 * 17,
        'Kerry': 147707 * 17,
        'Kildare': 24473 * 17,
        'Kilkenny': 99232 * 17,
        'Laois': 84697 * 17,
        'Leitrim': 32044 * 17,
        'Limerick': 183863 * 17,
        'Longford': 40873 * 17,
        'Louth': 128884 * 17,
        'Mayo': 130507 * 17,
        'Meath': 195044 * 17,
        'Monaghan': 61386 * 17,
        'Offaly': 77961 * 17,
        'Roscommon': 64544 * 17,
        'Sligo': 65535 * 17,
        'Tipperary': 159553 * 17,
        'Waterford': 116176 * 17,
        'Westmeath': 88770 * 17,
        'Wexford': 149722 * 17,
        'Wicklow': 142425 * 17
    }
    min_num = int(nameMap[name] * 0.7)
    max_num = int(nameMap[name] * 1.3)
    return random.randint(min_num, max_num)
##### 返回电力调度的差
def return_mapChange_data(name):
    nameMap = {
        'Carlow': 61931 * 17,
        'Cavan': 76176 * 17,
        'Clare': 118817 * 17,
        'Cork': 127419 * 17,
        'Donegal': 159192 * 17,
        'Dublin': 1345402 * 17,
        'Galway': 66000 * 17,
        'Kerry': 147707 * 17,
        'Kildare': 24473 * 17,
        'Kilkenny': 99232 * 17,
        'Laois': 84697 * 17,
        'Leitrim': 32044 * 17,
        'Limerick': 183863 * 17,
        'Longford': 40873 * 17,
        'Louth': 128884 * 17,
        'Mayo': 130507 * 17,
        'Meath': 195044 * 17,
        'Monaghan': 61386 * 17,
        'Offaly': 77961 * 17,
        'Roscommon': 64544 * 17,
        'Sligo': 65535 * 17,
        'Tipperary': 159553 * 17,
        'Waterford': 116176 * 17,
        'Westmeath': 88770 * 17,
        'Wexford': 149722 * 17,
        'Wicklow': 142425 * 17
    }
    min_num = int(nameMap[name] * 0.7)
    max_num = int(nameMap[name] * 1.3)
    now = random.randint(min_num, max_num)
    ## 组织文字，自动化调度
    if(now - nameMap[name] > 0):
        text = str(name) + "城市请求电力：" + str(now-nameMap[name]) + 'kw/h'
    else:
        text = str(name) + "城市调离电力：" + str(nameMap[name] - now) + 'kw/h'
    return abs(now - nameMap[name]), text




# 构建模型
def build_model(trainX, trainY, look_back):
    '''
    构建网络
    :param trainX: 自变量，格式为 n*1*3
    :param trainY: 因变量，格式为 n*1
    :param look_back: 预测的步长， 与自变量中的 3 一模一样
    :return:
    '''
    # 构建 LSTM 网络
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    return model


if __name__ == '__main__':
    # 这里是 测试的部分
    ##数据预处理 load dataset
    dataset = read_csv(r'../file/用户负荷测试数据1/1000.csv', header=0, usecols=['rain', 'temp', 'wetb', 'dewpt', '负荷'])
    dataset[['dewpt', '负荷']] = dataset[['负荷', 'dewpt']]
    values = dataset.values
    # 归一化 normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # 转成有监督数据 frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # 删除不预测的列 drop columns we don't want to predict
    reframed.drop(reframed.columns[[5, 6, 7, 8, 9]], axis=1, inplace=True)
    # 数据准备
    # 把数据分为训练数据和测试数据 split into train and test sets
    values = reframed.values
    # 拿一年的时间长度训练
    n_train_hours = 365 * 24
    # 划分训练数据和测试数据
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # 拆分输入输出 split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    ##模型定义 design network
    model = Sequential()
    model.add(LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # 模型训练 fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    model.save(r'model_data/lstm_multi.h5')

    print(history.history['loss'])
    print(history.history['val_loss'])

    # 输出 plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # 进行预测 make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # 预测数据逆缩放 invert scaling for forecast
    inv_yhat = concatenate((test_X, yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 4]
    inv_yhat = np.array(inv_yhat)
    # 真实数据逆缩放 invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_X, test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 4]

    # 画出真实数据和预测数据
    pyplot.plot(inv_yhat, label='prediction')
    pyplot.plot(inv_y, label='true')
    pyplot.legend()
    pyplot.show()

    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    pass
