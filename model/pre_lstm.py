# -*- coding: utf-8 -*-
#@Time:2022/7/5 16:41
#@Author: FZY
#@File:pre_lstm.py
#@Software:Pycharm
#@Describe:
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from sklearn.metrics import mean_squared_error   # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from sklearn.metrics import mean_absolute_percentage_error  # MAPE
from sklearn.metrics import r2_score             # R2
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import math

model = load_model(r'../model/model_data/lstm_multi.h5')

#转成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    #数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        #预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    #拼接 put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除值为NAN的行 drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prediction(filepath):
    dataset = read_csv(filepath, header=0, usecols=['rain', 'temp', 'wetb', 'dewpt', '负荷'])
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
    test_X = values[: ,:-1]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(test_X.shape)
    yhat = model.predict(test_X)
    real_values = values
    pred_values = concatenate((values[:, :-1], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(pred_values)
    real_y = real_values[: , -1]
    pred_y = inv_yhat[: , -1]
    MSE = mean_squared_error(real_y, pred_y)
    RMSE = math.sqrt(MSE)
    MAE = mean_absolute_error(real_y, pred_y)
    MAPE = mean_absolute_percentage_error(real_y, pred_y)
    return real_y, pred_y, MSE, RMSE, MAE, MAPE


if __name__ == '__main__':
    filepath = r'../file\用户负荷测试数据1\1000.csv'
    y1, y2, mse = prediction(filepath)
    x = []
    for i in range(len(y1)):
        x.append(i+1)
    plt.plot(x, y1, 'r')
    plt.plot(x, y2, 'b')
    plt.show()
    print(y1, y2, mse)