# -*- coding: utf-8 -*-
#@Time:2022/6/27 15:05
#@Author: FZY
#@File:test.py
#@Software:Pycharm
#@Describe:

from model.lstm_pre import main
from model.DBSCAN import kmeans_data, paintOne, return_weight, allWeight

if __name__ == '__main__':
    ### TODO 这里控制所有的变量
    filePath = r"../jupy_test/test_datasets1/1000.csv"
    y_col = ['负荷']
    look_back = 3
    rawData, predictData, MSE, RMSE, MAE, MAPE, R2 = main(filePath, y_col, look_back)
    # print(predict_one_step([0.2564,0.2366,0.6585]))
    for i in range(len(rawData)):
        print(predictData[i] - rawData[i])
    # filepath = r"F:\study\com\softwareCup\文档\数据集\CER_Electricity_Gas\CER Smart Metering Project\CER Electricity Revised March 2012\CER_Electricity_Data\Survey data - CSV format\Smart meters SME post-trial survey data.xlsx"
    # del_percent = 0.5
    # print(allWeight(filepath, del_percent))