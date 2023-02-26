# -*- coding: utf-8 -*-
#@Time:2022/6/27 19:06
#@Author: FZY
#@File:app_test.py
#@Software:Pycharm
#@Describe:

from model.DBSCAN import paint3d
if __name__ == '__main__':
    filepath = r"./file/用户问卷信息1.xlsx"
    del_percent = 0.5
    func = "DBSCAN"
    paint3d(filepath, del_percent, func)



    # dirpath = r"./file/第四问"
    # load_col = '负荷'
    # date_col = "时间"
    # data, date, all_data, MSE, RMSE, MAE, MAPE, R2 = read_dir(dirpath, date_col, load_col)
    # path = "F:\\study\\com\\softwareCup\\website\\back\\jupy_test\\data_set\\air_pollution.csv"
    # print(path.replace("\\"+path.split('\\')[-1], ""))
    # all_data = np.array(all_data)
    # print(MSE, RMSE, MAE, MAPE, R2)

    pass
