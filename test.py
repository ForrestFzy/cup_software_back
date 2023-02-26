# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import category_encoders as ce  # 编码分类变量
# from sklearn.metrics import accuracy_score
#
# filepath = r'./outputFiles/01_output2.csv'
# x_cols = ['平均缴费次数(年)', '平均缴费金额（元）', '总缴费次数', '总缴费金额（元）']
# y_col = '用户类型'
#
# if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
#     df = pd.read_excel(str(filepath))
# elif str(filepath)[-4:] == '.csv':
#     df = pd.read_csv(str(filepath))
# Y = df[y_col]
# X = df[x_cols]
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# encoder = ce.OrdinalEncoder()
# y_train = encoder.fit_transform(y_train)
# y_test = encoder.transform(y_test)
# feature_names = encoder.get_params()
#
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
#
# from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# import numpy as np
#
#
#
# varlist = np.linspace(1e-10, 1e-5, 20)
# dataList = []
# gnb_list = []
# gnb_isotonic_list = []
# gnb_sigmoid_list = []
# for i in varlist:
#     gnb = GaussianNB(var_smoothing=i)
#     gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method="isotonic")
#     gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method="sigmoid")
#     gnb.fit(X_train, y_train)
#     gnb_isotonic.fit(X_train, y_train)
#     gnb_sigmoid.fit(X_train, y_train)
#     y_prob1 = gnb.predict(X_test)
#     y_prob2 = gnb_isotonic.predict(X_test)
#     y_prob3 = gnb_sigmoid.predict(X_test)
#     dataList.append(i)
#     gnb_list.append(accuracy_score(y_prob1, y_test))
#     gnb_isotonic_list.append(accuracy_score(y_prob2, y_test))
#     gnb_sigmoid_list.append(accuracy_score(y_prob3, y_test))
# print(dataList)
# print(gnb_list)
# print(gnb_isotonic_list)
# print(gnb_sigmoid_list)

import os
savepath = os.path.join('./outputF', '居民客户的用电缴费习惯分析2.csv')
if(os.path.exists(savepath)):
    pass
else:
    os.makedirs(savepath)