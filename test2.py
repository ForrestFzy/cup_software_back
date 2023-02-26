from sklearn.model_selection import train_test_split
import pandas as pd
import category_encoders as ce  # 编码分类变量
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

filepath = r'./outputFiles/01_output2.csv'
x_cols = ['平均缴费次数(年)', '平均缴费金额（元）', '总缴费次数', '总缴费金额（元）']
y_col = '用户类型'

if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
    df = pd.read_excel(str(filepath))
elif str(filepath)[-4:] == '.csv':
    df = pd.read_csv(str(filepath))
Y = df[y_col]
X = df[x_cols]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
encoder = ce.OrdinalEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
feature_names = encoder.get_params()

data_list = []
train_list = []
test_list = []
for i in list(np.linspace(0, 1, 100)):
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=i)
    clf.fit(X_train, y_train)
    y_pred1 = clf.predict(X_train)
    y_pred2 = clf.predict(X_test)
    data_list.append(round(i, 3))
    train_list.append(round(accuracy_score(y_train, y_pred1),3))
    test_list.append(round(accuracy_score(y_test, y_pred2),3))
print(data_list, train_list, test_list)




