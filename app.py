import os

from flask import Flask, request, make_response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import json
import pandas as pd
import time
from model.lstm_pre import *
import numpy as np

app = Flask(__name__)
# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://team:123456@120.25.148.251:3306/software_cup"
# 设置这一项是每次请求结束后都会自动提交数据库中的变动
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# 实例化
db = SQLAlchemy(app)

##### 插入留言
@app.route('/insert_msg', methods=['GET', 'POST'])
def insert_msg():
    title = request.args.get('title')
    content = request.args.get('content')
    if (title == None):
        title = request.json['title']
        content = request.json['content']
    try:
        sql = "insert into leaveword(title, content) values (:title, :content);"
        db.session.execute(sql, {'title': title, 'content': content})
        return {'code': 200}
    except:
        return {'code': 500}

##### 查询留言
@app.route('/select_msg', methods=['GET', 'POST'])
def select_msg():
    return_dict = {
        'code': 404,
        'data': []
    }
    sql = "SELECT title, content, replied from leaveword;"
    results = db.session.execute(sql)
    data = []
    for i in results:
        tmp = {}
        tmp['title'] = i[0]
        tmp['content'] = i[1]
        if(i[2]==int(0)):
            tmp['replied'] = False
        else:
            tmp['replied'] = True
        data.append(tmp)
    return_dict['data'] = data
    return_dict['code'] = 200
    return return_dict

##### 根据路径下载文件
@app.route('/download1', methods=['GET', 'POST'])
def download1():
    filepath = request.args.get('filepath')
    if (filepath == None):
        filepath = request.json['filepath']
    name = filepath.split('\\')[-1]  # 切割出文件名称
    filePath = filepath.replace(name, '')
    response = send_from_directory(filePath, name, as_attachment=True)
    return response

##### 检查是否有更新
@app.route('/update', methods=['GET', 'POST'])
def update():
    '''
    是否更新软件的函数
    :return: 
    '''
    sql = "SELECT version, title, content, link from isupdate ORDER BY id DESC limit 1;"
    results = db.session.execute(sql)
    data = []
    for i in results:
        tmp = {}
        tmp['version'] = str(i[0])
        tmp['title'] = str(i[1])
        tmp['content'] = i[2]
        tmp['link'] = i[3]
        data.append(tmp)
    return_dict = {
        'code': 404,
        'data': []
    }
    return_dict['data'] = data
    return_dict['code'] = 200

    return return_dict

##### 获取最近访问
@app.route('/getlastpath', methods=['GET', 'POST'])
def getlastpath():
    '''
    获取最近文件路径的方法，只获取最后5条
    :return:
    '''
    sql = 'SELECT filepath from lastpath ORDER BY id DESC limit 5;'
    results = db.session.execute(sql)
    data = []
    for i in results:
        data.append(i[0])
    return {'code': 200, 'data': data}

## %% 这里是任务一的调用函数的接口
@app.route('/analyzefile', methods=['GET', 'POST'])
def analyzefile():
    '''
    文件分析
    :return:
    '''
    return_dict = {'code': 500}
    from utils.session1 import file_analysis
    filepath = request.args.get('filepath')
    u_name = request.args.get('u_name')
    u_date = request.args.get('u_date')
    u_money = request.args.get('u_money')
    if (filepath == None):
        filepath = request.json['filepath']
        u_name = request.json['u_name']
        u_date = request.json['u_date']
        u_money = request.json['u_money']
    result = file_analysis(filepath, u_name=u_name, u_data=u_date, u_money=u_money)
    return result


## %% 这里是任务一的函数调用接口
# 获取文件的列名和内容(已经调整为根据本地路径获取文件的列名和内容)
@app.route('/getFileContent', methods=['GET', 'POST'])
def getFileContent():
    '''
    获取文件的列名和内容，对应界面的《数据预览》
    :return:
    '''
    return_dict = {
        'code': 500,
        'columns': [],
        'data': []
    }
    from utils.session1 import getColumns
    filepath = request.args.get('filename')
    if (filepath == None):
        filepath = request.json['filename']
    columns, content = getColumns(filepath)
    # 组织列名格式
    tmp_columns = []
    for i in range(len(columns)):
        tmp = dict()
        tmp['prop'] = i + 1
        tmp['label'] = columns[i]
        tmp_columns.append(tmp)
    content = content.values.tolist()
    tmp_content = []
    for i in range(len(content)):
        tmp = dict()
        for j in range(len(columns)):
            tmp[str(j + 1)] = content[i][j]
        tmp_content.append(tmp)
    return_dict['code'] = 200
    return_dict['columns'] = tmp_columns
    return_dict['data'] = tmp_content
    del tmp_columns, tmp_content
    return return_dict

# # 获取文件夹的列名和内容(已经调整为根据本地路径获取文件的列名和内容)
@app.route('/getDirContent', methods=['GET', 'POST'])
def getDirContent():
    return_dict = {
        'code': 500,
        'columns': [],
        'data': [],
        "subfile": [],
    }
    from utils.session1 import getColumns
    dirpath = request.args.get('dirpath')
    if (dirpath == None):
        dirpath = request.json['dirpath']
    dirpath = dirpath.replace( dirpath.split("\\")[-1] , "")
    for i in os.listdir(dirpath):
        filepath = os.path.join(dirpath, i)
        columns, content = getColumns(filepath)
        content = content.values.tolist()
        # 组织列名格式
        tmp_columns = []
        for i in range(len(columns)):
            tmp = dict()
            tmp['prop'] = i + 1
            tmp['label'] = columns[i]
            tmp_columns.append(tmp)
        # 组织内容格式
        tmp_content = []
        for i in range(len(content)):
            tmp = dict()
            for j in range(len(columns)):
                tmp[str(j + 1)] = content[i][j]
            tmp_content.append(tmp)
        return_dict['code'] = 200
        return_dict['columns'] = tmp_columns
        return_dict['data'].append(tmp_content)
        return_dict["subfile"].append(filepath)
    return return_dict

# 这是任务一中分析文件内容的代码，返回平均值
@app.route('/fileAnalyze', methods=['GET', 'POST'])
def fileAnalyze():
    '''
    根据文件名，返回平均值
    方便作图，还需要返回:
     - 平均缴费次数, 平均缴费金额 <----- session1.file_analysis
     - 横轴:时间; 纵轴:金额,次数 <----- session1.paint_time_to_count
     - 每个人的平均缴费金额和平均缴费次数
    :return:
    '''
    # try:
    from utils.session1 import file_analysis, paint_time_to_count
    from utils.session1_credit import cal_credit
    filepath = request.args.get('filename')
    u_name = request.args.get('u_name')
    u_date = request.args.get('u_date')
    u_money = request.args.get('u_money')
    if (filepath == None):
        filepath = request.json['filename']
        u_name = request.json['u_name']
        u_date = request.json['u_date']
        u_money = request.json['u_money']
    #### 第一问的信用分
    creditScore, user, score = cal_credit(filepath)
    ####
    basePath = filepath
    result = file_analysis(basePath, u_name, u_date, u_money)
    result2 = paint_time_to_count(basePath, u_name, u_date, u_money)
    #### 将每个人的平均缴费次数和平均缴费金额保存在服务器中
    # 组织保存路径
    savepath = os.path.join('./outputFiles', '居民客户的用电缴费习惯分析1.csv')
    if (os.path.exists(savepath)):
        pass
    else:
        os.makedirs(savepath)
    result.to_csv(savepath, index=False, encoding='utf_8_sig')
    columns = list(result.columns)
    content = result.values.tolist()
    #### 组织列名的格式
    tmp_columns = []
    for i in range(len(columns)):
        tmp = dict()
        tmp['prop'] = i + 1
        tmp['label'] = columns[i]
        tmp_columns.append(tmp)
    # 组织数据内容的格式
    tmp_content = []
    for i in range(len(content)):
        tmp = dict()
        for j in range(len(columns)):
            tmp[str(j + 1)] = content[i][j]
        tmp_content.append(tmp)
    return_dict = {
        'code': 200,
        'columns': tmp_columns,
        'data': tmp_content,
        'userid': user,
        'score': score,
    }
    return_dict['session1Name'] = filepath.split('.')[-2] + '_output1.csv'
    # 将绘图的数据一起组织
    if (result2['code'] == 200):
        return_dict['paint1'] = result2
    ## 计算整体的平均缴费金额和缴费次数
    return_dict['avg_money'] = round(result[result.columns[4]].sum() / len(result), 2)
    return_dict['avg_count'] = round(result[result.columns[3]].sum() / len(result), 2)
    return_dict['savePath'] = os.path.abspath(os.path.join('./outputFiles', '居民客户的用电缴费习惯分析1.csv'))
    return return_dict
    # except:
    #     return {'code':500}

### 这里时分析文件夹的内容的代码，先合并，然后利用文件分析的方法
@app.route('/dirAnalyze', methods=['GET', 'POST'])
def dirAnalyze():
    from utils.session1 import file_analysis, paint_time_to_count
    dirPath = request.args.get('dirPath')
    u_name = request.args.get('u_name')
    u_date = request.args.get('u_date')
    u_money = request.args.get('u_money')
    if (dirPath == None):
        dirPath = request.json['dirPath']
        u_name = request.json['u_name']
        u_date = request.json['u_date']
        u_money = request.json['u_money']
    dirPath = dirPath.replace( dirPath.split('\\')[-1], "" )
    df = pd.DataFrame()
    for i in os.listdir(dirPath):
        filepath = os.path.join(dirPath, i)
        df = df.append(pd.read_excel(filepath))
    ## 合并后的文件为 df
    savepath = os.path.join("./outputFiles", "居民客户的用电缴费习惯分析1.csv")
    if (os.path.exists(savepath)):
        pass
    else:
        os.makedirs(savepath)
    df.to_csv(savepath, encoding='utf_8_sig')
    result = file_analysis(savepath, u_name, u_date, u_money)
    result2 = paint_time_to_count(savepath, u_name, u_date, u_money)
    result.to_csv(savepath, index=False, encoding='utf_8_sig')
    columns = list(result.columns)
    content = result.values.tolist()
    #### 组织列名的格式
    tmp_columns = []
    for i in range(len(columns)):
        tmp = dict()
        tmp['prop'] = i + 1
        tmp['label'] = columns[i]
        tmp_columns.append(tmp)
    # 组织数据内容的格式
    tmp_content = []
    for i in range(len(content)):
        tmp = dict()
        for j in range(len(columns)):
            tmp[str(j + 1)] = content[i][j]
        tmp_content.append(tmp)
    return_dict = {
        'code': 200,
        'columns': tmp_columns,
        'data': tmp_content
    }
    return_dict['session1Name'] = filepath.split('.')[-2] + '_output1.csv'
    # 将绘图的数据一起组织
    if (result2['code'] == 200):
        return_dict['paint1'] = result2
    ## 计算整体的平均缴费金额和缴费次数
    return_dict['avg_money'] = round(result[result.columns[4]].sum() / len(result), 2)
    return_dict['avg_count'] = round(result[result.columns[3]].sum() / len(result), 2)
    return_dict['savePath'] = os.path.abspath(os.path.join('./outputFiles', '居民客户的用电缴费习惯分析1.csv'))
    return return_dict



## %% 这里是任务二的函数调用接口
# 这是任务二主要的调用函数
@app.route('/fileCategory', methods=['GET', 'POST'])
def fileCategory():
    from utils.session2 import category
    filepath = request.args.get('filename')
    avg_count = request.args.get('avg_count')
    avg_money = request.args.get('avg_money')
    u_count = request.args.get('u_count')
    u_money = request.args.get('u_money')
    if (filepath == None):
        filepath = request.json['filename']
        avg_count = request.json['avg_count']
        avg_money = request.json['avg_money']
        u_count = request.json['u_count']
        u_money = request.json['u_money']
    avg_count = float(avg_count)
    avg_money = float(avg_money)
    result = category(filepath, avg_count=avg_count, avg_money=avg_money, u_count=u_count, u_money=u_money)
    return result


# 下载任务二的结果文件的表格
@app.route('/downloadSession2', methods=['GET', 'POST'])
def downloadSession2():
    filepath = request.args.get('filename')
    if (filepath == None):
        filepath = request.json['filename']
    response = send_from_directory('./outputFiles', filepath, as_attachment=True)
    return response


@app.route('/downloadSession3', methods=['GET', 'POST'])
def downloadSession3():
    filepath = request.args.get('filename')
    if (filepath == None):
        filepath = request.json['filename']
    response = send_from_directory('./outputFiles3', filepath, as_attachment=True)
    return response


@app.route('/downloadData', methods=['GET', 'POST'])
def downloadData():
    filepath = request.args.get('filename')
    if (filepath == None):
        filepath = request.json['filename']
    response = send_from_directory('./uploadFiles', filepath, as_attachment=True)
    return response


#### ********* TODO 任务三的部分 ******************
@app.route('/session3', methods=['GET', 'POST'])
def session3():
    from model.random_forrest import random_forrest
    filepath = request.args.get('filepath')
    x_cols = request.args.get('x_cols')
    y_col = request.args.get('y_col')
    depth = request.args.get('depth')
    if (filepath == None):
        filepath = request.json['filepath']
        x_cols = request.json['x_cols']
        y_col = request.json['y_col']
        depth = request.json['depth']
    train_score, test_score, df, savepath = random_forrest(filepath, x_cols, y_col, depth)
    return_dict = {
        'code': 200,
        'data': df,
        'savepath': savepath,
        'measure': {
            'train_score': train_score,
            'test_score': test_score
        }
    }
    return return_dict

@app.route('/gauss', methods=['GET', 'POST'])
def gauss():

    from model.GaussianNB import gaussModel, paintGauss, gaussPrediction
    filepath = request.args.get('filepath')
    x_cols = request.args.get('x_cols')
    y_col = request.args.get('y_col')
    if (filepath == None):
        filepath = request.json['filepath']
        x_cols = request.json['x_cols']
        y_col = request.json['y_col']
    # 如果没有选择因变量，则直接调用预测的方法
    if(y_col==None):
        # 这里直接调用预测的方法

        pass
    else:
        # 如果选择了因变量，则训练
        train_score, test_score, df, savepath = gaussModel(filepath, x_cols, y_col)
        var_list, train_list, test_list = paintGauss(filepath, x_cols, y_col)
    return_dict = {
        'code': 200,
        'data': df,
        'savepath': savepath,
        'measure': {
            'train_score': train_score,
            'test_score': test_score
        },
        'var_list': var_list,
        'train_list': train_list,
        'test_list': test_list
    }
    return return_dict

@app.route('/DT', methods=['GET', 'POST'])
def DT():
    from model.DT import DTModel, paintDT
    filepath = request.args.get('filepath')
    x_cols = request.args.get('x_cols')
    y_col = request.args.get('y_col')
    criterion = request.args.get('criterion')
    splitter = request.args.get('splitter')
    max_depth = request.args.get('max_depth')
    # min_samples_split = request.args.get('min_samples_split')
    # min_samples_leaf = request.args.get('min_samples_leaf')
    # min_weight_fraction = request.args.get('min_weight_fraction_leaf')
    # max_features = request.args.get('max_feature')
    # min_impurity_decrease = request.args.get('min_impurity_decrease')
    # ccp_alpha = request.args.get('ccp_alpha')
    if (filepath == None):
        filepath = request.json['filepath']
        x_cols = request.json['x_cols']
        y_col = request.json['y_col']
        criterion = request.json['criterion']
        splitter = request.json['splitter']
        max_depth = request.json['max_depth']
        # min_samples_split = request.json['min_samples_split']
        # min_samples_leaf = request.json['min_samples_leaf']
        # min_weight_fraction_leaf = request.json['min_weight_fraction_leaf']
        # max_feature = request.json['max_feature']
        # min_impurity_decrease = request.json['min_impurity_decrease']
        # ccp_alpha = request.json['ccp_alpha']
    train_score, test_score, df, savepath = DTModel(filepath, x_cols, y_col,
                                                    criterion=criterion,
                                                    splitter=splitter,
                                                    max_depth=max_depth,
                                                    # min_samples_split=min_samples_split,
                                                    # min_samples_leaf=min_samples_leaf,
                                                    # min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                    # max_feature=max_feature,
                                                    # min_impurity_decrease=min_impurity_decrease,
                                                    # ccp_alpha=ccp_alpha
                                                    )
    dataList, gini_train, gini_test, entropy_train, entropy_test, log_train, log_test = paintDT(filepath, x_cols, y_col)
    return_dict = {
        'code': 200,
        'data': df,
        'savepath': savepath,
        'measure': {
            'train_score': train_score,
            'test_score': test_score
        },
        'dataList': dataList,
        'gini_train': gini_train,
        'gini_test': gini_test,
        'entropy_train': entropy_train,
        'entropy_test': entropy_test
    }
    return return_dict


@app.route('/svc', methods=['GET', 'POST'])
def svc():
    from model.SVC import svcModel, paintSVM
    filepath = request.args.get('filepath')
    x_cols = request.args.get('x_cols')
    kernel = request.args.get('kernel')
    max_iter = request.args.get('max_iter')
    y_col = request.args.get('y_col')
    if (filepath == None):
        filepath = request.json['filepath']
        x_cols = request.json['x_cols']
        y_col = request.json['y_col']
        kernel = request.json['kernel']
        max_iter = request.json['max_iter']
    max_iter = int(max_iter)
    train_score, test_score, df, savepath = svcModel(filepath, x_cols, y_col, kernel, max_iter)
    dataList, linear_train, linear_test, poly_train, poly_test, rbf_train, rbf_test, sigmoid_train, sigmoid_test, precomputed_train, precomputed_test = paintSVM(
        filepath, x_cols, y_col)
    return_dict = {
        'code': 200,
        'data': df,
        'savepath': savepath,
        'measure': {
            'train_score': train_score,
            'test_score': test_score
        },
        'dataList': dataList,
        'linear_train': linear_train,
        'linear_test': linear_test,
        'poly_train': poly_train,
        'poly_test': poly_test,
        'rbf_train': rbf_train,
        'rbf_test': rbf_test,
        'sigmoid_train': sigmoid_train,
        'sigmoid_test': sigmoid_test,
        'precomputed_train': precomputed_train,
        'precomputed_test': precomputed_test
    }
    return return_dict

########## 下面的是第四问的所有的接口
@app.route('/session4getColumns', methods=['GET', 'POST'])
def session4getColumns():
    return_dict = {
        'code': 500,
        'columns': [],
        'data': []
    }
    dirpath = request.args.get('dirpath')
    if (dirpath == None):
        dirpath = request.json['dirpath']
    dirpath = dirpath.replace(dirpath.split('\\')[-1], "")
    filepath = os.path.join(dirpath, os.listdir(dirpath)[0])
    # 读取数据
    if (str(filepath)[-5:] == '.xlsx' or str(filepath)[-4:] == '.xls'):
        df = pd.read_excel(str(filepath))
    elif str(filepath)[-4:] == '.csv':
        df = pd.read_csv(str(filepath))
    elif str(filepath)[-4:] == '.txt':
        df = pd.read_csv(str(filepath), sep=' ')
    columns = df.columns
    # 组织列名格式
    tmp_columns = []
    for i in range(len(columns)):
        tmp = dict()
        tmp['prop'] = i + 1
        tmp['label'] = columns[i]
        tmp_columns.append(tmp)
    return_dict['code'] = 200
    return_dict['columns'] = tmp_columns
    return return_dict


@app.route('/session4FileAnalysis', methods=['GET', 'POST'])
def session4FileAnalysis():
    return_dict = {
        'code': 500,
        'date_list': [],
        'data_list': [],
        'evaluate': {
            'MSE': 0,
            'RMSE': 0,
            'MAE': 0,
            'MAPE': 0,
            'r2': 0,
        }
    }
    filepath = request.args.get('filepath')
    date_col = request.args.get('date_col')
    x_cols = request.args.get('x_cols')
    y_col = request.args.get('y_col')
    step = request.args.get('step')
    func = request.args.get('func')
    if (filepath == None):
        filepath = request.json['filepath']
        date_col = request.json['date_col']
        x_cols = request.json['x_cols']
        y_col = request.json['y_col']
        step = request.json['step']
        func = request.json['func']

    from model.lstm_pre import main, read_date
    y_cols = []
    y_cols.append(y_col)
    step = int(step)
    rawData, predictData, MSE, RMSE, MAE, MAPE, R2 = main(filepath, y_cols, step)
    return_dict['code'] = 200
    return_dict['data_list'] = rawData[-100:-1]
    return_dict['date_list'] = read_date(filepath, date_col)[-100:-1]
    return_dict['evaluate']['MSE'] = MSE
    return_dict['evaluate']['RMSE'] = RMSE
    return_dict['evaluate']['MAE'] = MAE
    return_dict['evaluate']['MAPE'] = MAPE
    return_dict['evaluate']['R2'] = R2
    return return_dict


@app.route('/session4PreOne', methods=['GET', 'POST'])
def session4PreOne():
    from model.lstm_pre import predict_one_step
    data = request.args.get('data')
    date = request.args.get('date')
    if (data == None):
        data = request.json['data']
        date = request.json['date']
    print(date, data)
    Predict, tmp_day = predict_one_step(date=date, data=data)
    print(Predict, tmp_day)
    return_dict = {
        'code': 200,
        'data': sum(list(Predict[0])) / 3,
        'newDay': tmp_day,
    }
    return return_dict

@app.route('/session4ReadData', methods=['GET', 'POST'])
def session4ReadData():
    dirpath = request.args.get('dirpath')
    load_col = request.args.get('load_col')
    date_col = request.args.get('date_col')
    if (date_col == None):
        dirpath = request.json['dirpath']
        load_col = request.json['load_col']
        date_col = request.json['date_col']

    # 这里的 dirpath 需要更改一下
    dirpath = dirpath.replace("\\" + dirpath.split('\\')[-1], "")
    data, date, all_data, all_date, MSE, RMSE, MAE, MAPE, R2, subFile = read_dir(dirpath, date_col, load_col)
    return {
        "code": 200,
        "date": date,
        "data": data,
        "data_length": len(data),
        "all_data": all_data,
        "all_date": all_date,
        "subFile": subFile,
        "evaluate": {
            "MSE": MSE,
            "RMSE": RMSE,
            "MAE": MAE,
            "MAPE": MAPE,
            "R2": R2
        }
    }

@app.route('/session4PreThree', methods=['GET', 'POST'])
def session4PreThree():
    dirpath = request.args.get('dirpath')
    load_col = request.args.get('load_col')
    date_col = request.args.get('date_col')
    if (date_col == None):
        dirpath = request.json['dirpath']
        load_col = request.json['load_col']
        date_col = request.json['date_col']
    dirpath = dirpath.replace(dirpath.split('\\')[-1], "")
    from model.lstm_pre import pre_three
    return pre_three(dirpath, date_col, load_col)


########## 下面的是第5问的所有的接口
@app.route('/session5ReadData', methods=['GET', 'POST'])
def session5ReadData():
    filepath = request.args.get('filepath')
    del_percent = request.args.get('del_percent')
    func = request.args.get('func')
    if (filepath == None):
        filepath = request.json['filepath']
        del_percent = request.json['del_percent']
        func = request.json['func']
    from model.DBSCAN import kmeans_data
    del_percent = float(del_percent)
    return_dict = kmeans_data(filepath, del_percent, func)
    return_dict['code'] = 200
    return return_dict


@app.route('/session5Columns', methods=['GET', 'POST'])
def session5Columns():
    filepath = request.args.get('filepath')
    del_percent = request.args.get('del_percent')
    func = request.args.get('func')
    if (filepath == None):
        filepath = request.json['filepath']
        del_percent = request.json['del_percent']
        func = request.json['func']
    del_percent = float(del_percent)
    from model.DBSCAN import kmeans, Dataframe2Json
    df = kmeans(filepath, del_percent, func)
    return_dict = Dataframe2Json(df)
    return_dict['code'] = 200
    return return_dict


@app.route('/session5PaintOne', methods=['GET', 'POST'])
def session5PaintOne():
    filepath = request.args.get('filepath')
    del_percent = request.args.get('del_percent')
    prop_col = request.args.get('prop_col')
    func = request.args.get('func')
    if (filepath == None):
        filepath = request.json['filepath']
        del_percent = request.json['del_percent']
        prop_col = request.json['prop_col']
        func = request.json['func']
    del_percent = float(del_percent)
    from model.DBSCAN import paintOne
    content = paintOne(filepath, del_percent, prop_col, func)
    length = len(content[0]) - 1
    return_dict = {
        'code': 200,
        'length': length,
        'content': content
    }
    return return_dict


@app.route('/session5Weight', methods=['GET', 'POST'])
def session5Weight():
    filepath = request.args.get('filepath')
    del_percent = request.args.get('del_percent')
    func = request.args.get('func')
    if (filepath == None):
        filepath = request.json['filepath']
        del_percent = request.json['del_percent']
        func = request.json['func']
    del_percent = float(del_percent)
    from model.DBSCAN import allWeight
    return_dict = allWeight(filepath, del_percent, func)
    X = np.array(return_dict['data'])
    from matplotlib import pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Used to display Chinese labels normally
    plt.rcParams['axes.unicode_minus'] = False  # Used to display negative signs normally
    plt.figure(dpi=300, figsize=(10, 9))
    plt.subplot(411)
    plt.yticks(range(4), labels=['高价值客户', '大众型客户', '潜力型客户', '低价值客户'])
    plt.imshow(X[:, 0:20])
    plt.colorbar()
    plt.subplot(412)
    plt.yticks(range(4), labels=['高价值客户', '大众型客户', '潜力型客户', '低价值客户'])
    plt.imshow(X[:, 20:40])
    plt.colorbar()
    plt.subplot(413)
    plt.yticks(range(4), labels=['高价值客户', '大众型客户', '潜力型客户', '低价值客户'])
    plt.imshow(X[:, 40:60])
    plt.colorbar()
    plt.subplot(414)
    plt.yticks(range(4), labels=['高价值客户', '大众型客户', '潜力型客户', '低价值客户'])
    plt.imshow(X[:, 60:80])
    plt.colorbar()
    plt.savefig(r"F:\study\com\softwareCup\code\template\test3\ev-project\src\renderer\assets\images\热力图.png")
    return return_dict


##### 第五问的报告中需要的图
@app.route('/sessionReportChart', methods=['GET', 'POST'])
def sessionReportChart():
    filepath = request.args.get('filepath')
    del_percent = request.args.get('del_percent')
    func = request.args.get('func')
    if (filepath == None):
        filepath = request.json['filepath']
        del_percent = request.json['del_percent']
        func = request.json['func']
    del_percent = float(del_percent)
    from model.DBSCAN import return_weight
    return_dict = return_weight(filepath, del_percent, func)
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
    return {
        "code":200,
        "x": data_x,
        "y": data_y,
    }

##### 任务四导出模型部分
@app.route('/downloadLSTM', methods=['GET', 'POST'])
def downloadLSTM():
    response = send_from_directory('./model/model_data', "企业电力营销模型.h5", as_attachment=True)
    return response

##### 任务五导出模型部分
@app.route('/downloadDBSCAN', methods=['GET', 'POST'])
def downloadDBSCAN():
    response = send_from_directory('./model/model_data', "电力用户集群分析模型.mdl", as_attachment=True)
    return response

@app.route('/cluster3d', methods=['GET', 'POST'])
def cluster3d():
    filepath = request.args.get('filepath')
    del_percent = request.args.get('del_percent')
    func = request.args.get('func')
    if (filepath == None):
        filepath = request.json['filepath']
        del_percent = request.json['del_percent']
        func = request.json['func']
    del_percent = float(del_percent)
    from model.DBSCAN import paint3d
    savepath = paint3d(filepath, del_percent, func)
    request.post("http://127.0.0.1:5000/uploadfile")
    return {
        "code": 200,
        "savepath": savepath,
    }
@app.route('/uploadfile', methods=['GET', 'POST'])
def uploadfile():
    '''
    上传文件的服务
    :return:
    '''
    try:
        # file = request.files.get('file')
        filepath = request.form.get('path')
        if (filepath == None):
            filepath = request.json['path']
        # name = str(file.filename)
        # file.save(os.path.join('uploadFiles', name))
        # 将最近访问过的文件放入数据库
        sql = "SELECT filepath from lastpath ORDER BY id DESC limit 1;"
        results = db.session.execute(sql)
        data = []
        for i in results:
            data.append(i[0])
        if (filepath not in data):
            sql = "insert into lastpath(filepath) values (:filepath);"
            db.session.execute(sql, {'filepath': filepath})
        return {'code': 200}
    except:
        return {'code': 500}

##### 地图动态效果展示
@app.route('/getMapData', methods=['GET', 'POST'])
def getMapData():
    from model.LSTM import return_map_data
    data = []
    name = ['Carlow', 'Cavan', 'Clare', 'Cork', 'Donegal', 'Dublin', 'Galway', 'Kerry', 'Kildare', 'Kilkenny', 'Laois',
            'Leitrim', 'Limerick', 'Longford', 'Louth', 'Mayo', 'Meath', 'Monaghan', 'Offaly', 'Roscommon', 'Sligo', 'Tipperary',
            'Tipperary', 'Waterford', 'Westmeath', 'Wexford', 'Wicklow']
    for i in name:
        tmp = {
            "name": i,
            "value": return_map_data(i)
        }
        data.append(tmp)
    return {'code':200,'data':data}
##### 地图动态效果展示
@app.route('/getMapChangeData', methods=['GET', 'POST'])
def getMapChangeData():
    from model.LSTM import return_mapChange_data
    data = []
    name = ['Carlow', 'Cavan', 'Clare', 'Cork', 'Donegal', 'Dublin', 'Galway', 'Kerry', 'Kildare', 'Kilkenny', 'Laois',
            'Leitrim', 'Limerick', 'Longford', 'Louth', 'Mayo', 'Meath', 'Monaghan', 'Offaly', 'Roscommon', 'Sligo', 'Tipperary',
            'Tipperary', 'Waterford', 'Westmeath', 'Wexford', 'Wicklow']
    text = []
    for i in name:
        tmp = {
            "name": i,
            "value": return_mapChange_data(i)[0],
        }
        text.append(return_mapChange_data(i)[1])
        data.append(tmp)
    return {'code':200,'data':data, 'text': text}



if __name__ == '__main__':
    app.run()
