import os

from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
import json
import random
import yagmail
import threading
import time

app = Flask(__name__)
# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://team:123456@120.25.148.251:3306/software_cup"
#设置这一项是每次请求结束后都会自动提交数据库中的变动
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN']=True
#实例化
db = SQLAlchemy(app)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/getsche', methods=['GET', 'POST'])
def getsche():
    try:
        sql = "select id, time, person, content from scheduleT;"
        results = db.session.execute(sql)
        data = []
        for i in results:
            tmp = {}
            tmp['id'] = str(i[0])
            tmp['time'] = str(i[1])
            tmp['person'] = i[2]
            tmp['content'] = i[3]
            data.append(tmp)
        return_dict = {
            'code': 404,
            'data': []
        }
        return_dict['data'] = data
        return_dict['code'] = 200
    except:
        return_dict['code'] = 500
    return json.dumps(return_dict)


@app.route('/insertsche', methods=['GET', 'POST'])
def insertsche():
    '''
    插入记录
    :return: 
    '''
    try:
        Stime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        person = request.args.get('person')
        content = request.args.get('content')
        if (person == None):
            person = request.json['person']
            content = request.json['content']
        sql = "insert into scheduleT(time, person, content) values (:time, :person, :content);"
        db.session.execute(sql, {'time': Stime, 'person': person, 'content':content})
        return {'code': 200}
    except:
        return {'code': 500}


@app.route('/deletesche', methods=['GET', 'POST'])
def deletesche():
    try:
        id = request.args.get('id')
        if (id == None):
            id = request.json['id']
        sql = "delete from scheduleT where id = :id;"
        db.session.execute(sql, {'id': id})
        return {'code': 200}
    except:
        return {'code': 500}


@app.route('/update', methods=['GET', 'POST'])
def update():
    '''
    是否更新软件的函数
    :return: 
    '''
    sql = "SELECT version, imgurl, content from isupdate ORDER BY id DESC limit 1;"
    results = db.session.execute(sql)
    data = []
    for i in results:
        tmp = {}
        tmp['version'] = str(i[0])
        tmp['imgurl'] = str(i[1])
        tmp['content'] = i[2]
        data.append(tmp)
    return_dict = {
        'code': 404,
        'data': []
    }
    return_dict['data'] = data
    return_dict['code'] = 200

    return return_dict


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    '''
    上传服务
    :return:
    '''
    file = request.files.get('files')
    name = str(file.filename)
    file.save(os.path.join('utils', name))
    return {'code':200}


@app.route('/getfiledir', methods=['GET', 'POST'])
def getfiledir():
    '''
    获取文件的列表
    '''
    return_dict = {
        "code": 404,
        "data": []
    }
    for i in os.listdir('./utils'):
        return_dict['data'].append(i)
    return_dict['code'] = 200
    return json.dumps(return_dict)


@app.route('/deletefile', methods=['GET', 'POST'])
def deletefile():
    '''
    删除文件的utils
    :return: 
    '''
    name = request.args.get('name')
    if (name == None):
        name = request.json['name']
    os.remove(os.path.join('./utils', name))
    return {'code':200}


## %% 这里是任务一的调用函数的接口
@app.route('/getfilecolumns', methods=['GET', 'POST'])
def getfilecolumns():
    '''返回文件的列名和内容'''
    return_dict = {'code': 500}
    from utils.session1 import getColumns
    filepath = request.args.get('filepath')
    if (filepath == None):
        filepath = request.json['filepath']
    print(filepath)
    columns, content = getColumns(filepath)
    content = content.values.tolist()
    if( type(columns)!=bool and len(columns)>0):
        return_dict = {
            'code': 200,
            'columns': columns,
            'data': content,
        }
    return return_dict


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


## %%
# 第一行是固定的，但是需要改一下接口的名字
@app.route('/leaveword', methods=['GET', 'POST'])
# 方法和上面接口名字是一样的
def leaveword():
    try:
        # 获取前端传来的参数
        word = request.args.get('word')
        if (word == None):
            word = request.json['word']
        # 插入到数据库
        # 1 写sql代码
        sql = "insert into leaveword(content) values (:word);"
        # 将sql代码提交到数据库执行，第二个参数与上述的values冒号后的变量对应
        db.session.execute(sql, {'word': word})
        return {'code': 200}
    except:
        return {'code': 500}


#   查询留言
@app.route('/selectword', methods=['GET', 'POST'])
def selectword():
    sql = "SELECT content from leaveword ORDER BY id DESC limit 1;"
    results = db.session.execute(sql)
    data = []
    for i in results:
        tmp = {}
        tmp['content'] = i[0]
        data.append(tmp)
    return_dict = {
        'code': 404,
        'data': []
    }
    return_dict['data'] = data
    return_dict['code'] = 200

    return return_dict


#  删除留言
@app.route('/deleteword', methods=['GET', 'POST'])
def deleteword():
    try:
        id = request.args.get('id')
        if (id == None):
            id = request.json['id']
        sql = "delete from leaveword where id = :id;"
        db.session.execute(sql, {'id': id})
        return {'code': 200}
    except:
        return {'code': 500}


#  软件说明
@app.route('/instruction', methods=['GET', 'POST'])
def instruction():
    sql = "SELECT version, imgurl, content from isupdate ORDER BY id DESC;"
    results = db.session.execute(sql)
    data = []
    for i in results:
        tmp = {}
        tmp['version'] = str(i[0])
        tmp['imgurl'] = str(i[1])
        tmp['content'] = i[2]
        data.append(tmp)
    return_dict = {
        'code': 404,
        'data': []
    }
    return_dict['data'] = data
    return_dict['code'] = 200

    return return_dict






##### 备份
@app.route('/uploaddir', methods=['GET', 'POST'])
def uploaddir():
    '''
    上传文件夹的服务
    :return:
    '''
    try:
        filepath = request.form.get('path')
        file = request.files.get('file')
        name = str(file.filename)
        # 新建文件夹
        tmp_path = filepath.split('\\')[-2]
        tmp_path = os.path.join('./uploadDir', tmp_path)
        if (not os.path.exists(tmp_path)):
            os.makedirs(tmp_path)
        # 保存文件在服务器
        file.save(os.path.join(tmp_path, name))
        # 将最近访问过的文件放入数据库
        tmp_path = filepath.split('\\')
        del tmp_path[-1]
        dirpath = ''
        for i in tmp_path:
            dirpath = os.path.join(dirpath, i)
        sql = "SELECT filepath from lastpath ORDER BY id DESC limit 1;"
        results = db.session.execute(sql)
        # 对比重复数据，最后一行重复的记录就不加入数据库了
        data = []
        for i in results:
            data.append(i[0])
        if (dirpath not in data):
            sql = "insert into lastpath(filepath) values (:dirpath);"
            db.session.execute(sql, {'dirpath': dirpath})
        return {'code': 200}
    except:
        return {'code': 500}
@app.route('/getfiledir', methods=['GET', 'POST'])
def getfiledir():
    '''
    获取文件的列表(没什么用了)
    '''
    return_dict = {
        "code": 404,
        "data": []
    }
    for i in os.listdir('./utils'):
        return_dict['data'].append(i)
    return_dict['code'] = 200
    return json.dumps(return_dict)
@app.route('/uploadfile', methods=['GET', 'POST'])
def uploadfile():
    '''
    上传文件的服务(最近路径)
    :return:
    '''
    try:
        file = request.files.get('file')
        filepath = request.form.get('path')
        name = str(file.filename)
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

@app.route('/deletefile', methods=['GET', 'POST'])
def deletefile():
    '''
    删除文件的utils
    :return:
    '''
    name = request.args.get('name')
    if (name == None):
        name = request.json['name']
    os.remove(os.path.join('./utils', name))
    return {'code': 200}


if __name__ == '__main__':
    app.run()
