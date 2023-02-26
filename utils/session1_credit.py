# -*- coding: utf-8 -*-
#@Time:2022/8/1 14:01
#@Author: FZY
#@File:session1_credit.py
#@Software:Pycharm
#@Describe:
import pandas as pd
import numpy as np


# 计算信用的代码
def cal_credit(filepath):
    df = pd.read_excel(filepath)
    ### 计算部分，计算出了信用分数
    tuoDays = np.array(list(df['拖欠天数']))
    MaxDays = np.array(list(df['允许拖欠天数']))
    creditScore = 1 - tuoDays / (2 * MaxDays)
    creditScore = creditScore.tolist()
    ### 汇总部分
    result = pd.DataFrame({
        "用户编号": df['用户编号'],
        '信用分数': creditScore
    })
    userid = result['用户编号'].unique().tolist()
    result = result.groupby("用户编号")['信用分数'].mean().tolist()
    ### 获取信用前10位用户的id和信用分
    user = []
    score = []
    rank = [index for index, value in sorted(list(enumerate(result)), key=lambda x: x[1], reverse=True)]
    for i in range(0, 10):
        user.append(str(userid[rank[i]]))
        score.append(result[rank[i]])
    return creditScore, user, score




if __name__ == '__main__':
    filepath = r"F:\study\com\softwareCup\website\back\file\用户缴费行为测试数据集1.xlsx"
    cal_credit(filepath)


