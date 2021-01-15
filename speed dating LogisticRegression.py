# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:39:54 2021

@author: 元元吃汤圆
"""


#%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

import imblearn
from palettable.colorbrewer.qualitative import Pastel1_3

df = pd.read_csv('D:/data/speed_dating_train.csv', encoding='gbk')
percent_missing = df.isnull().sum() * 100 / len(df) #计算出每一列缺失值的百分值
missing_value_df = pd.DataFrame({
    'column_name': df.columns,
    'percent_missing': percent_missing
}) #创建一个dataframe，显示出每一列缺失值的百分值
missing_value_df.sort_values(by='percent_missing') #并且按照百分值从小到大排列



"""探索性数据分析"""
# 多少人通过Speed Dating找到了对象
plt.subplots(figsize=(3,3), dpi=110) #figsize图像大小，像素是300*300，100分辨率的图形
# 构造数据
size_of_groups = df.match.value_counts().values  
#value_counts()是一种查看表格某列中有多少个不同值的快捷方法，并计算每个不同值有在该列中有多少重复值。
#此时查看数据集match的值及其频率。显示结果：[6922,1355]
single_percentage=round(size_of_groups[0]/sum(size_of_groups) * 100,2) #四舍五入
matched_percentage = round(size_of_groups[1]/sum(size_of_groups)* 100,2) 
names = [
    'Single:' + str(single_percentage) + '%',
    'Matched' + str(matched_percentage) + '%'] 
# 创建饼图
plt.pie(
    size_of_groups, 
    labels=names, 
    labeldistance=1.2, 
)
plt.show()



"""匹配成功率是否和性别有关（个人认为此探索无用，比率受到参加男女奇数的影响），
女生是否脱单率会更高呢。女生匹配几率16.34%，男生16.4%"""

df[df.gender == 0] #查看女生的数据
# 多少女生通过Speed Dating找到了对象
plt.subplots(figsize=(3,3), dpi=110,)
# 构造数据
size_of_groups=df[df.gender == 0].match.value_counts().values # 男生只需要吧0替换成1即可

single_percentage = round(size_of_groups[0]/sum(size_of_groups) * 100,2) 
matched_percentage = round(size_of_groups[1]/sum(size_of_groups)* 100,2) 
names = [
    'Single:' + str(single_percentage) + '%',
    'Matched' + str(matched_percentage) + '%']
 
# 创建饼图
plt.pie(
    size_of_groups, 
    labels=names, 
    labeldistance=1.2, 
)
plt.show()



"""是什么样的人在参加快速相亲这样的活动呢？真的都是大龄青年（年龄大于30）嘛？画出年龄分布图
此问题不用做分析也可以回答，当然不是。因为教授做数据采集时参加的都是该大学的师生呀！主要分布在20~30岁"""
age = df[np.isfinite(df['age'])]['age']
plt.hist(age,bins=35)
plt.xlabel('Age')
plt.ylabel('Frequency')



date_df = df[[
    'iid', 'gender', 'pid', 'match', 'int_corr', 'samerace', 'age_o',
       'race_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb',
       'pf_o_sha', 'dec_o', 'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'like_o',
       'prob_o', 'met_o', 'age', 'race', 'imprace', 'imprelig', 'goal', 'date',
       'go_out', 'career_c', 'sports', 'tvsports', 'exercise', 'dining',
       'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
       'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'attr1_1',
       'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'attr3_1', 'sinc3_1',
       'fun3_1', 'intel3_1', 'dec', 'attr', 'sinc', 'intel', 'fun', 'like',
       'prob', 'met'
]]

# heatmap
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = date_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)




# preparing the data
clean_df = df[['attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o','match']]
clean_df.dropna(inplace=True) #删除缺失值
X=clean_df[['attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o',]]
y=clean_df['match']

oversample = imblearn.over_sampling.SVMSMOTE() #数据有严重的不均衡，这里我们可以用SVMSMOTE来增加一下我们的数据量避免模型出现过度拟合。
X, y = oversample.fit_resample(X, y)

# 做训练集和测试集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


# logistic regression classification model
model = LogisticRegression(C=1, random_state=0)
lrc = model.fit(X_train, y_train)
predict_train_lrc = lrc.predict(X_train)
predict_test_lrc = lrc.predict(X_test)
print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_lrc))
print('Validation Accuracy:', metrics.accuracy_score(y_test, predict_test_lrc))

lrc.predict_proba([[8.0,6.0,7.0,7.0,6.0,8.0,]])
