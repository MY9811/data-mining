# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:05:07 2021

@author: 元元吃汤圆
"""


import numpy as np
import csv
import scipy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import imblearn

#读取数据
df = pd.read_csv(open('D:/data/speed_dating_train.csv'))
df.head()

# 放入6个特征并去除缺失值
clean_df = df[['attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o','match']]
clean_df.dropna(inplace=True) #删除缺失值
X=clean_df[['attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o',]]
y=clean_df['match']

#过采样
oversample = imblearn.over_sampling.SVMSMOTE() #数据有严重的不均衡，这里我们可以用SVMSMOTE来增加一下我们的数据量避免模型出现过度拟合。
X, y = oversample.fit_resample(X, y) 

# 做训练集和测试集分割
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

#随机森林，未调参
rfc = RandomForestClassifier()
rfc = rfc.fit(x_train,y_train) #用训练数据拟合分类器模型  y_train是x_train的标签
result = rfc.score(x_test,y_test) #求准确率
roc_auc_score(y_test,rfc.predict_proba(x_test)[:,1]) #求ROC分数，更靠谱

#可视化各feature的重要性
print("各feature的重要性：%s" % rfc.feature_importances_) #加起来比率为1
importances = rfc.feature_importances_
std =np.std([tree.feature_importances_ for tree in rfc.estimators_],axis=0)
indices = np.argsort(importances)[::-1]#使顺序变成从大到小
print("Feauture Ranking")
for f in range(min(20,x_train.shape[1])):
    print("%2d) %-*s %f" %(f + 1,30, x_train.columns[indices[f]],importances[indices[f]]))
    
plt.figure()
plt.title("Feature Importances")
plt.bar(range(x_train.shape[1]),importances[indices],color="blue",yerr=std[indices],align="center") #柱体在 x 轴上的坐标位置,柱体的高度
plt.xticks(range(x_train.shape[1]),x_train.columns[indices]) #一个是刻标(locs)，另一个是刻度标签
plt.xlim([-1,x_train.shape[1]]) #输出x轴的范围值
plt.show()


#网格搜索，超参数调优
param_test1 = {'n_estimators':range(25,500,25),
               'min_samples_split':range(60,200,20),
               'min_samples_leaf':range(25,500,25),
               'max_depth':range(3,30,2),
               'criterion':['gini','entropy']} #创建一个关于分类器的数量的字典，范围是25-499，步长25
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(random_state=0), 
    param_grid = param_test1,
    scoring='roc_auc',
    cv=5) 
# GridSearchCV参数是分类器，超参数，评分方式
#min_samples_split 分割内部节点所需要的最小样本数量
#min_samples_leaf 需要在叶子结点上的最小样本数量. max_depth:数的最大深度
#cv交叉验证是几折

#用调优后的模型拟合数据
gsearch1.fit(x_train,y_train)
print(gsearch1.beat_params_, gsearch1.best_score_)
best_rcf = gsearch1.best_estimator_ #从中取最好的参数来建立模型
best_rcf.fit(x_train,y_train) 
roc_auc_score(y_test,best_rcf.predict_proba(x_test)[:,1]) #求超参数调优后的roc值，可以与未调参的随机森林ROC值比较。



#ROC可视化
predictions_validation = gsearch1.best_estimator_.predict_proba(x_test)[:,1]
fpr,tpr,_ = roc_curve(y_test,predictions_validation)
roc_auc = auc(fpr,tpr)
plt.title('ROC Validation')
plt.plot(fpr,tpr,color='darkorange',lw=2,lable='AUC = %0.2f'%roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--');
plt.legend(loc = 'lower right')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()









#备用：用交叉验证挑选最佳模型
clf = DecisionTreeClassifier(max_depth = None, min_samples_split = 2,random_state=0) #参数如何设置
scores1 = cross_val_socre(clf,x_train,y_train)
print(scores1.mean)
scores2 = cross_val_socre(rcf,x_train,y_train)
print(scores2.mean)





