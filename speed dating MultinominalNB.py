# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:22:56 2021

@author: 元元吃汤圆
"""


import numpy as np
import csv
import scipy
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
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
MultNB = MultinomialNB(alpha=0.95)
parameters={'class_prior':[[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6]]}
grid_search= GridSearchCV(MultNB,parameters,scoring='roc_auc',cv=5)

grid_search.fit(x_train,y_train)
print("best_score",grid_search.best_score_)
print("best_params",grid_search.best_params_)
#将最好的赋值给knn_clf
MultNB_clf=grid_search.best_estimator_
MultNB_clf.fit(x_train,y_train) #用最好的参数模拟
roc_auc_score(y_test,MultNB_clf.predict_proba(x_test)[:,1])


#ROC可视化
predictions_validation = MultNB_clf.predict_proba(x_test)[:,1]
fpr,tpr,_ = roc_curve(y_test,predictions_validation)
roc_auc = auc(fpr,tpr)
plt.title('ROC Validation')
plt.plot(fpr,tpr,color='darkorange',lw=2,label='AUC = %0.2f'%roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--');
plt.legend(loc = 'lower right')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()