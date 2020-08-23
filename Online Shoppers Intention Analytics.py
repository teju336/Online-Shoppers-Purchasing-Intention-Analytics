# -*- coding: utf-8 -*-
"""
ALY6140 Group 10
Online Shoppers Purchasing Intention Analytics
Logistic Regression, Random Forest, SVM
Written by <Tejaswini_Somasundar, Phanindar_Golla>
"""
# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc

# Read the dataset
def 
shoppers = pd.read_csv('online_shoppers_intention.csv')

 #Get the description of the dataset
print(shoppers.head())
print(shoppers.describe())
print(shoppers.info())

# Data cleansing
shoppers['Administrative'] = shoppers['Administrative'].astype('category')
shoppers['Informational'] = shoppers['Informational'].astype('category')
shoppers['ProductRelated'] = shoppers['ProductRelated'].astype('category')
shoppers['OperatingSystems'] = shoppers['OperatingSystems'].astype('category')
shoppers['Browser'] = shoppers['Browser'].astype('category')
shoppers['Region'] = shoppers['Region'].astype('category')
shoppers['TrafficType'] = shoppers['TrafficType'].astype('category')
shoppers['Weekend'] = shoppers['Weekend'].astype('int')
shoppers['Revenue'] = shoppers['Revenue'].astype('int')
print(shoppers.info())

# Find out the columns that need to be removed
shoppers_corr = shoppers.corr()
sns.heatmap(shoppers_corr)

# Remove the columns that have low correlation with Revenue
shoppers.drop(['Administrative', 'Informational_Duration', 'ProductRelated'],inplace = True,axis = 1)
#EDA
sns.barplot(x='Month',y='Revenue',data=shoppers)
sns.countplot(x='VisitorType',data=shoppers)
sns.violinplot(x='Browser',y='ExitRates',data=shoppers)

# Split the dataset
x = shoppers.drop('Revenue', axis=1)
y = shoppers['Revenue']
x = pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

""" Logistic Regression """

# Model Training
logis = LogisticRegression(random_state = 50, max_iter = 5000)
lr = logis.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

# Model Evaluation
m = confusion_matrix(y_test,y_pred_lr)
print("Logistic Regression:")
print("Accuracy: ", (m[0][0] + m[1][1]) / (m[0][0] + m[0][1] + m[1][0] + m[1][1]))
print(classification_report(y_test,y_pred_lr))

# Draw the ROC and AUC of LR model
logit_roc_auc = roc_auc_score(y_test, logis.predict(x_test))
fpr1, tpr1, thresolds1 = roc_curve(y_test, logis.predict_proba(x_test)[:,1])
plt.figure()
roc_auc1 = auc(fpr1, tpr1)
plt.plot(fpr1, tpr1, label = 'Logistic Regression (area = %0.2f)' % roc_auc1)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Logis ROC')
plt.show()

""" Random Forest """

# Model Training
rf = RandomForestClassifier(random_state = 0)
rf_model = rf.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

# Model Evaluation
rf_cm = confusion_matrix(y_test, y_pred_rf)
print("Random Forest: ")
print("Accuracy: ", (rf_cm[0][0] + rf_cm[1][1]) / (rf_cm[0][0] + rf_cm[0][1] + rf_cm[1][0] + rf_cm[1][1]))
print(classification_report(y_test, y_pred_rf))

# Draw the ROC and AUC of RF model
rf_roc_auc = roc_auc_score(y_test, rf_model.predict(x_test))
fpr2, tpr2, thresolds2 = roc_curve(y_test, rf_model.predict_proba(x_test)[:,1])
plt.figure()
roc_auc2 = auc(fpr2, tpr2)
plt.plot(fpr2, tpr2, label = 'Random Forest (area = %0.2f)' % roc_auc2)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Random Forest ROC')
plt.show()

""" SVM """
print("Default params of SVM:")
print(SVC().get_params())
# Check out the parameters of SVM
"""" Remove the comment to find the best parameters of SVM model. It may take a few minutes to run. 
#print("Default params of SVM:")
#print(SVC().get_params())

# Tuning the model
svc = SVC(kernel='rbf')
parameters = {'gamma':[0.0001, 0.0005, 0.001], 'C':[0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4]}
svc_grid = GridSearchCV(svc, param_grid=parameters, cv=5)
svc_grid.fit(x, y)
print("CV Results:" )
print(svc_grid.cv_results_)
print("Best Params:")
print(svc_grid.best_params_)  
print("Best Score:")
print(svc_grid.best_score_)
"""

# Select the optimal parameters for training
svc = SVC(kernel='rbf', C=0.8, gamma= 0.0001, probability=True)
svc.fit(x_train, y_train)
y_pred_SVM = svc.predict(x_test)
yhat = svc.predict(x_train)

# Model Evaluation
np.mean(y_train==yhat)
np.mean(y_test==y_pred_SVM)
svm_cm = confusion_matrix(y_test, y_pred_SVM)
print("SVM: ")
print("Accuracy: ", (svm_cm[0][0] + svm_cm[1][1]) / (svm_cm[0][0] + svm_cm[0][1] + svm_cm[1][0] + svm_cm[1][1]))
print(classification_report(y_test, y_pred_SVM))

# Draw the ROC and AUC of SVM model
svm_roc_auc = roc_auc_score(y_test, svc.predict(x_test))
fpr3, tpr3, thresolds3 = roc_curve(y_test, svc.predict_proba(x_test)[:,1])
plt.figure()
roc_auc3 = auc(fpr3, tpr3)
plt.plot(fpr3, tpr3, label = 'SVM (area = %0.2f)' % roc_auc3)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('SVM ROC')
plt.show()
"""
y_prob = svc.predict_proba(x_test)
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
plt.figure()
plt.plot(fpr, tpr, color='r', lw=2)
auc = roc_auc_score(y_test, y_prob[:, 1])
print("AUC of SVM:")
print(auc)
"""

# Draw all roc_auc in one figure
Font={'size':12, 'family':'Times New Roman'}
plt.figure()
plt.plot(fpr1, tpr1, 'b', label = 'Logistic Reg. = %0.2f' % roc_auc1, color='g')
plt.plot(fpr2, tpr2, 'b', label = 'Random Forest = %0.2f' % roc_auc2, color='r')
plt.plot(fpr3, tpr3, 'b', label = 'SVM = %0.2f' % roc_auc3, color='b')
plt.legend(loc = 'lower right', prop=Font)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive Rate', Font)
plt.xlabel('False Positive Rate', Font)
plt.tick_params(labelsize=15)
plt.show()
best_score = abs(fpr1 - tpr1).max(),abs(fpr2 - tpr2).max(),abs(fpr3 - tpr3).max()
print("The ROC of each model is: ")
print("Logistic Regression, Random Forest, SVM")
print(best_score)
