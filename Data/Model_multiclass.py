import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost 
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
#from imblearn.over_sampling import SMOTE

import glob
brand_dict = dict()

brand_dict["Lancome"] = pd.read_csv("Lancome_features.csv")
brand_dict["Lancome"].dropna(inplace=True)


print(len(brand_dict["Lancome"].columns), brand_dict["Lancome"].columns)



brand_dict["Lancome"].sort_values(by="Date", inplace=True)
X = brand_dict["Lancome"][[
       'month', 'day', 'weekday', 'avg_discount', 
        'discount_past1day', 'discount_past2day',
       'discount_past3day', 'discount_past4day', 'discount_past5day',
       'discount_past6day', 'discount_past7day', 'discount_min_past15day',
       'discount_min_past30day', 'discount_max_past15day',
       'discount_max_past30day', 'discount_mean_past15day',
       'discount_mean_past30day', 'nday_away_Martin_Luther_King_Jr_Day',
       'nday_away_New_Years_Day', 'nday_away_Memorial_Day',
       'nday_away_Columbus_Day', 'nday_away_Veterans_Day',
       'nday_away_Independence_Day', 'nday_away_Labor_Day',
       'nday_away_Washingtons_Birthday', 'nday_away_Thanksgiving',
       'nday_away_Christmas_Day', 'nday_away_anyholiday']]
print(len(X.columns))




features_n = brand_dict["Lancome"][[ 
       'month', 'day', 'weekday', 'nday_away_anyholiday', 
       'discount_past1day', 'discount_past2day',
       'discount_past3day', 'discount_past4day', 'discount_past5day',
       'discount_past6day', 'discount_past7day',
       'discount_max_past15day', 'discount_max_past30day', 'discount_mean_past15day', 'discount_mean_past30day']]
corr = features_n.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")
plt.figure(figsize = (32,15))
ax = sns.heatmap(corr, mask=mask, annot=True,
            square=True, linewidths=1,cbar=True)
ax.set_yticklabels(
    ['Month of year', 'Day of year', 'Day of week', 'Distance to holidays',
     'Discount past 1 day', 'Discount past 2 days', 'Discount past 3 days', 
     'Discount past 4 days', 'Discount past 5 days', 'Discount past 6 days', 'Discount past 7 days',
     'Max discount (past 15 days)', 'Max discount (past 30 days)', 
     'Average discount (past 15 days)', 'Average discount (past 30 days)'],
    fontsize = 20,
    verticalalignment = 'center'
);
ax.set_xticklabels(
    ['Month of year', 'Day of year', 'Day of week', 'Distance to holidays',
     'Discount past 1 day', 'Discount past 2 days', 'Discount past 3 days', 
     'Discount past 4 days', 'Discount past 5 days', 'Discount past 6 days', 'Discount past 7 days',
     'Max discount (past 15 days)', 'Max discount (past 30 days)', 
     'Average discount (past 15 days)', 'Average discount (past 30 days)'],
    rotation = 30,
    fontsize = 20,
    horizontalalignment = 'right'
);



brand_dict["Lancome"]['avg_discount'].unique()




data_year = brand_dict["Lancome"][brand_dict["Lancome"]['discount_today'] == 1] 
sns.factorplot('month', data = data_year, kind='count')


# ### Label multi-class
# 


print(brand_dict["Lancome"]["Y_avg_discount_3d"].unique())
brand_dict["Lancome"]["Y_avg_discount_3d"].hist()

brand_dict["Lancome"]['label'] = 0
brand_dict["Lancome"].loc[brand_dict["Lancome"]['Y_avg_discount_7d'] == 10, 'label'] = 0
brand_dict["Lancome"].loc[brand_dict["Lancome"]['Y_avg_discount_7d'] == 20, 'label'] = 1
brand_dict["Lancome"].loc[brand_dict["Lancome"]['Y_avg_discount_7d'] == 30, 'label'] = 1
brand_dict["Lancome"].loc[brand_dict["Lancome"]['Y_avg_discount_7d'] == 40, 'label'] = 2
brand_dict["Lancome"].loc[brand_dict["Lancome"]['Y_avg_discount_7d'] == 50, 'label'] = 2
brand_dict["Lancome"].loc[brand_dict["Lancome"]['Y_avg_discount_7d'] == 60, 'label'] = 3
brand_dict["Lancome"].loc[brand_dict["Lancome"]['Y_avg_discount_7d'] == 70, 'label'] = 3
brand_dict["Lancome"].loc[brand_dict["Lancome"]['Y_avg_discount_7d'] == 80, 'label'] = 3

condition0a = brand_dict["Lancome"]['Y_avg_discount_7d'] >= 0
condition0b = brand_dict["Lancome"]['Y_avg_discount_7d'] <= 15
condition1a = brand_dict["Lancome"]['Y_avg_discount_7d'] >= 16
condition1b = brand_dict["Lancome"]['Y_avg_discount_7d'] <= 30
condition2a = brand_dict["Lancome"]['Y_avg_discount_7d'] >= 31
condition2b = brand_dict["Lancome"]['Y_avg_discount_7d'] <= 95
print("0 -- 15%")
print(brand_dict["Lancome"].loc[condition0a & condition0b, 'Y_avg_discount_7d'].mean() )
print(brand_dict["Lancome"].loc[condition0a & condition0b, 'Y_avg_discount_7d'].std() )
print('\n')
print("16 -- 30%")
print(brand_dict["Lancome"].loc[condition1a & condition1b, 'Y_avg_discount_7d'].mean() )
print(brand_dict["Lancome"].loc[condition1a & condition1b, 'Y_avg_discount_7d'].std() )
print('\n')
print("31 -- 95%")
print(brand_dict["Lancome"].loc[condition2a & condition2b, 'Y_avg_discount_7d'].mean() )
print(brand_dict["Lancome"].loc[condition2a & condition2b, 'Y_avg_discount_7d'].std() )


y = brand_dict["Lancome"]['label']



X_lastday, y_lastday = X.iloc[-1:, ], y.iloc[-1:, ]
X,y = X.iloc[0:-1, ], y.iloc[0:-1, ]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(X_lastday)
print("\n")
print(y_lastday)


# ### Modeling

# First, try Logistic Regression as Baseline Model


logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)
print(score)




y_hat_train = logisticRegr.predict(X_train)
y_hat_test = logisticRegr.predict(X_test)
cm_train = confusion_matrix(y_train, y_hat_train)
cm_test = confusion_matrix(y_test, y_hat_test)

print(classification_report(y_train, y_hat_train))
print(classification_report(y_test, y_hat_test))
print("Confusion Matrix:")
print(cm_train)
print(cm_test)
plt.figure(1)
y_pred_prob = logisticRegr.predict_proba(X_test)[:, 1]
#fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_prob)
#plt.plot(fpr_grd,tpr_grd)
#auc(fpr_grd,tpr_grd)

#
## Next try XGBoost model
#
#
#xgb = XGBClassifier(random_state=2)
#xgb.fit(X_train, np.array(y_train))
#xgb.score(X_test,y_test)
#
#
#plot_importance(xgb, importance_type='gain', title='Feature importance', height=0.5)
#plt.rcParams['figure.figsize'] = [10, 10]
#plt.show()
#
#y_hat_train = xgb.predict(X_train)
#y_hat_test = xgb.predict(X_test)
#cm_train = confusion_matrix(y_train, y_hat_train)
#cm_test = confusion_matrix(y_test, y_hat_test)
#
#print(classification_report(y_train, y_hat_train))
#print(classification_report(y_test, y_hat_test))
#print("Confusion Matrix:")
#print(cm_train)
#print(cm_test)
#
#plt.figure(1)
#y_pred_prob = xgb.predict_proba(X_test)[:, 1]
##fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_prob)
##plt.plot(fpr_grd,tpr_grd)
##auc(fpr_grd,tpr_grd)
#
#
#
#
#estimator = XGBClassifier(
#    objective= 'multi:softmax',
#    nthread=4,
#    seed=42,
#    num_class=3,
#)
#
#
#
#parameters = {
#   'max_depth': range (2, 10, 1),
#   'n_estimators': range(60, 220, 40),
#   'learning_rate': [0.1, 0.01, 0.05],
#}
#grid_search = GridSearchCV(
#   estimator=estimator,
#   param_grid=parameters,
#   scoring = 'f1_macro', # or 'f1_micro', ‘f1_weighted’
#   n_jobs = 10,
#   cv = 10,
#   verbose=True
#)
#grid_search.fit(X_test, y_test)
#
#
#
#
#grid_search.best_estimator_
#
#
#
#xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0,
#              learning_rate=0.1, max_delta_step=0, max_depth=9,
#              min_child_weight=1, missing=None, n_estimators=180, n_jobs=1,
#              nthread=4, num_class=3, objective='multi:softprob',
#              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#              seed=42, silent=None, subsample=1, verbosity=1)
#xgb.fit(X_train, np.array(y_train))
#xgb.score(X_test, y_test)
#
#
#
#
#y_hat_train = xgb.predict(X_train)
#y_hat_test = xgb.predict(X_test)
#cm_train = confusion_matrix(y_train, y_hat_train)
#cm_test = confusion_matrix(y_test, y_hat_test)
#
#print(classification_report(y_test, y_hat_test))
#print("Confusion Matrix:")
#print(cm_train)
#print(cm_test)
#
#plt.figure(1)
#y_pred_prob = xgb.predict_proba(X_test) #[:, 1]
#print(y_pred_prob[0:5])
#
#
#
#
#plot_importance(xgb, importance_type='gain', max_num_features=5, title='Feature importance', height=0.5)
#plt.rcParams['figure.figsize'] = [10, 10]
#plt.show()
#
#
#
#
#y_hat_lastday = xgb.predict(X_lastday)
#y_hat_lastday

