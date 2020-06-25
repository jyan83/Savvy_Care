# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 05:13:01 2020

@author: Jin Yan
"""

import numpy as np
import pandas as pd
from numpy import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error  
from sklearn.metrics import log_loss, roc_auc_score, f1_score, confusion_matrix, classification_report, mean_squared_error
import xgboost as xgb
from sklearn.pipeline import Pipeline

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score 
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

# Import cleaned dataset
df = pd.read_csv("lancome_clean.csv")


#%% Baseline model for classfication
# load supervised datasets
X_train = load('Data/regression_train_X.npy', allow_pickle=True)
X_test = load('Data/regression_test_X.npy', allow_pickle=True)
y_train = load('Data/regression_train_y.npy', allow_pickle=True)
y_test = load('Data/regression_test_y.npy', allow_pickle=True)

y_train_c, y_test_c = y_train, y_test

for i in y_train:
    if i == 0:
        y_test_c[i] = "0"
    elif i == 0.15:
        y_test_c[i] = "1"
    elif i == 0.2:
        y_test_c[i] = "2"
    elif i == 0.25:
        y_test_c[i] = "2"
RNG = 42

#     Fit logistic regression model to train data and test on test data
clf = LogisticRegression(C = 2, penalty='l2', max_iter=1000)  # The value of C should be determined by nested validation
#X_train = np.arange(1,len(X_train)+1)
clf.fit(X_train, y_train)


## Initiate a FS-Logit pipeline
fs = SelectKBest(f_classif, k=2)
logit = LogisticRegression()

pipeline = Pipeline([
        ('fs', fs),
        ('logit', logit)
    ])
    
# Train classifier
pipeline.fit(X_train, y_train)
# Get prediction on test set
y_test_preds = pipeline.predict(X_test)
y_test_probas = pipeline.predict_proba(X_test)
print(y_test_preds.shape, y_test_probas.shape)


# Evaluate predictions
print('Accuracy: %.5f' % accuracy_score(y_test, y_test_preds))
print('F1 score: %.5f' % f1_score(y_test, y_test_preds))
print('AUROC: %.5f' % roc_auc_score(y_test, y_test_probas[:, 1]))
print('AUPRC: %.5f' % average_precision_score(y_test, y_test_probas[:, 1]))


# To plot ROC and PRC
from sklearn.metrics import roc_curve, precision_recall_curve

# Compute FPR, TPR, Precision by iterating classification thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_test_probas[:, 1])
precision, recall, thresholds = precision_recall_curve(y_test, y_test_probas[:, 1])

# Plot
fig, axes = plt.subplots(1, 2)
axes[0].plot(fpr, tpr)
axes[0].set_xlabel('FPR')
axes[0].set_ylabel('TPR')
axes[0].set_title('ROC')

axes[1].plot(recall, precision)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('PRC')

axes[0].set_xlim([-.05, 1.05])
axes[0].set_ylim([-.05, 1.05])
axes[1].set_xlim([-.05, 1.05])
axes[1].set_ylim([-.05, 1.05])

fig.tight_layout()
plt.show()


#%% More classifiers compare the performance
n_estimators = 20

#In Random Forest:
#class_weight='balanced': uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
#class_weight='balanced_subsample': is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.

clfs = [LogisticRegression(),
               GaussianNB(),
               SVC(kernel='linear', class_weight='balanced'),
               RandomForestClassifier(n_estimators=n_estimators, random_state=RNG),
               RandomForestClassifier(n_estimators=n_estimators, random_state=RNG, class_weight='balanced'),
               RandomForestClassifier(n_estimators=n_estimators, random_state=RNG, class_weight='balanced_subsample'),
               xgb.XGBClassifier(n_estimators=n_estimators, subsample=.9, seed=RNG)]


clf_names = ['Logit',
             'NB',
             'SVM',
             'RF',
             'RF_2',
             'RF_3',
             'GBM']


cv = StratifiedKFold(n_splits=30)
#clf_names = ['GBM', 'RF', 'RF_balanced', 'RF_balanced2']
all_scores = np.zeros((len(clf_names), cv.n_splits))
for i in range(len(clfs)):
    clf = clone(clfs[i])
    scores = cross_val_score(clf, X_test, y_test,  cv=cv, scoring='roc_auc')
    all_scores[i] = scores
    
fig, ax = plt.subplots()
sns.boxplot(data=pd.DataFrame(all_scores.T, columns=clf_names), ax=ax)
ax.set_ylabel('AUROC')





