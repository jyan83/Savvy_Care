import numpy as np
import pandas as pd
from numpy import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error  
from sklearn.metrics import log_loss, roc_auc_score, f1_score, confusion_matrix, classification_report, mean_squared_error
import xgboost as xgb
from sklearn.pipeline import Pipeline

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold, StratifiedKFold, cross_val_score 
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import clone
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from xgboost import plot_importance
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import pickle

# Import cleaned dataset
#df = pd.read_csv("lancome_clean.csv")


#%% Baseline model for classfication
# Import cleaned dataset
PIK = "Classification.dat"

file = open("Classification.dat",'rb')
object_file = pickle.load(file)
file.close()
[X_train, X_test, y_train, y_test] = object_file

# load supervised datasets
#X_train = load('Data/classification_train_X.npy', allow_pickle=True)
#X_test = load('Data/classification_test_X.npy', allow_pickle=True)
#y_train = load('Data/classification_train_y.npy', allow_pickle=True)
#y_test = load('Data/classification_test_y.npy', allow_pickle=True)
RNG = 42
#X, Y = np.load('Data/classification_X.npy', allow_pickle=True), np.load('Data/classification_y.npy', allow_pickle=True)
## split into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 1-365/len(df),
#                                                    stratify=None, random_state=RNG)
#     Fit logistic regression model to train data and test on test data
#clf = LogisticRegression(C = 2, penalty='l2', max_iter=1000)  # The value of C should be determined by nested validation
##X_train = np.arange(1,len(X_train)+1)
#clf.fit(X_train, y_train)
#
#
## Initiate a FS-Logit pipeline
fs = SelectKBest(f_classif, k=10)
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
# Compute FPR, TPR, Precision by iterating classification thresholds
fpr_baseline, tpr_baseline, thresholds_baseline = roc_curve(y_test, y_test_probas[:, 1])
precision_baseline, recall_baseline, thresholds = precision_recall_curve(y_test, y_test_probas[:, 1])

# Plot
fig, axes = plt.subplots(1, 2)
axes[0].plot(fpr_baseline, tpr_baseline)
axes[0].set_xlabel('FPR')
axes[0].set_ylabel('TPR')
axes[0].set_title('ROC')

axes[1].plot(recall_baseline, precision_baseline)
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

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)

#In Random Forest:
#class_weight='balanced': uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
#class_weight='balanced_subsample': is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.

clfs = [LogisticRegression(),
               GaussianNB(),
               SVC(kernel='linear', class_weight='balanced'),
               RandomForestClassifier(n_estimators=n_estimators, random_state=RNG),
               RandomForestClassifier(n_estimators=n_estimators, random_state=RNG, class_weight='balanced'),
               RandomForestClassifier(n_estimators=n_estimators, random_state=RNG, class_weight='balanced_subsample'),
               GradientBoostingClassifier(),
               xgb.XGBClassifier(n_estimators=n_estimators, subsample=.9, seed=RNG)]


clf_names = ['Logit',
             'NB',
             'SVM',
             'RF',
             'RF_2',
             'RF_3',
             'GBM',
             'XGB']


cv = StratifiedKFold(n_splits=10,random_state=2,shuffle=True)
#clf_names = ['GBM', 'RF', 'RF_balanced', 'RF_balanced2']
all_scores = np.zeros((len(clf_names), cv.n_splits))
for i in range(len(clfs)):
    clf = clone(clfs[i])
    scores = cross_val_score(clf, X_test, y_test,  cv=cv, scoring='roc_auc')
    all_scores[i] = scores
    
fig, ax = plt.subplots()
sns.boxplot(data=pd.DataFrame(all_scores.T, columns=clf_names), ax=ax)
ax.set_ylabel('AUROC')

# Select the optimal aglorithm to classify
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svm = round(svc.score(X_train, y_train) * 100, 2)
print("Support vector machines accuracy: ", acc_svm)

SVM_param = [{
    'kernel': ['rbf', 'sigmoid'], 
    'gamma': [ 0.01, 0.03, 0.05, 0.1],
    'C': [1, 2, 3, 5, 10, 20, 30, 50, 100]}]
gs_SVM = GridSearchCV(svc,param_grid = SVM_param, cv=cv, scoring="roc_auc", n_jobs= 2, verbose = 1)
svm_result = gs_SVM.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (svm_result.best_score_, svm_result.best_params_))
means = svm_result.cv_results_['mean_test_score']

stds = svm_result.cv_results_['std_test_score']
params = svm_result.cv_results_['params']
svm_best_parameters = gs_SVM.best_params_
svm_best_accuracy = gs_SVM.best_score_
print("Best: %r" % svm_best_parameters)

svc = SVC(kernel=svm_best_parameters['kernel'],
          gamma=svm_best_parameters['gamma'],
          C=svm_best_parameters['C'])
svc.fit(X_train, y_train)
y_score = svc.fit(X_train, y_train).decision_function(X_test)

y_pred = svc.predict(X_test)
    
#Print Precision, Recall and F1 scores
print(classification_report(y_test, y_pred))

clf.fit(X_train, y_train)
plot_importance(clf, importance_type='gain')
plt.show()

# Compute FPR, TPR, Precision by iterating classification thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

# Plot
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='SVM ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_baseline, tpr_baseline, color='darkblue',
         lw=lw, label='Baseline ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_test_probas[:, 1]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


gb =xgb.XGBClassifier(random_state=0)
gb_param = {'n_estimators': stats.randint(150, 1000),
              'learning_rate': stats.uniform(0.01, 0.6),
              'subsample': stats.uniform(0.3, 0.9),
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'colsample_bytree': stats.uniform(0.5, 0.9),
              'min_child_weight': [1, 2, 3, 4]
             }
gs_GBC = RandomizedSearchCV(gb, 
                         param_distributions = gb_param,
                         cv = cv,  
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)
grid_result=gs_GBC.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
gb_best_parameters = gs_GBC.best_params_
gb_best_accuracy = gs_GBC.best_score_
print("Best: %r" % gb_best_parameters)

gb = GradientBoostingClassifier(random_state=0,n_estimators=gb_best_parameters['n_estimators'],
                   learning_rate=gb_best_parameters['learning_rate'],
                   max_depth=gb_best_parameters['max_depth'])
gb.fit(X_train, y_train)
y_score = gb.fit(X_train, y_train).decision_function(X_test)

y_pred = gb.predict(X_test)
    
#Print Precision, Recall and F1 scores
print(classification_report(y_test, y_pred))

clf.fit(X_train, y_train)
plot_importance(clf, importance_type='gain')
plt.show()

# Compute FPR, TPR, Precision by iterating classification thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

# Plot
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#%% Save data for classification
PIK = "Frontend/datasets/Classification.dat"
#y_pred['Date'] = Date
data = [X_train, X_test, y_train, y_test, y_pred, gb]
with open(PIK, "wb") as f:
    pickle.dump(data, f)
with open(PIK, "rb") as f:
    print(pickle.load(f))
