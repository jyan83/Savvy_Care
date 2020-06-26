import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold, StratifiedKFold, cross_val_score 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score 
from sklearn.base import clone
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, classification_report 
from xgboost import plot_importance
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import pickle


#%% Baseline model for classfication
# Import cleaned dataset
PIK = "../Data/Classification.dat"

file = open("../Data/lancome_Classification.dat",'rb')
object_file = pickle.load(file)
file.close()
[X_train, X_test, y_train, y_test] = object_file

RNG = 42

## Initiate a FS-Logit pipeline
fs = SelectKBest(f_classif, k=5)
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
        KNeighborsClassifier(),
        GaussianNB(),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        xgb.XGBClassifier()]

clf_names = ['Logit',
             'KNN',
             'NB',
             'SVC',
             'DF',
             'RF',
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


clf.fit(X_train, y_train)
plot_importance(clf, importance_type='gain')
plt.show()

#%% Select the optimal aglorithm to classify
## Select support vector machine to do the hyperparameter tunning
#svc = SVC()
#svc.fit(X_train, y_train)
#Y_pred = svc.predict(X_test)
#acc_svm = round(svc.score(X_train, y_train) * 100, 2)
#print("Support vector machines accuracy: ", acc_svm)
#
#SVM_param = [{
#    'kernel': ['rbf', 'sigmoid'], 
#    'gamma': [ 0.01, 0.03, 0.05, 0.1],
#    'C': [1, 2, 3, 5, 10, 20, 30, 50, 100]}]
#gs_SVM = GridSearchCV(svc,param_grid = SVM_param, cv=cv, scoring="roc_auc", n_jobs= 2, verbose = 1)
#svm_result = gs_SVM.fit(X_train,y_train)
## summarize results
#print("Best: %f using %s" % (svm_result.best_score_, svm_result.best_params_))
#means = svm_result.cv_results_['mean_test_score']
#
#stds = svm_result.cv_results_['std_test_score']
#params = svm_result.cv_results_['params']
#svm_best_parameters = gs_SVM.best_params_
#svm_best_accuracy = gs_SVM.best_score_
#print("Best: %r" % svm_best_parameters)
#
#svc = SVC(kernel=svm_best_parameters['kernel'],
#          gamma=svm_best_parameters['gamma'],
#          C=svm_best_parameters['C'])
#svc.fit(X_train, y_train)
#y_score = svc.fit(X_train, y_train).decision_function(X_test)
#
#y_pred = svc.predict(X_test)
#    
##Print Precision, Recall and F1 scores
#print(classification_report(y_test, y_pred))
#
## Compute FPR, TPR, Precision by iterating classification thresholds
#fpr, tpr, thresholds = roc_curve(y_test, y_score)
#roc_auc = auc(fpr, tpr)
#precision, recall, thresholds = precision_recall_curve(y_test, y_score)
#
## Plot
#plt.figure()
#lw = 2
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label='SVM ROC curve (area = %0.2f)' % roc_auc)
#plt.plot(fpr_baseline, tpr_baseline, color='darkblue',
#         lw=lw, label='Baseline ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_test_probas[:, 1]))
#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.show()


## Select extreme gradient boosting machine to do the hyperparameter tunning
#gb = xgb.XGBClassifier(objective = 'binary:logistic')
#gb_param = {'n_estimators': stats.randint(150, 1000),
#              'learning_rate': stats.uniform(0.01, 0.6),
#              'subsample': stats.uniform(0.3, 0.9),
#              'max_depth': [3, 4, 5, 6, 7, 8, 9],
#              'colsample_bytree': stats.uniform(0.5, 0.9),
#              'min_child_weight': [1, 2, 3, 4]
#             }
#gs_GBC = RandomizedSearchCV(gb, 
#                         param_distributions = gb_param,
#                         cv = cv,  
#                         scoring = 'roc_auc', 
#                         error_score = 0, 
#                         verbose = 3, 
#                         n_jobs = -1)
#grid_result=gs_GBC.fit(X_train,y_train)
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#gb_best_parameters = gs_GBC.best_params_
#gb_best_accuracy = gs_GBC.best_score_
#print("Best: %r" % gb_best_parameters)
#
#gb = xgb.XGBClassifier(grid_result.best_params_)
#gb.fit(X_train, y_train)
#y_score = gb.fit(X_train, y_train).predict_proba(X_test)
#
#y_pred = gb.predict(X_test)
#    
##Print Precision, Recall and F1 scores
#print(classification_report(y_test, y_pred))
#
## Compute FPR, TPR, Precision by iterating classification thresholds
#fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
#roc_auc = auc(fpr, tpr)
#precision, recall, thresholds = precision_recall_curve(y_test, y_score[:, 1])
#
## Plot
#plt.figure()
#lw = 2
#plt.plot(fpr, tpr, color='darkorange',
#         lw=lw, label='XGBoost ROC curve (area = %0.2f)' % roc_auc)
#plt.plot(fpr_baseline, tpr_baseline, color='darkblue',
#         lw=lw, label='Baseline ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_test_probas[:, 1]))
#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.show()


## Random forest hyperparameter tunning
# Number of trees in random forest
n_estimators = stats.randint(150, 1000)
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, 
                               cv = cv, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random_result=rf_random.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (rf_random_result.best_score_, rf_random_result.best_params_))
means = rf_random_result.cv_results_['mean_test_score']
stds = rf_random_result.cv_results_['std_test_score']
params = rf_random_result.cv_results_['params']
rf_best_parameters = rf_random_result.best_params_
rf_best_accuracy = rf_random_result.best_score_
print("Best: %r" % rf_best_parameters)

rf = RandomForestClassifier(n_estimators=rf_best_parameters['n_estimators'],
                   max_features=rf_best_parameters['max_features'],
                   max_depth=rf_best_parameters['max_depth'],
                   min_samples_split=rf_best_parameters['min_samples_split'],
                   min_samples_leaf=rf_best_parameters['min_samples_leaf'],
                   bootstrap=rf_best_parameters['bootstrap'])
rf.fit(X_train, y_train)
y_score = rf.fit(X_train, y_train).predict_proba(X_test)

y_pred = rf.predict(X_test)
    
#Print Precision, Recall and F1 scores
print(classification_report(y_test, y_pred))

# Compute FPR, TPR, Precision by iterating classification thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
roc_auc = auc(fpr, tpr)
precision, recall, thresholds = precision_recall_curve(y_test, y_score[:, 1])

#%% Save data for classification
PIK = "Frontend/datasets/Classification.dat"
#y_pred['Date'] = Date
data = [X_train, X_test, y_train, y_test, y_pred, gb]
with open(PIK, "wb") as f:
    pickle.dump(data, f)
with open(PIK, "rb") as f:
    print(pickle.load(f))
