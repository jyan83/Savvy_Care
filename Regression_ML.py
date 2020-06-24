# spot check nonlinear algorithms
from numpy import load
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score

from sklearn.metrics import log_loss, roc_auc_score, f1_score, confusion_matrix, classification_report, mean_squared_error

from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
import pickle


# load supervised datasets
# Import cleaned dataset
PIK = "Regression.dat"

file = open("Regression.dat",'rb')
object_file = pickle.load(file)
file.close()
[X_train, X_test, y_train, y_test] = object_file
X = np.concatenate([X_train, X_test])
X[X==0] = 'nan'
Y = np.concatenate([y_train, y_test])

time_split = TimeSeriesSplit(n_splits=10)

# prepare a list of ml models
def get_models(models=dict()):
	# non-linear models
	models['knn'] = KNeighborsRegressor(n_neighbors=7)
	models['cart'] = DecisionTreeRegressor()
	models['extra'] = ExtraTreeRegressor()
	models['svmr'] = SVR()
	# # ensemble models
	n_trees = 100
	models['ada'] = AdaBoostRegressor(n_estimators=n_trees)
	models['bag'] = BaggingRegressor(n_estimators=n_trees)
	models['rf'] = RandomForestRegressor(n_estimators=n_trees)
	models['et'] = ExtraTreesRegressor(n_estimators=n_trees)
	models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)
	print('Defined %d models' % len(models))
	return models

n_trees = 100

rgs = [KNeighborsRegressor(),
       DecisionTreeRegressor(),
       AdaBoostRegressor(),
       BaggingRegressor(),
       RandomForestRegressor(),
       MLPRegressor(),
       GradientBoostingRegressor(),
       ExtraTreesRegressor()]

rgs_names = ['knn',
             'DT',
             'AdaBoost',
             'Bagging',
             'RF',
             'MLP',
             'GBM',
             'ExtraTrees']

fig = plt.figure()

for i, item in enumerate(rgs):
    print(rgs_names[i])
    rgs = item
    cv_scores = cross_val_score(rgs, X_train, y_train, cv=time_split, scoring='neg_mean_squared_error')
    rgs.fit(X_train, y_train)

    # score training set
    Y_pred = rgs.predict(X_train)
    train_acc = np.sqrt(mean_squared_error(y_train, Y_pred))
    print ('training accuracy: {}'.format(train_acc))
    
    # score test set
    Y_pred = rgs.predict(X_test)
    test_acc = np.sqrt(mean_squared_error(y_test, Y_pred))
    print ('validation accuracy: {}'.format(test_acc))
    
    plt.scatter(np.arange(len(X)), Y, c='k', label='_nolegend_', zorder=1)

    plt.scatter(np.arange(len(X_test))+len(X_train), Y_pred, s=10,
             label=rgs_names[i]+'(fit: %.3f, predict: %.3f)' % (train_acc, test_acc))
    plt.ylim([0,1])
plt.legend(loc='best')
plt.show()

#%% Save data
PIK = "Frontend/datasets/lancome_Regression_result.dat"

data = [Y_pred]
with open(PIK, "wb") as f:
    pickle.dump(data, f)


