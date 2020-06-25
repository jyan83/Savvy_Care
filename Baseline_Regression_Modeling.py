import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score

from sklearn import svm
from sklearn.metrics import log_loss, roc_auc_score, f1_score, confusion_matrix, classification_report, mean_squared_error
from sklearn import linear_model
from numpy import load

import pywt


# Import cleaned dataset
df = pd.read_csv("lancome_clean.csv")

df['Posted_date'] = pd.to_datetime(df['Posted_date'])  
df['date_delta'] = (df['Posted_date'] - df['Posted_date'].min())  / np.timedelta64(1,'D')+1

# Evaluation metric
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = np.sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
    
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')
    
# Get training, testing datasets
df_reg = df.drop("GWP", axis=1)
X = df_reg.drop("Discount_off", axis=1)
X = df_reg[["date_delta"]]
Y = df_reg['Discount_off']
# denoise discount using wavelet transform
#Y = pd.Series(denoise_signal(Y))
#X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, train_size = 1-365/len(df), shuffle = False)
X_train = load('Data/regression_train_X.npy', allow_pickle=True)
X_test = load('Data/regression_test_X.npy', allow_pickle=True)
Y_train = load('Data/regression_train_y.npy', allow_pickle=True)
Y_test = load('Data/regression_test_y.npy', allow_pickle=True)
time_split = TimeSeriesSplit(n_splits=10)


## train SVM 
regressors = [svm.SVR(),
#        linear_model.SGDRegressor(),
        linear_model.BayesianRidge(),
        linear_model.LassoLars(),
        linear_model.ARDRegression(),
        linear_model.PassiveAggressiveRegressor(),
        linear_model.TheilSenRegressor(),
        linear_model.LinearRegression()]

name = ['svm.SVR',
#        'SGDRegressor',
        'BayesianRidge',
        'LassoLars',
        'ARDRegression',
        'PassiveAggressiveRegressor',
        'TheilSenRegressor',
        'LinearRegression']

fig = plt.figure()

for i, item in enumerate(regressors):
    print(name[i])
    clf = item
    cv_scores = cross_val_score(clf, X_train, Y_train, cv=time_split, scoring='neg_mean_squared_error')
    clf.fit(X_train, Y_train)

    # score training set
    Y_pred = clf.predict(X_train)
    train_acc = round(mean_squared_error(Y_train, Y_pred) * 100, 2)
    
    print ('training accuracy: {}'.format(train_acc))
    
    # score test set
    Y_pred = clf.predict(X_test)
    test_acc = round(mean_squared_error(Y_test, Y_pred) * 100, 2)
    print ('validation accuracy: {}'.format(test_acc))
    
#    plt.scatter(X.index, df_reg['Discount_off'], c='k', label='_nolegend_', zorder=1)
    plt.scatter(np.arange(len(X)), df_reg['Discount_off'], c='k', label='_nolegend_', zorder=1)

#    plt.scatter(X_train.index, Y_train, c='g', s=50, label='_nolegend_',
#                zorder=2, edgecolors=(0, 0, 0))
#    plt.scatter(X_test.index, Y_pred, s=10,
    plt.scatter(np.arange(len(X_test))+len(X_train), Y_pred, s=10,
             label=name[i]+'(fit: %.3f, predict: %.3f)' % (train_acc, test_acc))
    plt.ylim([0,1])
plt.legend(loc='best')
plt.show()

# Regression does not work, there is no siginificant patterns/non-stationary in your discount :(

