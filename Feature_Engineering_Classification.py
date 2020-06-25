import numpy as np
import pandas as pd
from pandas import DataFrame

import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pickle

# Import cleaned dataset
df = pd.read_csv("lancome_clean2.csv")

def add_features(df):
    # Add lag features
    temps = DataFrame(df['Discount_off'].values)
    shifted_1day = temps.shift(1)
    shifted_7day = temps.shift(7)
    shifted_14day = temps.shift(14)
    shifted_1month = temps.shift(30)
    shifted_1year = temps.shift(365)
    
    # Rolling window statistics
    window_1day = shifted_1day.rolling(window=1)
    window_7day = shifted_7day.rolling(window=7)
    window_14day = shifted_14day.rolling(window=14)
    window_1month = shifted_1month.rolling(window=30)
    window_1year = shifted_1month.rolling(window=365)
    
    means_1day = window_1day.mean()
    means_7day = window_7day.mean()
    means_14day = window_14day.mean()
    means_1month = window_1month.mean()
#    means_1year = window_1year.mean()
    
    maxs_1day = window_1day.max()
    maxs_7day = window_7day.max()
    maxs_14day = window_14day.max()
    maxs_1month = window_1month.max()
#    maxs_1year = window_1year.max()

    
    sum_1day = window_1day.sum()
    sum_7day = window_7day.sum()
    sum_14day = window_14day.sum()
    sum_1month = window_1month.sum()
#    means_1year = window_1year.mean()
    
    
    dataframe = pd.concat([shifted_1day, shifted_7day, shifted_14day, shifted_1month, shifted_1year,
                           means_1day, means_7day, means_14day, means_1month,
                           maxs_1day, maxs_7day, maxs_14day, maxs_1month,
                           sum_1day, sum_7day, sum_14day, sum_1month,
                           df['year'], df['month'], df['day'], df['Weekend_FLG'],
                           df['Event_1day'], df['Event_7days'], df['Event_14days'], df['Event_1month']], axis=1)
    dataframe.columns = ['t-1','t-7','t-14','t-30','t-365',
                         'mean-1','mean-7','mean-14','mean-30',
                         'max-1','max-7','max-14','max-30',
                         'sum-1','sum-7','sum-14','sum-30',
                         'year','month','day','Weekend_FLG',
                         'Event_1day','Event_7day','Event_14day','Event_month']
    
    return dataframe

# Prepare Input data
def prepare_inputs(df):
    dummies_Event_1day = pd.get_dummies(df['Event_1day'], prefix= 'Event_1day')
    dummies_Event_7day = pd.get_dummies(df['Event_7day'], prefix= 'Event_7day')
#    dummies_Event_14day = pd.get_dummies(df['Event_14day'], prefix= 'Event_14day')
#    dummies_Event_1month = pd.get_dummies(df['Event_month'], prefix= 'Event_1month')
    dummies_Weekend_FLG = pd.get_dummies(df['Weekend_FLG'], prefix= 'Weekend_FLG')
    
    df = pd.concat([df, dummies_Event_1day, dummies_Event_7day, dummies_Weekend_FLG], axis=1)
    df.drop(['Event_1day', 'Event_7day', 'Event_14day', 'Event_month', 'Weekend_FLG'], axis=1, inplace=True)
    return df

# prepare target
def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc


X = add_features(df)

X = X.replace(np.nan, 0)
Y = df['Discount_off']
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, random_state=42)
X_train = prepare_inputs(X_train)
X_test = prepare_inputs(X_test)

spike_cols = [col for col in X_train.columns if col in X_test.columns]
X_train = X_train[spike_cols]

spike_cols = [col for col in X_test.columns if col in X_train.columns]
X_test = X_test[spike_cols]

# prepare input data
#X_train_enc, X_test_enc = prepare_inputs(X_train, X_test.iloc)





#%% Feature Selection
# 1. Correlation feature selection
# configure to select all features
def correlation_feture(X_train, y_train):
    fs = SelectKBest(score_func=f_regression, k=10)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    # what are scores for the features
    for i in range(len(fs.scores_)):
    	print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    plt.figure()
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.title('Correlation Feature Selection')
    plt.show()
    return X_train_fs, X_test_fs, fs

# 2. Mutual information feature selection
# configure to select all features
def MI_feture(X_train, y_train):
    fs = SelectKBest(score_func=mutual_info_regression, k=10)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    # what are scores for the features
    for i in range(len(fs.scores_)):
    	print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    plt.figure()
    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.title('Mutual Information Feature Selection')
    plt.show()
    return X_train_fs, X_test_fs, fs

X_train_cf, X_test_cf, _ = correlation_feture(X_train, y_train)
X_train_mi, X_test_mi, _ = MI_feture(X_train, y_train)


#%% Modeling With Selected Features
# fit the model
model = LinearRegression()
model.fit(X_train, y_train)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
rmse = np.sqrt(mean_squared_error(y_test, yhat))
print('Linear regression with full features, RMSE: %.3f' % rmse)

model.fit(X_train_cf, y_train)
# evaluate the model
yhat = model.predict(X_test_cf)
# evaluate predictions
rmse = np.sqrt(mean_squared_error(y_test, yhat))
print('Linear regression with correlation features, RMSE: %.3f' % rmse)

model.fit(X_train_mi, y_train)
# evaluate the model
yhat = model.predict(X_test_mi)
# evaluate predictions
rmse = np.sqrt(mean_squared_error(y_test, yhat))
print('Linear regression with mutual information features, RMSE: %.3f' % rmse)


##%% Tune the Number of Selected Features
## define the evaluation method
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#
## define the pipeline to evaluate
#model = LinearRegression()
#fs = SelectKBest(score_func=mutual_info_regression)
#pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
#
## define the grid
#grid = dict()
#grid['sel__k'] = [i for i in range(X.shape[1]-20, X.shape[1]+1)]
#
## define the grid search
#search = GridSearchCV(pipeline, grid, scoring='average_precision', n_jobs=-1, cv=cv)
## perform the search
#results = search.fit(X_train, y_train)
#
## summarize best
#print('Best MAE: %.3f' % results.best_score_)
#print('Best Config: %s' % results.best_params_)
## summarize all
#means = results.cv_results_['mean_test_score']
#params = results.cv_results_['params']
#for mean, param in zip(means, params):
#    print(">%.3f with: %r" % (mean, param))
#    
## define number of features to evaluate
#num_features = [i for i in range(X.shape[1]-19, X.shape[1]+1)]
## enumerate each number of features
#results = list()
#for k in num_features:
#	# create pipeline
#	model = LinearRegression()
#	fs = SelectKBest(score_func=mutual_info_regression, k=k)
#	pipeline = Pipeline(steps=[('sel',fs), ('lr', model)])
#	# evaluate the model
#	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
#	scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#	results.append(scores)
#	# summarize the results
#	print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))
## plot model performance for comparison
#pyplot.boxplot(results, labels=num_features, showmeans=True)
#pyplot.show()

#%% Save data for regression
np.save('Data/regression_train_X.npy', X_train_cf)
np.save('Data/regression_train_y.npy', y_train)
np.save('Data/regression_test_X.npy', X_test_cf)
np.save('Data/regression_test_y.npy', y_test)

PIK = "Regression.dat"

data = [X_train, X_test, y_train, y_test]
with open(PIK, "wb") as f:
    pickle.dump(data, f)
with open(PIK, "rb") as f:
    print(pickle.load(f))

#%% Save data for classification
Y = df['GWP']
# split into train and test sets
_, _, y_train, y_test = train_test_split(X, Y, train_size = 0.7, random_state=42)
PIK = "Classification.dat"

data = [X_train, X_test, y_train, y_test]
with open(PIK, "wb") as f:
    pickle.dump(data, f)
with open(PIK, "rb") as f:
    print(pickle.load(f))
    
