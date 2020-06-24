import numpy as np
import pandas as pd
from pandas import DataFrame

import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.base import clone

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
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

from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from tscv import GapWalkForward
from scipy.stats import randint as sp_randint

import pickle

# Import cleaned dataset
df = pd.read_csv("lancome_clean.csv")
#
## Add missing date to dataframe
#idx = pd.date_range(df['Posted_date'].min(), df['Posted_date'].max())
#df = df.set_index('Posted_date').reindex(idx).fillna(0.0).rename_axis('Posted_date').reset_index()    

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
#    window_1year = shifted_1month.rolling(window=365)
    
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
    
#    df.Posted_date,
    dataframe = pd.concat([
                           shifted_1day, shifted_7day, shifted_14day, shifted_1month, shifted_1year,
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
def prepare_targets(df):
	le = LabelEncoder()
	le.fit(df)
	df = le.transform(df)
	return df

X = add_features(df)
X = X.replace(np.nan, 0)

y = df['Discount_off']
X = prepare_inputs(X)
X['label'] = y
X_train, X_test, y_train, y_test = X[6:-365], X[-365:-1], y[6:-365], y[-365:-1]


rgs = [
#       KNeighborsRegressor(),
       DecisionTreeRegressor(),
       AdaBoostRegressor(),
       BaggingRegressor(),
       RandomForestRegressor(),
#       MLPRegressor(),
       GradientBoostingRegressor(),
       ExtraTreesRegressor()]

rgs_names = [
#        'knn',
             'DT',
             'AdaBoost',
             'Bagging',
             'RF',
#             'MLP',
             'GBM',
             'ExtraTrees']


#% split dataset
scores = []
cv = GapWalkForward(n_splits=20, gap_size=0, test_size=28*3)  
all_scores = np.zeros((len(rgs_names), cv.n_splits))
  
for i in range(len(rgs)):
    clf = clone(rgs[i])
#    for train, test in cv.split(X):
#        y_train = train['label']
#        X_train = train.drop['label']
#        y_test = test['label']
#        X_test = test.drop['label']    

    print(rgs_names[i])
#    rgs = item
    cv_scores = cross_val_score(clf, X_train, y_train, cv = cv, scoring='neg_mean_squared_error')
    all_scores[i] = cv_scores
       
fig, ax = plt.subplots()
sns.boxplot(data=pd.DataFrame(all_scores.T, columns=rgs_names), ax=ax)
ax.set_ylabel('RMSE')

#lgbm = LGBMRegressor()
#
#lgbmPd = {" max_depth": [-1,2]
#         }
#
#model = RandomizedSearchCV(
#    estimator = lgbm,
#    param_distributions = lgbmPd,
#    n_iter = 10,
#    n_jobs = -1,
#    iid = True,
#    cv = cv,
#    verbose=5,
#    pre_dispatch='2*n_jobs',
#    random_state = None,
#    return_train_score = True)

xt = ExtraTreesRegressor()

xtPd = {"n_estimators": sp_randint(10, 110),
        "criterion": ["mse", "mae"],
              "bootstrap": [True, False]}

model = RandomizedSearchCV(
    estimator = xt,
    param_distributions = xtPd,
    n_iter = 10,
    n_jobs = -1,
    iid = True,
    cv = cv,
    verbose=5,
    pre_dispatch='2*n_jobs',
    random_state = None,
    return_train_score = True)

grid_result = model.fit(X_train,y_train)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev in zip(means, stds):
    print("%f (%f):" % (mean, stdev))
lgbm_best_parameters = model.best_params_
lgbm_best_accuracy = model.best_score_
Y_pred = model.predict(X_test)

plt.figure()
plt.plot(np.arange(len(X)), y, c='gray', label = 'ground_truth', linewidth=1)
plt.plot(np.arange(len(X)-len(Y_pred),len(X)),Y_pred, c='r', label = 'prediction')
plt.vlines(len(X)-365,0,1.05, color='k',linestyles='solid', linewidth=2, linestyle = '--')
plt.ylim([0.0, 1.05])
plt.xlabel('Observed days')
plt.ylabel('Discount % OFF')
plt.title('ExtraTrees Regressor performance')
plt.legend(loc='upper left')
plt.show()

rmse = abs((Y_pred-y_test.values).mean())

#%% Save data for regression
PIK = "Frontend/datasets/Regression.dat"

data = [X_train, X_test, y_train, y_test, Y_pred, model]
with open(PIK, "wb") as f:
    pickle.dump(data, f)
with open(PIK, "rb") as f:
    print(pickle.load(f))