import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, learning_curve, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from xgboost import plot_importance
from sklearn import svm
from sklearn.metrics import mean_squared_error  
from sklearn import linear_model


# Import cleaned dataset

df = pd.read_csv("lancome_clean.csv")


#def lagged_features(df_long, lag_features, window=7, widths=[4], lag_prefix='lag', lag_prefix_sep='_'):
#    """
#    Function calculates lagged features (only for columns mentioned in lag_features)
#    based on time_feature column. The highest value of time_feature is retained as a row
#    and the lower values of time_feature are added as lagged_features
#    :param df_long: Data frame (longitudinal) to create lagged features on
#    :param lag_features: A list of columns to be lagged
#    :param window: How many lags to perform (0 means no lagged feature will be produced)
#    :param widths: How many days to roll back and calculate the statistical features
#    :param lag_prefix: Prefix to name lagged columns.
#    :param lag_prefix_sep: Separator to use while naming lagged columns
#    :return: Data Frame with lagged features appended as columns
#    """
#    if not isinstance(lag_features, list):
#        # So that while sub-setting DataFrame, we don't get a Series
#        lag_features = [lag_features]
#
#    if window <= 0:
#        return df_long
#
#    df_working = df_long[lag_features].copy()
#    df_result = df_long.copy()
#    for i in range(1, window+1):
#        df_temp = df_working.shift(i)
#        df_temp.columns = [lag_prefix + lag_prefix_sep + str(i) + lag_prefix_sep + x
#                           for x in df_temp.columns]
#        df_result = pd.concat([df_result.reset_index(drop=True),
#                               df_temp.reset_index(drop=True)],
#                               axis=1)
#    for width in widths:
#        
#        window2 = df_working.rolling(window=width)
#        window3 = df_long[['Discount_off']].copy().shift(width - 1).rolling(window=width)
#
#        dataframe = window2.sum()
#        dataframe.columns = [lag_prefix+'_past_'+str(width)+'_sum']
#        dataframe2 = window3.max()
#        dataframe2.columns = [lag_prefix+'_past_'+str(width)+'_max_amount']
#        dataframe3 = window3.mean()
#        dataframe3.columns = [lag_prefix+'_past_'+str(width)+'_mean_amount']
#        df_result = pd.concat([df_result.reset_index(drop=True),
#                               dataframe.reset_index(drop=True)],
#                               axis=1)
#        df_result = pd.concat([df_result.reset_index(drop=True),
#                               dataframe2.reset_index(drop=True)],
#                               axis=1)
#        df_result = pd.concat([df_result.reset_index(drop=True),
#                               dataframe3.reset_index(drop=True)],
#                               axis=1)
#
#    return df_result
#
#df['Discount_YesOrNo'] = 1
#df.loc[df['Discount_off'] == 0, 'Discount_YesOrNo'] = 0
#df = lagged_features(df, ['Discount_YesOrNo'], window=7, widths=[15,30])


def phase2clean(df):
    #data type dictionary
    dummies_weekday = pd.get_dummies(df['weekday'], prefix= 'weekday')
    
    dummies_Event = pd.get_dummies(df['Event'], prefix= 'Event')
    
    dummies_month = pd.get_dummies(df['month'], prefix= 'month')
    
#    dummies_day = pd.get_dummies(df['day'], prefix= 'day')
    df['Posted_date'] = pd.to_datetime(df['Posted_date'])  
    df['date_delta'] = (df['Posted_date'] - df['Posted_date'].min())  / np.timedelta64(1,'D')
    
#    df = pd.concat([df, dummies_weekday, dummies_Event,dummies_month,dummies_day], axis=1)
    df = pd.concat([df, dummies_weekday, dummies_Event,dummies_month], axis=1)

    df.drop(['Posted_date', 'year', 'month', 'day', 'weekday', 'Event', 'Unnamed: 0', 'Store', 'Discount', 'Comments_count',
             'Bookmarks_count', 'Shares_count'], axis=1, inplace=True)
    
    return df

df_dummy = phase2clean(df)
df_dummy = df_dummy.fillna(0)

#%% Baseline model for regression:
# Get training, testing datasets
df_reg = df_dummy.drop("GWP", axis=1)
X = df_reg.drop("Discount_off", axis=1)
#X = df_reg["date_delta"]
Y = df_reg['Discount_off']
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, train_size = 0.8, shuffle = False)

## train SVM 
classifiers = [svm.SVR(),
#        linear_model.SGDRegressor(),
#        linear_model.BayesianRidge(),
        linear_model.LassoLars(),
#        linear_model.ARDRegression(),
#        linear_model.PassiveAggressiveRegressor(),
#        linear_model.TheilSenRegressor(),
        linear_model.LinearRegression()]

name = ['svm.SVR',
#        'SGDRegressor',
#        'BayesianRidge()',
        'LassoLars()',
#        'ARDRegression()',
#        'PassiveAggressiveRegressor()',
#        'TheilSenRegressor()',
        'LinearRegression']
for i,item in enumerate(classifiers):
    print(item)
    clf = item
    clf.fit(X_train, Y_train)

#clf = GaussianNB()
#clf.fit(X_train, Y_train)

#Y_pred = clf.predict(X_test)
#scores = mean_squared_error(Y_test, Y_pred)

    # score training set
    Y_pred = clf.predict(X_train)
    train_acc = round(mean_squared_error(Y_train, Y_pred) * 100, 2)
    print ('training accuracy: {}'.format(train_acc))
    
    # score test set
    Y_pred = clf.predict(X_test)
    test_acc = round(mean_squared_error(Y_test, Y_pred) * 100, 2)
    print ('validation accuracy: {}'.format(test_acc))
    
    plt.scatter(X.index, Y, c='k', label='_nolegend_', zorder=1,
                edgecolors=(0, 0, 0))
    plt.scatter(X_train.index, Y_train, c='g', s=50, label='_nolegend_',
                zorder=2, edgecolors=(0, 0, 0))
    plt.scatter(X_test.index, Y_pred, s=10,
             label=name[i]+'(fit: %.3f, predict: %.3f)' % (train_acc, test_acc))
plt.legend(loc='best')
plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"Number of training samples")
        plt.ylabel(u"Accuracy")
        plt.gca().invert_yaxis()
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"training accuracy")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"validation accuracy")
    
        plt.legend(loc="best")
        
        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    return train_scores_mean.mean(),train_scores_std.mean(),test_scores_mean.mean(),test_scores_std.mean()