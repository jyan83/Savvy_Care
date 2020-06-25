import os, gc
import re, string
import calendar
import holidays
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
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from xgboost import plot_importance
import pickle

# ## Import the cleaned datasets

lancome = pd.read_csv("lancome_clean.csv")


All = [lancome]


# ## Functions for feature engineering:

def str2datetime(df, cols):
    for col in cols:
        try:
            temp= pd.to_datetime(df[col],format='%Y-%m-%d')
        except:
            temp= pd.to_datetime(df[col],format='%m/%d/%Y')
        df.loc[:,col] = temp
    return df

def add_time_features(df, prefix='Posted'):   
    # The string prefix must be 'Posted' or 'End'.
    
    df[prefix+'_year'] = df[prefix+'_date'].dt.year
    df[prefix+'_month'] = df[prefix+'_date'].dt.month
    df[prefix+'_month_year'] = df[prefix+'_date'].dt.to_period('M')
    return df

def add_more_time_features(df, prefix='Posted'):   
    # The string prefix must be 'Posted' or 'End'.
    
    df[prefix+'_dayofweek'] = df[prefix+'_date'].dt.dayofweek
    df[prefix+'_day'] = df[prefix+'_date'].dt.day
    df[prefix+'_dayofyear'] = df[prefix+'_date'].dt.dayofyear
    return df
    
calendar.setfirstweekday(calendar.SATURDAY)
def get_week_of_month(df):
    year = df['Posted_date'].dt.year.values
    month = df['Posted_date'].dt.month.values
    day = df['Posted_date'].dt.day.values
    week_of_month = []
    for i in range(len(year)):
        x = np.array(calendar.monthcalendar(year[i], month[i]))
        week_of_month.append(np.where(x==day[i])[0][0] + 1)
    df['Posted_weekofmonth'] = week_of_month
    return df

def add_holiday_features(df):
    df = df.sort_values(by='Posted_date')
    for name in holidays.US(years=2016).values():
        if 'Observed' in name:
            continue
        else:
            df['days_away' + name] = [0]*df.shape[0]
    
    year_range = range(df['Posted_year'].min(), df['Posted_year'].max()+1)
    for y in year_range:
        for ptr, name in holidays.US(years=y).items():
            if 'Observed' in name:
                continue
            else:
                df.loc[df['Posted_year'] == y,'days_away'+name] = (pd.to_datetime(ptr) - df.loc[df['Posted_year'] == y, 'Posted_date']).dt.days.apply(lambda x: np.exp(-x*x/(20*20)) if x > -20 and x < 20 else 0)
    # The Gaussian distance to the nearest holiday
    df['distance_away_holiday']=df[["days_awayNew Year's Day", 'days_awayMartin Luther King, Jr. Day',
                                    "days_awayWashington's Birthday", 'days_awayMemorial Day',
                                    'days_awayIndependence Day', 'days_awayLabor Day', 'days_awayColumbus Day',
                                    'days_awayVeterans Day', 'days_awayThanksgiving', 'days_awayChristmas Day']].max(axis=1)
    
    return df


def lagged_features(df_long, lag_features, window=7, widths=[4], lag_prefix='lag', lag_prefix_sep='_'):
    """
    Function calculates lagged features (only for columns mentioned in lag_features)
    based on time_feature column. The highest value of time_feature is retained as a row
    and the lower values of time_feature are added as lagged_features
    :param df_long: Data frame (longitudinal) to create lagged features on
    :param lag_features: A list of columns to be lagged
    :param window: How many lags to perform (0 means no lagged feature will be produced)
    :param widths: How many days to roll back and calculate the statistical features
    :param lag_prefix: Prefix to name lagged columns.
    :param lag_prefix_sep: Separator to use while naming lagged columns
    :return: Data Frame with lagged features appended as columns
    """
    if not isinstance(lag_features, list):
        # So that while sub-setting DataFrame, we don't get a Series
        lag_features = [lag_features]

    if window <= 0:
        return df_long

    df_working = df_long[lag_features].copy()
    df_result = df_long.copy()
    for i in range(1, window+1):
        df_temp = df_working.shift(i)
        df_temp.columns = [lag_prefix + lag_prefix_sep + str(i) + lag_prefix_sep + x
                           for x in df_temp.columns]
        df_result = pd.concat([df_result.reset_index(drop=True),
                               df_temp.reset_index(drop=True)],
                               axis=1)
    for width in widths:
        
        window2 = df_working.rolling(window=width)
        window3 = df_long[['Discount_amount']].copy().shift(width - 1).rolling(window=width)

        dataframe = window2.sum()
        dataframe.columns = [lag_prefix+'_past_'+str(width)+'_sum']
        dataframe2 = window3.max()
        dataframe2.columns = [lag_prefix+'_past_'+str(width)+'_max_amount']
        dataframe3 = window3.mean()
        dataframe3.columns = [lag_prefix+'_past_'+str(width)+'_mean_amount']
        df_result = pd.concat([df_result.reset_index(drop=True),
                               dataframe.reset_index(drop=True)],
                               axis=1)
        df_result = pd.concat([df_result.reset_index(drop=True),
                               dataframe2.reset_index(drop=True)],
                               axis=1)
        df_result = pd.concat([df_result.reset_index(drop=True),
                               dataframe3.reset_index(drop=True)],
                               axis=1)

    return df_result


def predict_future(df_long, future_features):
    """
    Function calculates future features (only for columns mentioned in future_features)
    based on time_feature column. The highest value of time_feature is retained as a row
    and the lower values of time_feature are added as future_features
    :param df_long: Data frame (longitudinal) to create future features on
    :param future_features: A list of columns to be forwarded
    :return: Data Frame with future features appended as columns
    """
    if not isinstance(future_features, list):
        # So that while sub-setting DataFrame, we don't get a Series
        future_features = [future_features]

    df_working = df_long[future_features].copy()[::-1]
    df_result = df_long.copy()

    future = df_working.rolling(window=7).max()[::-1]
    future.columns = ['Discount_next_7days']
    future_amount = df_working.rolling(window=14).max()[::-1]
    future_amount.columns = ['Discount_next_14days']

    df_result = pd.concat([df_result.reset_index(drop=True),
                               future.reset_index(drop=True)],
                               axis=1)
    df_result = pd.concat([df_result.reset_index(drop=True),
                               future_amount.reset_index(drop=True)],
                               axis=1)

    return df_result

All_more_features = []
for df in All:
    df = str2datetime(df, ["Posted_date"])
    df = add_time_features(df)
    df = add_more_time_features(df)
    df = add_holiday_features(df)
    df = get_week_of_month(df)
    df['Discount_YesOrNo'] = 1
    df.loc[df['Discount_amount'] == 0, 'Discount_YesOrNo'] = 0
    df = lagged_features(df, ['Discount_YesOrNo'], window=7, widths=[15,30])
    df = predict_future(df, ['Discount_amount'])
    All_more_features.append(df)


lancome_discount = All_more_features[0]


sns.factorplot('Posted_month', data = lancome_discount[lancome_discount['Discount_amount'] != 0], kind='count')


labels = ['Extra 15% off', 'Extra 20% off', 'Extra 25% off', 'Extra 30% off']
sizes = [lancome_discount[(lancome_discount['Posted_month'] == 11)&(lancome_discount['Discount_amount'] == 0.15)].shape[0],
         lancome_discount[(lancome_discount['Posted_month'] == 11)&(lancome_discount['Discount_amount'] == 0.2)].shape[0],
         lancome_discount[(lancome_discount['Posted_month'] == 11)&(lancome_discount['Discount_amount'] == 0.25)].shape[0],
         lancome_discount[(lancome_discount['Posted_month'] == 11)&(lancome_discount['Discount_amount'] == 0.3)].shape[0]
        ]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()


sns.factorplot('Posted_month', data = lancome_discount[lancome_discount['Discount_amount'] != 0], kind='count')


ax = sns.lineplot(x="Posted_weekofmonth", y="Discount_amount",
             data=lancome_discount[(lancome_discount['Posted_month'] == 11) & (lancome_discount['Discount_amount'] != 0)])
ax.set(xlabel='Week of the month', ylabel ='Discount amount')
plt.show()


lancome_discount['target'] = 0
lancome_discount.loc[lancome_discount['Discount_next_7days'] == 0.2, 'target'] = 1
lancome_discount.loc[lancome_discount['Discount_next_7days'] == 0.25, 'target'] = 2
lancome_discount.loc[lancome_discount['Discount_next_7days'] == 0.15, 'target'] = 1
lancome_discount.loc[lancome_discount['Discount_next_7days'] == 0.3, 'target'] = 2



# ## Forward-chaining Cross-validation:

train_lancome = lancome_discount[(lancome_discount['Posted_date'] < datetime(2018, 6, 1)) & (lancome_discount['Posted_date'] > datetime(2015, 6, 1))]
test_lancome = lancome_discount[(lancome_discount['Posted_date'] >= datetime(2018, 6, 1))]



# Define object to handle time series splits
tscv = TimeSeriesSplit(n_splits=3)

# Loop over time series splits, fit model, and test on test data
feature_to_use = [col for col in train_lancome.columns if col not in ['Posted_year', 'Posted_day', 'Discount_amount', 'Posted_date', 'target',
                                                                   'Discount_next_7days', 'Discount_next_14days', 'Discount_YesOrNo',
                                                                   'Posted_month_year',"days_awayNew Year's Day", 'days_awayMartin Luther King, Jr. Day', 
                                                                   "days_awayWashington's Birthday", 'days_awayMemorial Day',
                                                                   'days_awayIndependence Day', 'days_awayLabor Day',
                                                                   'days_awayColumbus Day', 'days_awayVeterans Day',
                                                                   'days_awayThanksgiving', 'days_awayChristmas Day']] # Time information has been captured in Hours, Minutes, Seconds
independent_col = ['target']
for train_index, test_index in tscv.split(train_lancome):
    print('------------------------------------')
    X_train, X_test = train_lancome.iloc[train_index][feature_to_use], train_lancome.iloc[test_index][feature_to_use]
    y_train, y_test = train_lancome.iloc[train_index][independent_col].values.ravel(), train_lancome.iloc[test_index][independent_col].values.ravel()
    
    # Fit logistic regression model to train data and test on test data
    #lr_mod = LogisticRegression(C = 2, penalty='l2', max_iter=1000)  # The value of C should be determined by nested validation
    #lr_mod.fit(X_train, y_train)
    
    #gnb = ComplementNB(alpha=2.0)
    #y_pred = gnb.fit(X_train, y_train).predict(X_test)
    # Fit XGBoost Classifier
    gbm = xgb.XGBClassifier(eval_metric = 'aucpr', reg_lambda = 0.4, reg_alpha = 0.5, colsample_bytree = 1, max_depth =9, n_estimators = 1300,seed=0)
    gbm.fit(X_train, y_train)
    #prediction = gbm.predict(X_test)
    
    
    #y_pred = lr_mod.predict(X_test)
    y_pred = gbm.predict(X_test)
    
    #Print Precision, Recall and F1 scores
    print(classification_report(y_test, y_pred))
    
plot_importance(gbm, importance_type='gain')
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()


gbm.fit(test_lancome[feature_to_use][:-7], test_lancome['target'][:-7])


prediction_gbm = gbm.predict(test_lancome[feature_to_use])
print(classification_report(test_lancome['target'][:-7], prediction_gbm[:-7]))


prediction_lr = lr_mod.predict(test_lancome[feature_to_use])
print(classification_report(test_lancome['target'][:-7], prediction_lr[:-7]))


prediction_gnb = gnb.predict(test_lancome[feature_to_use])
print(classification_report(test_lancome['target'], prediction_gnb))


feature_target = feature_to_use
feature_target.append('target')


# In[41]:


corr = train_lancome[feature_target].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")
plt.figure(figsize = (32,15))
ax = sns.heatmap(corr, mask=mask, annot=True,
            square=True, linewidths=1,cbar=False)
            #cbar_kws={'label': 'Correlation coefficient'})
ax.set_xticklabels(
    ['Month', 'Day of week', 'Day of year', 'Proximity to holiday', 'Week of month',
    'Sales event 1 day before', 'Sales event 2 days before', 'Sales event 3 days before', 
    'Sales event 4 days before', 'Sales event 5 days before', 'Sales event 6 days before',
    'Sales event 7 days before',
    'Number of events (last 15 days)', 'Maximum discount (last 15 days)', 'Average discount (last 15 days)',
    'Number of events (last 30 days)', 'Maximum discount (last 30 days)', 'Average discount (last 30 days)'],
    rotation=45,
    fontsize=18,
    horizontalalignment='right'
);
ax.set_yticklabels(
    ['Month', 'Day of week', 'Day of year', 'Proximity to holiday', 'Week of month',
    'Sales event 1 day before', 'Sales event 2 days before', 'Sales event 3 days before', 
    'Sales event 4 days before', 'Sales event 5 days before', 'Sales event 6 days before',
    'Sales event 7 days before',
    'Number of events (last 15 days)', 'Maximum discount (last 15 days)', 'Average discount (last 15 days)',
    'Number of events (last 30 days)', 'Maximum discount (last 30 days)', 'Average discount (last 30 days)',
    'Target'],
    fontsize=22,
    verticalalignment = 'center'
);


# ## Use XGBoost as the final model:

def Forward_Chaining(df_train, df_test, days, n=3):

# Define object to handle time series splits
    tscv = TimeSeriesSplit(n_splits=n)

# Loop over time series splits, fit model, and test on test data
    feature_to_use = [col for col in df_train.columns if col not in ['Posted_year', 'Posted_day','Discount_amount', 'Posted_date', 'target',
                                                                   'Discount_next_7days', 'Discount_next_14days', 'Discount_YesOrNo',
                                                                   'Posted_month_year',"days_awayNew Year's Day", 'days_awayMartin Luther King, Jr. Day', 
                                                                   "days_awayWashington's Birthday", 'days_awayMemorial Day',
                                                                   'days_awayIndependence Day', 'days_awayLabor Day',
                                                                   'days_awayColumbus Day', 'days_awayVeterans Day',
                                                                   'days_awayThanksgiving', 'days_awayChristmas Day']] # Time information has been captured in Hours, Minutes, Seconds
    independent_col = ['target']
    for train_index, test_index in tscv.split(df_train):
        print('------------------------------------')
        X_train, X_test = df_train.iloc[train_index][feature_to_use], df_train.iloc[test_index][feature_to_use]
        y_train, y_test = df_train.iloc[train_index][independent_col].values.ravel(), df_train.iloc[test_index][independent_col].values.ravel()
    
    # Fit XGBoost Classifier
        gbm = xgb.XGBClassifier(eval_metric = 'aucpr', reg_lambda = 0.4, reg_alpha=0.5, colsample_bytree = 1, max_depth =9, n_estimators = 1300, seed=0)
        gbm.fit(X_train, y_train)
        y_pred = gbm.predict(X_test)
    
    #Print Precision, Recall and F1 scores
        print(classification_report(y_test, y_pred))
    
    plot_importance(gbm, importance_type='gain')
    plt.show()

    gbm.fit(df_train[feature_to_use], df_train['target'])
    prediction = gbm.predict(df_test[feature_to_use][:-days])
    print(classification_report(df_test['target'][:-days], prediction))
    return gbm

def last_train(model, df_test, features):
    model.fit(df_test[features], df_test['target'])
    return model
    


model_7_lancome_2=Forward_Chaining(train_lancome, test_lancome, 7, 3)


print(classification_report(y_test, prediction))


# ## Output the final models:

pickle.dump(model_lancome, open('front_end/models/Nike_2019-05-29.sav', 'wb'))


# In[1985]:


pickle.dump(model_14_lancome_, open('front_end/models/Nike_14days_2019-05-29.sav', 'wb'))


# In[2000]:


