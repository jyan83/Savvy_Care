import numpy as np
import pandas as pd
from pandas import DataFrame

import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold, StratifiedKFold, cross_val_score 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, classification_report 

from scipy import stats

import pickle

all_categories = ['lancome', 'estee-lauder', 'la-mer', 'clinique', 'kiehls', 'clarins', 
                  'bobbi-brown-cosmetics','giorgio-armani-beauty','loccitane','origins']

for brand in all_categories:
    plt.close('all') 
    # Import cleaned dataset
    df = pd.read_csv("../Data/" + brand + "_clean.csv")

    def add_features(df):
        
        # select only discount and GWP days to keep after getting all the features
        a = df.index[(df['GWP'] == True) | (df['Discount_off'] != 0) ].tolist()
    
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
        
        dataframe = dataframe.ix[a]
        return dataframe, a
    
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
    
    
    X, list_index = add_features(df)
    
    X = X.replace(np.nan, 0)
    Y = df['GWP'].ix[list_index]
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
    
    #%% Classification Modeling
    
    RNG = 42

    # Select extreme gradient boosting machine to do the hyperparameter tunning
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
    
    # Compute FPR, TPR, Precision by iterating classification thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    
    # Plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='XGBoost ROC curve (area = %0.2f)' % roc_auc)
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
    