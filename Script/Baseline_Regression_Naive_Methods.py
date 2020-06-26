# naive forecast strategies
# Reference: https://machinelearningmastery.com/naive-methods-for-forecasting-household-electricity-consumption/

from math import sqrt
import numpy as np
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
    train, test = data[6:-365], data[-365:-1]
	# restructure into windows of weekly data
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# evaluate a single model
def evaluate_model(model_func, train, test):
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = model_func(history)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores, predictions

# daily persistence model
def daily_persistence(history):
	# get the data for the prior week
	last_week = history[-1]
	# get the total active power for the last day
	value = last_week[-1, 0]
	# prepare 7 day forecast
	forecast = [value for _ in range(7)]
	return forecast

# weekly persistence model
def weekly_persistence(history):
	# get the data for the prior week
	last_week = history[-1]
	return last_week[:, 0]

# week one year ago persistence model
def week_one_year_ago_persistence(history):
	# get the data for the prior week
	last_week = history[-52]
	return last_week[:, 0]

# load the new file
dataset = read_csv('Data/lancome_clean.csv', header=0, infer_datetime_format=True, parse_dates=['Posted_date'], index_col=['Posted_date'])
dataset = dataset[['Discount_off','GWP']]
# split into train and test
train, test = split_dataset(dataset.values)
# define the names and functions for the models we wish to evaluate
models = dict()
models['daily'] = daily_persistence
models['weekly'] = weekly_persistence
models['week-oya'] = week_one_year_ago_persistence
# evaluate each model
days = ['wed', 'thr', 'fri', 'sat', 'sun', 'mon', 'tue']

models_score = []
for name, func in models.items():
    # evaluate and get scores
    score, scores, predictions = evaluate_model(func, train, test)
    # summarize scores
    summarize_scores(name, score, scores)
    # plot scores
    plt.plot(days, scores, marker='o', label=name)
    models_score.append(score)
# show plot
plt.legend()
plt.show()

fig, ax = plt.subplots()
width = 0.15  # the width of the bars
labels = ['daily', 'weekly', 'week-oka']
x = np.arange(len(labels))  # the label locations
plt.bar(1 - width, models_score[0], width/2, label='daily')
plt.bar(1 , models_score[1], width/2, label='weekly')
plt.bar(1 + width, models_score[2], width/2, label='week-oka')
ax.set_xticks([0.8,1.3])
ax.set_xticklabels([])
plt.legend()


plt.figure()
output = np.concatenate(predictions).ravel()
plt.plot(np.arange(len(dataset['Discount_off'])), dataset['Discount_off'], c='gray', label = 'ground truth', linewidth=1)
plt.plot(np.arange(len(dataset['Discount_off'])-364,len(dataset['Discount_off'])), output, c='r', label = 'prediction')
plt.vlines(len(dataset['Discount_off'])-365,0,1.05, color='k',linestyles='solid', linewidth=2, linestyle = '--')
plt.ylim([0.0, 1.05])
plt.xlabel('Observed days')
plt.ylabel('Discount % OFF')
plt.title('Regression Naive Forecast Performance')
plt.legend(loc='upper left')
plt.show()

rmse = abs((dataset['Discount_off'].ix[-365:-1]-output).mean())