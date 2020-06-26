# Savvy Care
This is a 3 weeks Insight Data Science Project.

## Table of Contents

- [Project Description](#Description)
- [How Does it Work?](#How-Does-it-Work?)
- [Results](#Results)
- [Built With](#GitHib-files)

___
## Description
People spend a lot on beauty maintenance, the Average Cost of Beauty Maintenance Could Put You Through Harvard!
Skincare products have long-shelf life, and always have discount. Why not refill your skincare when there is a discount? 
Deals & Steals tool leverages ML models for discount estimation to support cost reduction and purchasing decisions for loyal skincare customers.

### Discount Prediction
___
## How Does it Work?
### Data Collection

Data is collected from Dealmoon.com. It is a online shopping guiding website that showing the current deals and the expired deals. 
A web scrawler using Selenium is built to collect the available deals information. Discount info of 12 popular skincare brands from 2010 to 2020 June were obtained. 

### Feature Engineering
- **Discount**

The scraped data containg discount information like "XX % OFF", "Buy 1 Get 1 Free", ... All of these are going to tranfer to percetange off. To predict the dicount, this becomes a regression problem.

- **Gift With Purchase (GWP)**

There are also a lot of deals are not having percetange off discount, but free gift with purchase. Based on the popularity from bookmarks, comments, and shares data, we can see GWP is also very attractive. Thus, beyond regression problem to predict the % off discount, a binary classification problem of if there is a gift with purchase is proposed.

- **Features**

| Calendar Features        | Lag Features           | Historical Statistics  |
| -------------------------|:----------------------:|:----------------------:|
| Year, Month, Day; Weekend Flag; A week before which holiday | Shifted deals in 1 day, week, month| Means, maxs, sum  |

In total, 18 features used.

### Modeling

- **Regression**

1. Time Series Cross-Validation

	Models will be evaluated using a scheme called walk-forward validation. Walk-forward is very similar to K-Fold except that it ignores the data after the test set.

2. Baseline: Naive Methods

	First develop and evaluate and compare the performance a suite of naive persistence forecasting methods. It provides a quantitative idea of how difficult the forecast problem is and provide a baseline performance by which more sophisticated forecast methods can be evaluated.
From EDA, the pattern of the data does not suitable for the regression or time series methods. 

3. Nonlinear Models

	Spot checking machine learning algorithms means evaluating a suite of different types of algorithms with minimal hyperparameter tuning.
	
- **Nonlinear Algorithms**
	- Decision Tree
	- k-Nearest Neighbors

- **Ensemble Algorithms**
	- Random Forest
	- Extra Trees
	- Adaboosting
	- XGBoost
	
4. Hyperparamter Tunning

	Random Search is used for Extra Trees Regressor selected from spot checking. 

- **Classification**

Here, gift with purchase is rare comparing to the whole dataset, this is an imbalanced classification. 

1. Selecting A Metric

	Because the positive class (GWP is True) is the most important, selecting a metric is the most important step in the project. The area under curve can be used. This will maximize the true positive rate and minimize the false positive rate.

2. Data Sampling

	Data sampling algorithms change the composition of the training dataset to improve the performance of a standard machine learning algorithm on an imbalanced classification problem. Here, data oversampling is used through SMOTE

3. Spot Checking Algorithms

	Spot checking machine learning algorithms means evaluating a suite of different types of algorithms with minimal hyperparameter tuning.

- **Linear Algorithms**
	- Logistic Regression
	- Naive Bayes
	
- **Nonlinear Algorithms**
	- Decision Tree
	- k-Nearest Neighbors
	- Support Vector Machine

- **Ensemble Algorithms**
	- Random Forest
	- Extra Trees
	- XGBoost

4. Hyperparamter Tunning

> Random Search is used for several optimal algorithms selected from spot checking. 
___
## Results
### Model Evaluation

Using Lancome as an example:

_Regression using RMSE:_
- Extra trees regression with hyperparameter tuning: _**0.0027**_
- Naive forecast: _**0.0012**_

_Classification using area under ROC curve:_
- XGboost with hyperparameter tuning: _**0.73**_
- Logistic regression: _**0.65**_

### Interesting observations
Refill your skincare on Memorial Day!

___
## GitHib files

### Web scraping
* `Data/Dealmoon/Web_scraping.py`: uses *Streamlit* to connect to Dealmoon.com to get web content, and *Pandas* to clean and store scraped content as a dataframe

### Data analysis and modeling
* `Script/EDA.py`: these files load .csv files, concatenate dataframes, pull out data of interest, rename/reorder columns, and remove spurious listings. 

* `Script/Regression.py`: Feature engineering for regression, and regression modeling.

* `Script/Classification.py`: Feature engineering for classification, and classification modeling.

#### Others
Some trial files using Lancome for example:

* `Script/Baseline_Regression_Naive_Methods.py`: run Naive Forecast for regression.

* `Script/Baseline_Regression_Modeling.py`: run some linear regression models for regression.

* `Script/Regression_ML_lancome.py`: run some nonlinear regression models for regression.

* `Script/Classification_lancome.py`: run some linear regression models for regression.


### Web app development

* `Streamlit run Frontend/Savvy_Care.py`: load web page.

### Requirements
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Statsmodels](https://www.statsmodels.org/stable/index.html)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [TSCV](https://github.com/WenjieZ/TSCV)

