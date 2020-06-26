# Savvy Care
This is a 3 weeks Insight Data Science Project.
..* 
..* 

## Table of Contents

- [Project Description](#Description)
- [How Does it Work?](# How Does it Work?)
- [Results](#Results)
- [Built With](# Built With)

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
The scraped data containg discount information like

- **Gift With Purchase**

- **Features**

| Calendar Features        | Lag Features           | Historical Statistics  |
| -------------------------|:----------------------:|:----------------------:|
| Year, Month, Day, Weekend Flag, A week before which holiday         | Shifted deals in 1 day, 1 week, 1 month, and 1 year | Means, maxs, sum  |


### Modeling

_Classification_
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
Random Search is used for several optimal algorithms selected from spot checking. 
___
## Results
### Model Evaluation
_Regression using RMSE:_
- Extra trees regression with hyperparameter tuning: _**0.0027**_
- Naive forecast: _**0.0012**_

_Classification using area under ROC curve:_
- XGboost with hyperparameter tuning: _**0.76**_
- Logistic regression: _**0.48**_

### Feature Importance
_Interesting observations:_
  1.  Memorial Day has most discounts



___
## Built With

###### This is still under development
