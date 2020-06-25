# Savvy Care
This is a 3 weeks Insight Data Science Project.
..* 
..* 

## Table of Contents

- [Project Description](#Description)
- [How Does it Work?](#How Does it Work?)
- [Results](#Results)
- [Built With](#Built With)

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
--EDA.py

### Modeling

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
