# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 01:28:58 2020

@author: Jin Yan
"""

## Import modules:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB



## Import data:
df_r = pd.read_csv("Data/Dealmoon/lancome.csv")
df_r.drop(["Brand"], axis=1, inplace=True)
print(df_r.shape)
df_r.head()

df_r['Posted_date'] =  pd.to_datetime(df_r['Posted_date'], errors='coerce')

df_r['Discount'].value_counts()
starting_year = df_r['Posted_date'].dt.year.min()  
current_year = datetime.datetime.now().date().year  

def combine(df_r):
    # Observation: Deal posts with the same "Deal ends" date tend to be duplicate posts for the same deal.  
    # Action: Group by the `posted date`.
    
    agg_func = {'Store': lambda a: a.unique(),
                'Discount': lambda a: a.unique(), 
                'Posted_date': 'min',
                'Comments_count': 'sum',
                'Bookmarks_count': 'sum',
                'Shares_count': 'sum'
                }
    
    
    df_r_agg = df_r.groupby(["Posted_date"]).agg(agg_func).reset_index(drop=True)
    return df_r_agg


df_n1 = combine(df_r)
df_r = df_n1.copy()

## Add missing date to dataframe
#idx = pd.date_range(df_r['Posted_date'].min(), df_r['Posted_date'].max())
#df_r = df_r.set_index('Posted_date').reindex(idx).fillna(0.0).rename_axis('Posted_date').reset_index()    

def add_holidays(df_r, starting_year, current_year):
    ## Import holidays
    import holidays
    US_holidays = sorted(holidays.US(years=range(starting_year,current_year)).items())
    
    # Add mother's day & farther's day to the holiday events
    import calendar
    
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    
    mothers_day = []
    fathers_day = []
    for year in range(starting_year,2020):
        monthcal = c.monthdatescalendar(year, 5)
        mothers_day.append( [day for week in monthcal for day in week if \
                        day.weekday() == calendar.SUNDAY and \
                        day.month == 5][2])
        monthcal = c.monthdatescalendar(year, 6)
        fathers_day.append( [day for week in monthcal for day in week if \
                        day.weekday() == calendar.SUNDAY and \
                        day.month == 6][2])
        
    for i in range(len(mothers_day)):
        US_holidays.append((mothers_day[0],"Mother's day"))
        US_holidays.append((fathers_day[0],"Fathers's day"))
    
    
    # adding weekend and holiday features.
    # discount happening before 7 days of the holidays
    df_r = df_r.assign(
        year        = lambda df_r: df_r['Posted_date'].dt.year,
        month       = lambda df_r: df_r['Posted_date'].dt.month,
        day         = lambda df_r: df_r['Posted_date'].dt.day,
        weekday     = lambda df_r: df_r['Posted_date'].dt.day_name(),
        Weekend_FLG = lambda df_r: df_r['weekday'].apply(lambda day: True if day in ['Friday', 'Saturday'] else False),
        Event_1day       = lambda df_r: df_r['Posted_date'].apply(lambda day: US_holidays[min(range(len(US_holidays)), key=lambda i: abs(US_holidays[i][0]-day.date()))][1] 
                                                        if abs((US_holidays[min(range(len(US_holidays)), key=lambda i: abs(US_holidays[i][0]-day.date()))][0] - day.date()).days)<1 else 'Normal'
                                                         ),
        Event_7days       = lambda df_r: df_r['Posted_date'].apply(lambda day: US_holidays[min(range(len(US_holidays)), key=lambda i: abs(US_holidays[i][0]-day.date()))][1] 
                                                        if abs((US_holidays[min(range(len(US_holidays)), key=lambda i: abs(US_holidays[i][0]-day.date()))][0] - day.date()).days)<7 else 'Normal'
                                                         ),
        Event_14days       = lambda df_r: df_r['Posted_date'].apply(lambda day: US_holidays[min(range(len(US_holidays)), key=lambda i: abs(US_holidays[i][0]-day.date()))][1] 
                                                        if abs((US_holidays[min(range(len(US_holidays)), key=lambda i: abs(US_holidays[i][0]-day.date()))][0] - day.date()).days)<14 else 'Normal'
                                                         ),
        Event_1month       = lambda df_r: df_r['Posted_date'].apply(lambda day: US_holidays[min(range(len(US_holidays)), key=lambda i: abs(US_holidays[i][0]-day.date()))][1] 
                                                        if abs((US_holidays[min(range(len(US_holidays)), key=lambda i: abs(US_holidays[i][0]-day.date()))][0] - day.date()).days)<30 else 'Normal'
                                                         ),
    )
        
    return df_r

df_r = add_holidays(df_r, starting_year, current_year)
    
#%% Discount label
# comment out to see the discount info
#df_r['Discount'].value_counts()
df_n = df_r.copy()  
def drop_ad(df_r):
    # drop the advertisement from the website 
    df_n = df_r.copy() 
    df_n['Discount'].loc[df_n.Discount.str.contains('|'.join(['Dealmoon', 'DEALMOON']))==False]
    df_n['Discount'].fillna(0, inplace=True)
    return df_n 
  
def extract_off(df_r):
    #extract the % OFF discount    
    df_n = df_r.copy()
    df_n['Discount_off'] = df_n['Discount'].str.extract(r'(?P<discount>\d+)[%]')
    df_n['Discount_off'] = df_n['Discount_off'].astype('float32')/100
    
    df_n['Discount_off'].loc[df_n['Discount'].str.contains('|'.join(['BOGO','Buy one get one free']))==True] = 0.5
    df_n['Discount_off'].fillna(0, inplace=True)
    return df_n

def extract_gwp(df_r):
    # extract GWP (gift with purchase)
    df_n = df_r.copy()
    df_n['GWP'] = df_n.Discount.str.contains('|'.join(['Gift', 'Gifts', 'Samples', 'Sample', 'GWP', 'Travel', 'Full']))
    df_n['GWP'].fillna(False, inplace=True)
#    df_n= df_n[df_n['Discount_off'].notna() | df_n['GWP']]
    return df_n


df_n = drop_ad(df_n)
df_n = extract_off(df_n)
df_n = extract_gwp(df_n)

print(df_n.shape)
df_n.head(20)


#%% Exploratory Visualization
# sort dataframe by date
df_n = df_n.sort_values(by=['Posted_date'])


# First visualize the discount in time domain
fig = plt.figure()
ax=plt.subplot()
ax.plot(df_n.Posted_date, df_n.Discount_off, 'o')
ax.set_xlabel(u"Observed days")
ax.set_xticks([])
ax.set_ylabel(u"Discount by % OFF")

ax2 = ax.twinx()
ax2.plot(df_n.Posted_date, df_n.GWP, 'd', c='red')
ax2.set_ylabel(u"Discount: GWP (gift with purchase)")
ax2.yaxis.set_label_position("right")
plt.yticks([1.0, 0.0], ["True", "False"])
ax2.set_title(u"Deals starting from "+ str(starting_year))

# Which discount is better? % OFF or GWP?
# Correlation matrix
corr = df_n.corr()
fig = plt.figure()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# Is there seasonality to the sales?
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# see the discount occurence by month
sns.catplot('month', data = df_r, kind='count', ax=ax1)

# see the discount occurence by weekdays/weekend
sns.catplot('weekday', data = df_r, kind='count', 
            order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'], ax=ax2)
plt.close(3)
plt.close(4)
plt.tight_layout()
plt.show()

# Seasonal line plots
groups = df_n.groupby(Grouper(key='Posted_date',freq='A'))
years = DataFrame()
plt.figure()
i = 1
n_groups = len(groups)
for name, group in groups:
    ax=plt.subplot(n_groups,1,i)
    i += 1
    plt.ylabel(group.year.min())
    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim([0,365])	
    plt.plot(np.arange(len(group)), group['Discount_off'])
plt.xlabel('days')
plt.suptitle('Seasonal Line Plots')
plt.show()

#df_n.to_csv('lancome_clean.csv')
df_n.to_csv('lancome_clean2.csv')
