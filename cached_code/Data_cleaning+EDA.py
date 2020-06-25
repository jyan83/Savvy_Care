import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter


# Import the datasets

lancome = pd.read_csv("Lancome.csv")


All = [lancome]


## Data Cleaning:

## 1. Separate the real *deals* from the *ads*
# Observation: Real deals or special offers typiclly have end dates.  
# Action: Separate them by checking whether a "deal ends" date exists.

def end_date_filter(df):
    df_w_end_date = df[df['End_date'].isnull() == False]
    df_w_o_end_date = df[df['End_date'].isnull() == True]
    return df_w_end_date

lancome = end_date_filter(lancome)


All_deals = [lancome]


## 2. Convert time features into `datetime` format
def str2datetime(df, cols):
    for col in cols:
        #if col == 'Posted_date':
        #    temp= pd.to_datetime(df[col],format='%m-%d-%Y')# - timedelta(1)
        try:
            temp= pd.to_datetime(df[col],format='%m/%d/%Y')
        except:
            temp= pd.to_datetime(df[col],format='%Y-%m-%d')
        df.loc[:,col] = temp
    return df



for deal in All_deals:
    deal = str2datetime(deal, ["Posted_date", "End_date"])


## 3. Combine duplicates
# 
# Observation: Deal posts with the same "Deal ends" date tend to be duplicate posts for the same deal.  
# Action: Group by the `End_date`.

agg_func = {
            'Title': lambda a: " / ".join(a),
            'Description': lambda a: " / ".join(a), 
            'Posted_date': 'min',
            'Comments_count': 'sum',
            'Bookmarks_count': 'sum',
            'Shares_count': 'sum'
            }

All_deals_agg = []
for deal in All_deals:
    All_deals_agg.append(deal.groupby("End_date", as_index=False).agg(agg_func))


## 3. Add more time related features

def add_time_features(df, prefix='Posted'):
    
    # The string prefix must be 'Posted' or 'End'.
    # More features can be added, eg. day of week, week of month and holiday features.
    
    df[prefix+'_year'] = df[prefix+'_date'].dt.year
    df[prefix+'_month'] = df[prefix+'_date'].dt.month
    df[prefix+'_month_year'] = df[prefix+'_date'].dt.to_period('M')
    return df


for df in All_deals_agg:
    for prefix in ['Posted', 'End']:
        df = add_time_features(df, prefix)


for deal in All_deals_agg:
    for i in range(deal.shape[0]):
        if pd.to_datetime(deal['End_date'].values[i]) + timedelta(1) == pd.to_datetime(deal['Posted_date'].values[i]):
            # This corrects the mislabeling of the Posted_date due to the comparision with datetime.today() 
            deal.loc[deal.index[i],'Posted_date'] = pd.to_datetime(deal.iloc[i]['End_date'])
        if deal['End_date'].values[i] < deal['Posted_date'].values[i]:
            #print(i)
            # This corrects the mislabeling the year of the End_date when scraping the website
            deal.iloc[i,0] = pd.to_datetime(deal.iloc[i]['End_date']) + timedelta(365)



lancome_deals = All_deals_agg[0]



#%% Exploratory Data Analysis:

lancome_deals.head(5)


# ## How the deals are distributed over months:

sns.factorplot('Posted_month', data = lancome_deals, kind='count')


sns.catplot(x='Posted_month', y='Bookmarks_count', kind='box', data =lancome_deals)


# ## Extract discount features from the deal description:

# *Based on the EDA, we found the following key words related to the discount:*
# 1. extra xx% off, up to xx% off and xx% off:  
# `regex`: `'((extra|(up+\s+to))\s)?\d+\%+\s+off'`
# 2. $\$$x.xx $\$$x.xx, $\$$xxx($\$$xxx), i.e. current price original price   :  
# `regex`: `'\$(\d)+(.\d{2})?(\s*)+\(?\$\d+(.\d{2})?\)?'`
# 3. $\$$ xx off $\$$ xx or $\$$ xx get $\$$ xx:  
# `regex`: `'\$(\d)+(.\d{2})?\s(off|get)\s\$\d+(.\d{2})?\)?'` 
# 4. Whether `select products` appears:  
# `regex`: `'select'`

reg_all = ['(extra|up+\s+to)?\s?(\d+\%+\s+off)', 
           '\$\d+.?\d*(?!\n)\s*\(?\$\d*.?\d*\)?', 
           '\$\d*.?\d{2}?\s+[get|off]+\s\$\d*.?\d{2}?',
           'select']


def p2f(x):
    x = x[:-4]
    return float(x.strip('%'))/100

def dollar2f(x):
    y = x.strip(', ')
    y = re.sub(r'[^\w\s]','',y)
    return float(y.strip('$'))

def find_dis_amount(df):
    n = len(df['Description'].values)
    dis_amount = []
    dis_label = []
    for x in range(n):
        m = []
        for reg in reg_all:
            m.append(re.findall(reg, df['Description'].values[x],flags=re.IGNORECASE))
        m_new = []
        label = None
        if m[0]:
            i = np.shape(m[0])[0]
            j = np.shape(np.shape(m[0]))[0]
            m_new=[p2f(m[0][k][-1]) for k in range(i)]
            if j > 1:
                label = [m[0][k][0].lower() for k in range(i) if m[0][k][0]]
        if m[1]:
            i = np.shape(m[1])[0]
            j = np.shape(np.shape(m[1]))[0]
            try:
                temp = m[1][0].split("$")
                prices = []
                for price in temp:
                    if not price:
                        continue
                    else:
                        prices.append(price)
                current, original = prices
            except:
                print(x)
                m_new.append(0)
            
            current = dollar2f(current)
            original = dollar2f(original)
            discount = current / original
            m_new.append(round(1-discount,2))
        if not m_new:
            dis_amount.append(0)
            dis_label.append('none')
        else:
            if label:
                if 'extra' in label:
                    dis_amount.append(m_new[label.index('extra')])
                    dis_label.append(label[label.index('extra')])
                elif 'up to' in label:
                    dis_amount.append(m_new[label.index('up to')])
                    dis_label.append(label[label.index('up to')])
                else:
                    dis_amount.append(m_new[0])
                    dis_label.append(label[0])
            else:
                dis_amount.append(m_new[0])
                dis_label.append('none')
        #if not label:
        #    dis_label.append('none')
       # else:
         #   dis_label.append(label[0])
    df['Discount_amount'] = dis_amount
    df['Discount_label'] = dis_label
    return df


lancome_deals = find_dis_amount(lancome_deals)


sns.pointplot(x="Posted_date", y="Discount_amount",
             hue="Discount_label",
             data=lancome_deals)

#
#sns.pointplot(x="Posted_month", y="Discount_amount",
#             data=lancome_deals[(lancome_deals['Discount_label'] == 'extra')])



def add_more_time_features(df, prefix='Posted'):
    
    # The string prefix must be 'Posted' or 'End'.
    # More features can be added, eg. day of week, week of month and holiday features.
    
    df[prefix+'_dayofweek'] = df[prefix+'_date'].dt.dayofweek
    df[prefix+'_day'] = df[prefix+'_date'].dt.day
    return df

lancome_deals=add_more_time_features(lancome_deals)



# Save only the "extra xx% off" type of sales event:

def new_discount_df(df, extra=True):
    Dates = pd.date_range(df['Posted_date'].min(), df['End_date'].max(), freq='D')
    #Dates = pd.date_range(df['Posted_date'].min(), pd.datetime(2019,11,1), freq='D')
    new_disc = []
    if extra:
        deals_extra = df[df['Discount_label'] == 'extra'].sort_values(by='Posted_date')
    else:
        deals_extra = df[df['Discount_label'] != 'up to'].sort_values(by='Posted_date')
    i = 0
    for date in Dates:
        if i < deals_extra.shape[0] and date >= deals_extra.iloc[i]['Posted_date']:
            #if deals_extra.iloc[i]['End_date'] < deals_extra.iloc[i]['Posted_date']:
            #    end_date = deals_extra.iloc[i]['End_date'] + timedelta(365)
            #else:
            end_date = deals_extra.iloc[i]['End_date']
            if date <= end_date:
                #print(i)
                new_disc.append(deals_extra.iloc[i]['Discount_amount'])
            else:
                new_disc.append(0)
                i += 1
        else:
            new_disc.append(0)
    Discount_amount = pd.DataFrame({'Posted_date': Dates, 'Discount_amount': new_disc})
    return Discount_amount


lancome_discount = new_discount_df(lancome_deals)


lancome_discount.to_csv('lancome_clean.csv', index=False)





