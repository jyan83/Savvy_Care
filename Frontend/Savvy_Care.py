from flask import Flask, render_template, request, jsonify
import numpy as np
import datetime
import pandas as pd
import streamlit as st
from typing import List, Optional

import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from datetime import datetime, timedelta
from PIL import Image
import random


def main():
    """Main function. Run this to run the app"""

    st.sidebar.title("SavvyCare")
    st.sidebar.header("Your Skincare Refill Management Tool")
    st.sidebar.markdown("Please choose the brand:")
    
    brands_dict = ['', 'Lancome', 'Estee Lauder', 'Lamer', 'Tom Ford', 'Clinique', 'Clinique',
                   'Shiseido', 'Kiehls', 'Clarins', 'CDP', 'Aramani', 'Bobbi Brown', 'Chantecaille']
            
    deal_name = st.sidebar.selectbox('', brands_dict, 
                                format_func=lambda x: 'Select an option' if x == '' else x)
    
#    st.sidebar.markdown('What discount type you want to predict...') 
#    
#    type_dict = ['', '% off', 'Gift with Purchse']
    
#    deal_type = st.sidebar.selectbox('', type_dict, 
#                                format_func=lambda x: 'Select an option' if x == '' else x)
    
    
    if (deal_name != "") :
        Y_pred, Y_pred_c = discount(deal_name)
        st.header("Suggestion is: ")        
        message_deal(Y_pred, Y_pred_c)
        st.sidebar.markdown('You can change a selection to predict...')    

    
def discount(brand=None):
    
    PIK = 'datasets/Regression.dat'

    file = open(PIK,'rb')
    object_file = pickle.load(file)
    file.close()
    
    [X_train, X_test, y_train, y_test, Y_pred, model] = object_file
    Y_pred = Y_pred[:28]
    
    PIK = 'datasets/Classification.dat'

    file = open(PIK,'rb')
    object_file_c = pickle.load(file)
    file.close()
    
    [X_train_c, X_test_c, y_train_c, y_test_c, Y_pred_c, clf_c] = object_file_c
    Y_pred_c = Y_pred_c[:28]
   
    plt.figure()
    plt.bar(np.arange(len(Y_pred))+1, Y_pred, color='k', label="% OFF")
    plt.bar(np.arange(len(Y_pred_c))+1, Y_pred_c, color='orangered', label="GWP", width=0.25)
    plt.xlim(0.01, 30)
    plt.ylim(0.01, 1)
    plt.grid()
    plt.xlabel("Predication in N days")                 
    plt.ylabel("Discount in %")                   
    plt.title("Prediction for "+ brand)
    plt.legend()
    st.pyplot()
    
    
    return Y_pred, Y_pred_c

#def gwp(brand=None):
#    
#    PIK = 'datasets/Classification.dat'
#
#    file = open(PIK,'rb')
#    object_file = pickle.load(file)
#    file.close()
#    
#    [X_train, X_test, y_train, y_test, Y_pred, clf] = object_file
#    
#    plt.figure()
#    plt.bar(np.arange(len(Y_pred)), Y_pred, color='orangered', label="__nolabel__")
#    plt.ylim(0, 370)
#    plt.ylim(0, 1)
#    plt.grid()
#    plt.xlabel("Predication in N days")                 
#    plt.ylabel("Yes")                   
#    plt.title("Prediction for gift with purchase")
#    st.pyplot()
#    
#    # return nearest GWP date:
#    mdays, mdays_gwp = min([(x,y) for x,y in enumerate(Y_pred) if y == True])
#    return mdays, mdays_gwp, Y_pred

    
def message_deal(Y_pred, Y_pred_c):
    # return the number of days in for the best discount
    max_discount_index = np.argmax(Y_pred)
    max_discount = Y_pred[max_discount_index]*100
    
    # return if the discount happen in today
    if Y_pred[0] == False:
        today = False
    else:
        today = True
        
    if Y_pred_c[0] == False:
        today_c = False
    else:
        today_c = True    
        
    # return nearest discount date:
    ndays, ndays_discount = min([(x,y) for x,y in enumerate(Y_pred) if y > 0])
    ndays_discount = ndays_discount*100
    
    if (today == False) & (today_c == False):
        st.markdown("No discount today :(")
    elif (today == False) & (today_c == True):
        st.markdown("You have gift with purchase today!")
    elif (today == True) & (today_c == True):
        st.markdown("Hurry up! You have gift with purchase and discount today!")
        
    return  {st.markdown("The largest discount this year will happen in %s days!"%max_discount_index),
             st.markdown("You will get %.0f %% Off, refill your skincare!"%max_discount),
             st.markdown("The nearst discount is going to be in %s days"%ndays+" with %.0f %% OFF!"%ndays_discount)}
    
main()
