import numpy as np
import datetime
import pandas as pd
import streamlit as st

from matplotlib import pyplot as plt
import pickle

global brands_dict, brands_name_dict
brands_dict = ['', 'lancome', 'estee-lauder', 'la-mer', 'clinique', 'kiehls', 'clarins', 
                  'bobbi-brown-cosmetics','giorgio-armani-beauty','loccitane','origins']
    
brands_name_dict = ['', 'lancome', 'estee-lauder', 'la-mer', 'clinique', 'kiehls', 'clarins', 
                  'bobbi-brown-cosmetics','giorgio-armani-beauty','loccitane','origins']

def main():
    """Main function. Run this to run the app"""

    st.sidebar.title("SavvyCare")
    st.sidebar.header("Your Skincare Refill Management Tool")
    st.sidebar.markdown("Please choose the brand:")
    
    
            
    deal_name = st.sidebar.selectbox('', brands_dict, 
                                format_func=lambda x: 'Select an option' if x == '' else x)
    
    p_days = st.slider("Choose how long you are willing to wait: ", min_value=1, 
                              max_value=365, value=30, step=1)-1
    
#    st.sidebar.markdown('What discount type you want to predict...') 
#    
#    type_dict = ['', '% off', 'Gift with Purchse']
    
#    deal_type = st.sidebar.selectbox('', type_dict, 
#                                format_func=lambda x: 'Select an option' if x == '' else x)
    
    
    if (deal_name != "") :
        Y_pred = discount(deal_name, p_days)
        Y_pred_c = gwp(deal_name, p_days)
        st.header("Suggestion is: ")        
        message_deal(Y_pred, Y_pred_c)
        st.sidebar.markdown('You can change a selection to predict...')    

    
def discount(brand=None, p_days=30):
    
    PIK = 'datasets/' + brand + '_Regression.dat'

    file = open(PIK,'rb')
    object_file = pickle.load(file)
    file.close()
    
    [X_train, X_test, y_train, y_test, Y_pred, model] = object_file    
   
    if Y_pred.any() != 0.0:
        plt.figure(num=None, figsize=(6, 4), dpi=80, facecolor='w', edgecolor='k')
        plt.bar(np.arange(len(Y_pred[:p_days]))+1, Y_pred[:p_days]*100, color='k', label="% OFF")
    #    plt.bar(np.arange(len(Y_pred_c))+1, Y_pred_c, color='orangered', label="GWP", width=0.25)
        plt.xlim(0.01, p_days)
        plt.ylim(0.01, 100)
        plt.grid()
        plt.xlabel("Predication in N days")                 
        plt.ylabel("Discount in %")          
    
        name_index = brands_dict.index(brand)
                 
        plt.title("Prediction for "+ brands_name_dict[name_index])
        plt.legend()
        st.pyplot()    
    
    return Y_pred

def gwp(brand=None, p_days=30):
    
    PIK = 'datasets/Classification.dat'

    file = open(PIK,'rb')
    object_file = pickle.load(file)
    file.close()
    
    [X_train, X_test, y_train, y_test, Y_pred, clf] = object_file
    
    import matplotlib.colors as clrs
    plt.figure(num=None, figsize=(6, 2), dpi=80, facecolor='w', edgecolor='k')
    cmap = clrs.ListedColormap(['red', 'green'])
    plt.yticks([1.0, 0.0], ["True",
                            "False"])
    plt.scatter(x = np.arange(len(Y_pred[:p_days])), y= Y_pred[:p_days], c=(Y_pred[:p_days] != True).astype(float), marker='d', cmap=cmap)#plt.cm.get_cmap('RdBu'))
#    plt.bar(np.arange(len(Y_pred)), Y_pred, color='orangered', label="__nolabel__")
    plt.xlim(0.01, p_days)
    plt.ylim(0, 1)
    plt.grid()
    plt.xlabel("Predication in N days")                 
#    plt.ylabel("GWP")                   
    plt.title("Prediction for gift with purchase")
    st.pyplot()
    
    # return nearest GWP date:
    mdays, mdays_gwp = min([(x,y) for x,y in enumerate(Y_pred) if y == True])
    return Y_pred

    
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
    if Y_pred.any() != 0.0:
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
    else:
        st.markdown("This brand does not have discount record!")
        ndays, ndays_discount = min([(x,y) for x,y in enumerate(Y_pred_c) if y > 0])
        ndays_discount = ndays_discount*100
        return  {st.markdown("Refill it when there is a gift with purchse promotion!"),
                 st.markdown("The nearest GWP will happen in %s days"%ndays)}
    
main()
