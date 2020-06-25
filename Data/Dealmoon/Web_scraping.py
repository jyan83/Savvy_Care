import time
import numpy as np
import pandas as pd
import unicodecsv as csv
from datetime import datetime, timedelta
# imoport packages for web scraping:
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains

    
def days_trans(date):
    string = date.split()
    if ['sec','secs', 'min', 'mins', 'hr', 'hrs'] in string:
        res = datetime.now().date()
    else:
        num = [int(x) for x in string if x.isdigit()]
        res = datetime.now() - timedelta(days=num[0])
        res = res.date()
    return res
    
    
def crawling(url, brand, page_max=100):
    driver = webdriver.Chrome()
    driver.get(url)

#    master = [['Brand', 'Store', 'Discount', 'Posted_date', 'End_date', 'Comments_count', 'Bookmarks_count', 'Shares_count']]
    master = [['Brand', 'Store', 'Discount', 'Posted_date', 'Comments_count', 'Bookmarks_count', 'Shares_count']]

    page = 0
    while True:
                
        check_height = driver.execute_script("return document.documentElement.scrollHeight")
        while True:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            try:
                WebDriverWait(driver, 10).until(lambda driver: driver.execute_script(
                    "return document.body.scrollHeight;") > check_height)
                check_height = driver.execute_script(
                    "return document.body.scrollHeight;")
            except:
                break

        elements = driver.find_elements_by_class_name('mlist')

        for element in elements:
            temp = [brand]

            #------ find time the store id of deal
            store_id = ''
            
            try:
                store_id = element.find_element_by_class_name("r.j-r").find_element_by_tag_name('a').text
            except:
                pass
            
            temp.append(store_id)
                
            #------ find title of deal
            discount = ''
            try:
                discount = element.find_element_by_class_name("subtitle").text
            except:
                pass
            
            temp.append(discount)
            
##            #------ find description of deal
##            des = element.find_element_by_class_name('brief').text
#            
#            #------ find time the deal was posted
#            try:
#                time = element.find_element_by_class_name('ib.published-date').text
#                time = time[0:-4]  #strip away 'Posted' and 'ago'
#                post_date = datetime.today()
#                if time[-4:] == 'days':
#                    post_date -= timedelta(days=int(time[0:-5]))
#                
#            except Exception as e:
#                print ('===TIME NOT FOUND')
#                print ('===deal skipped')
#                continue
#            
#            temp.append(post_date.strftime("%m/%d/%Y"))                 
            
            #------ find the end date of the deal
            date = ''
            
            try:
                date = element.find_element_by_class_name("r.j-r").find_element_by_tag_name('span').text
            except:
                pass
            
            date = days_trans(date)
            
            temp.append(date)
            
            
            stats = element.find_element_by_class_name("stat-count")
            stats_nums = stats.find_elements_by_class_name("j-count")
            
            #------ find number of comments for the deal
            num_comments = 0
            try:
                num_comments = stats_nums[0].text
            except Exception as e:
                pass
            
            temp.append(num_comments)

            #------ find number of bookmarks for the deal
            num_bookmarks = 0
            try:
                num_bookmarks = stats_nums[1].text
            except Exception as e:
                pass
            
            temp.append(num_bookmarks)
            
            #------ find number of shares for the deal
            num_shares = 0
            try:
                num_shares = stats_nums[2].text
            except Exception as e:
                pass
            
            temp.append(num_shares)
            
            #------ append to master list
            master.append(temp)

        try:
            load = driver.find_element_by_class_name("next_link")
            page += 1
            print("Page {}".format(page))

            # See if the last page has been reached
            page_num = driver.find_element_by_class_name('pages').find_element_by_class_name('current').text
            
            if page_num == str(page_max):
                print ('Last page reached')
                break
            else:
                load.click()          
        except:
            print ("===Can't go to the next page")
            break 
            
    return master

def saveCSV(filename, data):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        
all_categories = ['lancome', 'estee%20lauder', 'la-mer', 'clinique', 'shiseido',
                  'kiehls', 'clarins', 'bobbi-brown-cosmetics','giorgio-armani-beauty']
#all_categories = ['giorgio-armani-beauty']
all_urls = []
for i in all_categories:
    all_urls.append('https://www.dealmoon.com/en/stores/'+i+'?sort=relevance&exp=y')

        
for i in range(len(all_categories)):
    print('Current scraping ' + all_categories[i])
    data = crawling(all_urls[i], all_categories[i])
    print('Finished scraping ' + all_categories[i] + ', data length: ' + str(len(data)))
    filename = all_categories[i] + '.csv'
    saveCSV(filename, data)        

