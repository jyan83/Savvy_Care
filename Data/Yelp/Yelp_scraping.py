#Business Search      URL -- 'https://api.yelp.com/v3/businesses/search'
#Business Match       URL -- 'https://api.yelp.com/v3/businesses/matches'
#Phone Search         URL -- 'https://api.yelp.com/v3/businesses/search/phone'

#Business Details     URL -- 'https://api.yelp.com/v3/businesses/{id}'
#Business Reviews     URL -- 'https://api.yelp.com/v3/businesses/{id}/reviews'

#Businesses, Total, Region

# Import the modules
import requests
import json

# Define a business ID
business_id = 'try'

# Define my API Key, My Endpoint, and My Header
API_KEY = 'YET6wYkbN-QO1U50FFzQFhsVrhU7EtNKmkYSdqETZfrvuRXWOouqP5Y4yhtiu6s6AXbzLuWFcqDnuK0QkUrQb3yCw8rVeTfbU4wJvSzG0zmWv6wPjO8Mvsnoe4PWXnYx'
ENDPOINT = 'https://api.yelp.com/v3/businesses/search'.format(business_id)
HEADERS = {'Authorization': 'bearer %s' % API_KEY}


Popular_cities = ['New York', 'Seattle', 'Chicago', 'Los Angeles', 'San Diego', 'Atlant']

for i in Popular_cities:
    # Define my parameters of the search
    # BUSINESS SEARCH PARAMETERS - EXAMPLE
    PARAMETERS = {'term': 'Bubble Tea',
                  'limit': 50,
    #              'offset': 50,
    #              'radius': 40000,
                  'location': i}


    # Make a request to the Yelp API
    response = requests.get(url = ENDPOINT,
                            params = PARAMETERS,
                            headers = HEADERS)
    
    # Conver the JSON String
    business_data = response.json()
    
    # print the response
    #print(json.dumps(business_data, indent = 3))
    
    with open(i+'.json', 'w') as openfile:
    	json.dump(business_data, openfile)
    