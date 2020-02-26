import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datalore.plot import *
from datalore.geo_data import *

confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')

"""
As we can see, countries with less cases don't display data for specific countries, 
then we should do an overall analysis depending only on the countries itself, instead on the cities. 
As we can see, last column belongs to the last day measured.see:
As we can see, countries with less cases don't display data for specific countries, 
then we should do an overall analysis depending only on the countries itself, instead on the cities.
As we can see, last column belongs to the last day measured. see:
"""

countries_confirmed = confirmed.groupby(['Country/Region']).sum().iloc[:,-1]
countries_recovered = recovered.groupby(['Country/Region']).sum().iloc[:,-1]
countries_deaths = deaths.groupby(['Country/Region']).sum().iloc[:,-1]

countries = []
numbers = []
china = 0
for index in countries_deaths.index:
    if index in ['Hong Kong', 'Macau', 'Others', 'Mainland China']:
        china+=countries_deaths[index]
    else:
        countries.append(index)
        numbers.append(countries_deaths[index])

countries.append('China')
numbers.append(china)

new_cases = pd.DataFrame({
    'name': countries,
    'Number of deaths': numbers
    })
ggplot() + geom_livemap(data=new_cases, mapping=aes(map_id='name',fill='Number of deaths'), level='country') + scale_fill_gradient(low = "#F0A999", high = "#ED2A00")
