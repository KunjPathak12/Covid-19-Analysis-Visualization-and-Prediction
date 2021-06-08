import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import numpy as np
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import cufflinks as cf
import folium
import warnings
warnings.filterwarnings('ignore')

pyo.init_notebook_mode(connected=True)
cf.go_offline()

pd.options.display.float_format = '{:.2f}'.format

pd.set_option('display.max_rows',None)

# Importing my dataset

df= pd.read_csv(r'/home/blunder-master/Downloads/archive/covid_19_data.csv')

df.head()

df.shape

df.tail()



# Data Analysis

df.info()

df.isnull().sum()

df.describe()

Confirmed=df.groupby('ObservationDate').sum()['Confirmed'].reset_index()

Confirmed

Deaths=df.groupby('ObservationDate').sum()['Deaths'].reset_index()

Deaths

Recovered=df.groupby('ObservationDate').sum()['Recovered'].reset_index()

Recovered

# Plotted through Pandas below

Confirmed.plot(kind='bar',x='ObservationDate',y='Confirmed',color='red')

# Plotted with Seaborn below

sns.barplot(x=Confirmed['ObservationDate'],y=Confirmed['Confirmed'])

# Plotted through Matplotlib below

plt.bar(x=Confirmed['ObservationDate'],height=Confirmed['Confirmed'])

# Plotted through Plotly below for number of Confirmed Cases

Confirmed.iplot(kind='bar',x='ObservationDate',y='Confirmed')

# PLotted through Plotly below for  Number of Deaths

Deaths.iplot(kind='bar',x='ObservationDate',y='Deaths',color='red')

# PLotted through Plotly below for  Number of Recovred Cases

Recovered.iplot(kind='bar',x='ObservationDate',y='Recovered',color='green')

# Prediction

from prophet import Prophet

df.head()

Confirmed.head()

Deaths.head()

Recovered.head()

Confirmed.tail()

Confirmed.head()

Deaths.tail()

Recovered.tail()



# Date=ds(datestamp, column= y)
# prophet takes dates in yyyy-mm-dd and hh:mm:ss
# so converting in first





# Prediction for Confirmed Cases Worldwide

Confirmed.columns=['ds','y']
Confirmed['ds']=pd.to_datetime(Confirmed['ds'])
Confirmed



predictModel=Prophet(interval_width=0.96)
predictModel.fit(Confirmed)

future=predictModel.make_future_dataframe(periods=300)
future

future.tail(30)

forecastConfirm=predictModel.predict(future)

forecastConfirm

forecastConfirm[['ds','yhat','yhat_lower','yhat_upper']]

forecastConfirm[['ds','yhat','yhat_lower','yhat_upper']].tail(300)

forecastConfirm.iplot(kind='bar',x='ds',y='yhat',color='purple')

# Visualization for Confirmed Cases Worldwide above





# Prediction for Deathtoll Worldwide

Deaths.columns=['ds','y']
Deaths['ds']=pd.to_datetime(Deaths['ds'])
Deaths

predictModel=Prophet(interval_width=0.96)
predictModel.fit(Deaths)

forecastDeaths=predictModel.predict(future)
forecastDeaths

forecastDeaths[['ds','yhat','yhat_lower','yhat_upper']]

forecastDeaths[['ds','yhat','yhat_lower','yhat_upper']].tail(300)

forecastDeaths.iplot(kind='bar',x='ds',y='yhat_upper',color='red')

# Visualization for Deathtoll Worldwide above





# Prediction for Recovered Cases Worlwide

Recovered.columns=['ds','y']
Recovered['ds']=pd.to_datetime(Recovered['ds'])
Recovered

predictModel=Prophet(interval_width=0.96)
predictModel.fit(Recovered)

future=predictModel.make_future_dataframe(periods=300)
future

Recovered.tail(300)

forecastRecovered=predictModel.predict(future)
forecastRecovered

forecastRecovered[['ds','yhat','yhat_lower','yhat_upper']]

forecastRecovered[['ds','yhat','yhat_lower','yhat_upper']].tail(300)

forecastRecovered.iplot(kind='bar',x='ds',y='yhat_upper',color='green')

# Visualization for Recovered Cases Worldwide above






