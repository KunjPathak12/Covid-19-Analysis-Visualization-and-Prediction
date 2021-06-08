#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import numpy as np
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import cufflinks as cf
import folium
import warnings
warnings.filterwarnings('ignore')


# In[61]:


pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[62]:


pd.options.display.float_format = '{:.2f}'.format


# In[63]:


pd.set_option('display.max_rows',None)


# # Importing my dataset

# In[64]:


df= pd.read_csv(r'/home/blunder-master/Downloads/archive/covid_19_data.csv')


# In[65]:


df.head()


# In[66]:


df.shape


# In[67]:


df.tail()


# In[ ]:





# # Data Analysis

# In[68]:


df.info()


# In[69]:


df.isnull().sum()


# In[70]:


df.describe()


# In[71]:


Confirmed=df.groupby('ObservationDate').sum()['Confirmed'].reset_index()


# In[72]:


Confirmed


# In[73]:


Deaths=df.groupby('ObservationDate').sum()['Deaths'].reset_index()


# In[74]:


Deaths


# In[75]:


Recovered=df.groupby('ObservationDate').sum()['Recovered'].reset_index()


# In[76]:


Recovered


# # Plotted through Pandas below

# In[77]:


Confirmed.plot(kind='bar',x='ObservationDate',y='Confirmed',color='red')


# # Plotted with Seaborn below

# In[78]:


sns.barplot(x=Confirmed['ObservationDate'],y=Confirmed['Confirmed'])


# # Plotted through Matplotlib below

# In[79]:


plt.bar(x=Confirmed['ObservationDate'],height=Confirmed['Confirmed'])


# # Plotted through Plotly below for number of Confirmed Cases

# In[80]:


Confirmed.iplot(kind='bar',x='ObservationDate',y='Confirmed')


# # PLotted through Plotly below for  Number of Deaths

# In[81]:


Deaths.iplot(kind='bar',x='ObservationDate',y='Deaths',color='red')


# # PLotted through Plotly below for  Number of Recovred Cases

# In[82]:


Recovered.iplot(kind='bar',x='ObservationDate',y='Recovered',color='green')


# In[ ]:





# In[ ]:





# # Prediction

# In[83]:


from prophet import Prophet


# In[84]:


df.head()


# In[85]:


Confirmed.head()


# In[86]:


Deaths.head()


# In[87]:


Recovered.head()


# In[88]:


Confirmed.tail()


# In[89]:


Confirmed.head()


# In[90]:


Deaths.tail()


# In[91]:


Recovered.tail()


# In[ ]:





# In[92]:


# Date=ds(datestamp, column= y)
# prophet takes dates in yyyy-mm-dd and hh:mm:ss
# so converting in first


# In[ ]:





# In[ ]:





# # Prediction for Confirmed Cases Worldwide

# In[93]:


Confirmed.columns=['ds','y']
Confirmed['ds']=pd.to_datetime(Confirmed['ds'])
Confirmed


# In[ ]:





# In[94]:


predictModel=Prophet(interval_width=0.96)
predictModel.fit(Confirmed)


# In[95]:


future=predictModel.make_future_dataframe(periods=300)
future


# In[96]:


future.tail(30)


# In[97]:


forecastConfirm=predictModel.predict(future)


# In[98]:


forecastConfirm


# In[99]:


forecastConfirm[['ds','yhat','yhat_lower','yhat_upper']]


# In[100]:


forecastConfirm[['ds','yhat','yhat_lower','yhat_upper']].tail(300)


# In[101]:


forecastConfirm.iplot(kind='bar',x='ds',y='yhat',color='purple')


# In[102]:


forecastConfirm.iplot(mode='markers+text',kind='scatter',x='ds',y='yhat_upper',color='purple')


# # Visualization for Confirmed Cases Worldwide

# In[ ]:





# In[ ]:





# # Prediction for Deathtoll Worldwide

# In[103]:


Deaths.columns=['ds','y']
Deaths['ds']=pd.to_datetime(Deaths['ds'])
Deaths


# In[104]:


predictModel=Prophet(interval_width=0.96)
predictModel.fit(Deaths)


# In[105]:


forecastDeaths=predictModel.predict(future)
forecastDeaths


# In[106]:


forecastDeaths[['ds','yhat','yhat_lower','yhat_upper']]


# In[107]:


forecastDeaths[['ds','yhat','yhat_lower','yhat_upper']].tail(300)


# In[108]:


forecastDeaths.iplot(kind='bar',x='ds',y='yhat_upper',color='red')


# In[109]:


forecastDeaths.iplot(mode='markers+text',kind='scatter',x='ds',y='yhat_upper',color='red')


# # Visualization for Deathtoll Worldwide

# In[ ]:





# In[ ]:





# # Prediction for Recovered Cases Worlwide

# In[110]:


Recovered.columns=['ds','y']
Recovered['ds']=pd.to_datetime(Recovered['ds'])
Recovered


# In[111]:


predictModel=Prophet(interval_width=0.96)
predictModel.fit(Recovered)


# In[112]:


future=predictModel.make_future_dataframe(periods=300)
future


# In[113]:


Recovered.tail(300)


# In[114]:


forecastRecovered=predictModel.predict(future)
forecastRecovered


# In[115]:


forecastRecovered[['ds','yhat','yhat_lower','yhat_upper']]


# In[116]:


forecastRecovered[['ds','yhat','yhat_lower','yhat_upper']].tail(300)


# In[117]:


forecastRecovered.iplot(kind='bar',x='ds',y='yhat_upper',color='green')


# In[118]:


forecastRecovered.iplot(mode='markers+text',kind='scatter',x='ds',y='yhat_upper',color='green')


# # Visualization for Recovered Cases Worldwide

# In[ ]:





# In[ ]:





# In[ ]:




