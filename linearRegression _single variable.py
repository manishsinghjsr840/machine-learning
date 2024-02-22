#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model as lm


# In[4]:


df=pd.read_csv("canada_per_capita_income.csv")


# In[13]:


df


# In[10]:


plt.scatter(df.year,df.per_capita_income(US$),color='red',marker='+')


# In[21]:


df=df.drop("per capita income (US$)",axis='columns')


# In[22]:


df


# In[34]:


new_df=df.drop('income',axis='columns')


# In[35]:





# In[29]:


plt.scatter(df.year,df.income,color='red',marker='+')
plt.xlabel('year')
plt.ylabel('income')
plt.title('per capital income of canada')


# In[36]:


model=lm.LinearRegression()
model.fit(new_df,df.income)


# In[38]:


model.predict([[2020]])


# In[39]:


model.coef_


# In[41]:


model.intercept_


# In[ ]:




