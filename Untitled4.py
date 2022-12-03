#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 


# In[2]:


import matplotlib.pyplot as plt


# In[5]:


x = np.linspace(-5.0 , 5.0 , 100)
y = np.sqrt(10**2 - x**2)
y = np.hstack([y , -y])
x = np.hstack([x , -x])


# In[15]:


x1 = np.linspace(-5.0 , 5.0 , 100)
y1 = np.sqrt(5**2 - x1**2)
x1 = np.hstack([x1 , -x1])
y1 = np.hstack([y1 , -y1])


# In[16]:


plt.scatter(y , x)
plt.scatter(y1 , x1)


# In[40]:


import pandas as pd 
df1 = pd.DataFrame(np.vstack([y , x]).T , columns = ['X1' , 'X2'])
df1['Y'] = 0

df2 = pd.DataFrame(np.vstack([y1 , x1]).T , columns = ['X1' , 'X2'])
df2['Y'] = 1
df = df1.append(df2)
df.head(5)


# In[22]:


###Defining dependent and independent features 

X = df.iloc[: , :2]
y = df.Y


# In[23]:


y


# In[24]:


###Splitting the data 
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)


# In[25]:


y_train


# In[27]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train , y_train)


# In[28]:


from sklearn.metrics import accuracy_score


# In[30]:


y_pred= classifier.predict(X_test)


# In[31]:


accuracy_score(y_test , y_pred)


# In[32]:



###THATS LOW 


# In[33]:


###NOW WE do polynomial kernel 


# In[34]:


##WE need to find some components first , which will be added to change the dimension


# In[36]:


df['X1_square'] = df['X1']**2

df['X2_square'] = df['X2']**2

df['X1*X2'] = df['X1'] * df['X2']


# In[37]:


df.head(10)


# In[41]:


X = df[['X1' , 'X2'  , 'X1_square' , 'X2_square' , 'X1*X2']]
Y = df[['Y']]


# In[42]:


Y


# In[43]:


X_train, X_test , y_train , y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 0)


# In[44]:


##WE have now included some extra features as out input parameters 
##NOw we would run our linear regression model , and find a perfect hyperplane


# In[50]:


classifier = SVC(kernel = 'linear')
classifier.fit(X_train , y_train)


# In[53]:


y_pred = classifier.predict(X_test)


# In[54]:


accuracy_score(y_test , y_pred)


# In[ ]:




