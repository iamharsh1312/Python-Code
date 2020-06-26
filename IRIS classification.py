#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn


# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[3]:


iris.keys()


# # DESCRIPTION OF DATASET

# In[4]:


print(iris['DESCR'])


# In[5]:


iris['target_names']


# In[6]:


iris['feature_names']


# In[8]:


type(iris['data'])


# In[9]:


iris['data'].shape


# In[10]:


iris['data'][:5]


# In[11]:


iris['target']


# # Training and testing data

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'],iris['target'], random_state = 0)


# In[15]:


X_train.shape


# In[16]:


X_test.shape


# In[17]:


X_test


# In[18]:


y_train.shape


# In[19]:


y_train


# In[20]:


y_test.shape


# # Nearest K neighbor

# In[22]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[23]:


knn.fit(X_train, y_train)


# In[25]:


import numpy as np
X_new = np.array([[5, 2.9, 1, 0.2]])
X_new.shape


# In[26]:


prediction = knn.predict(X_new)
prediction


# In[27]:


iris['target_names'][prediction]


# # Evaluating the model

# In[28]:


knn.score(X_test, y_test)


# In[ ]:




