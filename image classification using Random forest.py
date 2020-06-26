#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Image classification using RandomForest: An example in Python using CIFAR10 Dataset

def Snippet_349(): 

    print()
    print(format('Image classification using RandomForest: An example in Python using CIFAR10 Dataset','*^88'))


# In[2]:


import warnings
warnings.filterwarnings("ignore")    


# In[3]:


# load libraries
from keras.datasets import cifar10
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score        


# In[2]:


import time
start_time = time.time()


# In[6]:


# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # X_train is 50000 rows of 3x32x32 values --> reshaped in 50000 x 3072
RESHAPED = 3072


# In[7]:


X_train = X_train.reshape(50000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[ ]:


y_train = y_train.flatten()
y_test = y_test.flatten()


# In[ ]:


# normalize the datasets
X_train /= 255.
X_test /= 255.
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[ ]:


# fit a RandomForest model to the data
model = RandomForestClassifier(n_estimators = 10)

cv_results = cross_val_score(model, X_train, y_train, 
                   cv = 2, scoring='accuracy', n_jobs = -1, verbose = 1)
model.fit(X_train, y_train)
print(); print(cv_results)    
print(); print(model)


# In[ ]:


# make predictions
expected_y  = y_test
predicted_y = model.predict(X_test)


# In[ ]:


# summarize the fit of the model
print(); print(metrics.classification_report(expected_y, predicted_y))
print(); print(metrics.confusion_matrix(expected_y, predicted_y))
print()
print("Execution Time %s seconds: " % (time.time() - start_time))    

Snippet_349()


# In[ ]:




