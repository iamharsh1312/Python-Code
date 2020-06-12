#!/usr/bin/env python
# coding: utf-8

# In[2]:


list1=[12,'a',6,7]
list1.append('harsh')
print(list1)


# In[3]:


list1.insert(2,'My captain')
print(list1)


# In[4]:


list1.extend([34,'python',99])
print(list1)


# In[6]:


tup=('harsh',54,'AI',78,1)
print(tup[3])


# In[10]:


tup1=('a','A','b')
tup2=tup+tup1
print(tup2)


# In[32]:


Dict = {'a':1,'b':2,'c':3,'d':4,'e':5}
del Dict['a']
print(Dict)


# In[34]:


del Dict['d']
print(Dict)

