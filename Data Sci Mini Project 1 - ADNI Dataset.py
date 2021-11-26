#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
df = pd.read_csv("ADNI-DATA.csv") ## Reads data in from the data set
numRIDs = df.RID.nunique() ## Finds the number of unique RID's in the second column of our dataset
print(numRIDs)


# In[8]:


## Find the max and min frequency of RID's to find max and min observations
print(df['RID'].value_counts().max())
print(df['RID'].value_counts().min())


# In[9]:


## Covariates: Identification information should not affect results so they are categorically not covariates. 
## Site ID could be argued to be a covariate as procedure may vary between test sites.
## The covariates are all information aside from the identification (patient) information


# In[10]:


## Prints the marital status of the first 10 participants
print(df.PTMARRY.head(10))


# In[11]:


## Finds the duplicates in the ADNI data set and prints them out 
print(df.duplicated().sum())
## We see there are no duplicated data in our set


# In[12]:


## Check the dataset for missing values (NaN) in columns
## and then report how many missing values per column

# Shows the number of columns with missing values, lists those columns, and then how many missing values per column
print(len(df.columns[df.isnull().any()]))
print(df.columns[df.isnull().any()])
df.isnull().sum()


# In[13]:


## We must handle the empty data in each column differently
## Throw out the two participants with no baseline as we have nothing to compare to
df.dropna(subset = ['DX.bl'], inplace=True)


# In[14]:


## We cannot assume the presence of the epsilon 4 allele of APOE in our study so we default these values to zero. 
## This may artificially increase the likelihood of AD in the general population
df['APOE4'].fillna(0, inplace = True)


# In[15]:


## We replace numerical values with the mean of the respective columns
df['ADAS11'].fillna((df['ADAS11'].mean()), inplace = True)
df['ADAS13'].fillna((df['ADAS13'].mean()), inplace = True)
df['MMSE'].fillna((df['MMSE'].mean()), inplace = True)
df['RAVLT.immediate'].fillna((df['RAVLT.immediate'].mean()), inplace = True)
df['RAVLT.learning'].fillna((df['RAVLT.learning'].mean()), inplace = True)
df['RAVLT.forgetting'].fillna((df['RAVLT.forgetting'].mean()), inplace = True)
df['RAVLT.perc.forgetting'].fillna((df['RAVLT.perc.forgetting'].mean()), inplace = True)
df['FAQ'].fillna((df['FAQ'].mean()), inplace = True)
df['Ventricles'].fillna((df['Ventricles'].mean()), inplace = True)
df['Hippocampus'].fillna((df['Hippocampus'].mean()), inplace = True)
df['WholeBrain'].fillna((df['WholeBrain'].mean()), inplace = True)
df['Entorhinal'].fillna((df['Entorhinal'].mean()), inplace = True)
df['Fusiform'].fillna((df['Fusiform'].mean()), inplace = True)
df['MidTemp'].fillna((df['MidTemp'].mean()), inplace = True)
df['ICV'].fillna((df['ICV'].mean()), inplace = True)

## Pros: No loss of data from deleting all rows with missing values
## Cons: Can cause data leakage and does not factor covariance between variables

df.isnull().sum()


# In[16]:


## Display the covariance table
df.corr()
##df[['DX.bl','APOE4', 'FDG', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT.immediate', 'RAVLT.learning', 'RAVLT.forgetting', 'RAVLT.perc.forgetting', 'FAQ', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']].corr()


# In[46]:


import matplotlib.pyplot as plt
import numpy as np
# Change the format of EXAMDATE for future purposes
df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])

## We strip the numbers from VISCODE - bl is time=0
df['VISCODE'] = df['VISCODE'].str.extract('(\d+)')
df['VISCODE'].fillna(0, inplace = True)

## Split the VISCODE and clinical data columns into subarrays separating each participant
viscodes = []
adas11s = []
adas13s = []
mmses = []
ravltIms = []
ravltLearn = []
ravltForg = []
ravltPercs = []
faqs = []
examdates = []
hippocampuses = []
icvs = []

q = []
r = []
s = []
t = []
u = []
v = []
w = []
x = []
y = []
z = []
a = []
b = []
for h, i, j, k, l, m, n, o, p, g, f, e in zip(df['VISCODE'], df['ADAS11'], df['ADAS13'], df['MMSE'], df['RAVLT.immediate'], df['RAVLT.learning'], df['RAVLT.forgetting'], df['RAVLT.perc.forgetting'], df['FAQ'], df['EXAMDATE'], df['Hippocampus'], df['ICV']):
    if h == 0 and q:
        dummy1 = q.copy()
        dummy2 = r.copy()
        dummy3 = s.copy()
        dummy4 = t.copy()
        dummy5 = u.copy()
        dummy6 = v.copy()
        dummy7 = w.copy()
        dummy8 = x.copy()
        dummy9 = y.copy()
        dummy10 = z.copy()
        dummy11 = a.copy()
        dummy12 = b.copy()
        viscodes.append(dummy1)
        adas11s.append(dummy2)
        adas13s.append(dummy3)
        mmses.append(dummy4)
        ravltIms.append(dummy5)
        ravltLearn.append(dummy6)
        ravltForg.append(dummy7)
        ravltPercs.append(dummy8)
        faqs.append(dummy9)
        examdates.append(dummy10)
        hippocampuses.append(dummy11)
        icvs.append(dummy12)
        q.clear()
        r.clear()
        s.clear()
        t.clear()
        u.clear()
        v.clear()
        w.clear()
        x.clear()
        y.clear()
        z.clear()
        a.clear()
        b.clear()
    if not q:
        q.append(h)
        r.append(i)
        s.append(j)
        t.append(k)
        u.append(l)
        v.append(m)
        w.append(n)
        x.append(o)
        y.append(p)
        z.append(g)
        a.append(f)
        b.append(e)
    if q and i != 0:
        q.append(h)
        r.append(i)
        s.append(j)
        t.append(k)
        u.append(l)
        v.append(m)
        w.append(n)
        x.append(o)
        y.append(p)
        z.append(g)
        a.append(f)
        b.append(e)
numArr = len(viscodes)


## Plots the clinical information vs the viscodes
for i in range(1, numArr):
    plt.plot(viscodes[i], adas11s[i], linewidth = '.15')

plt.xlabel('Time (Months)')
plt.ylabel('ADAS11')
plt.title('ADAS11 Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(viscodes[i], adas13s[i], linewidth = '.15')

plt.xlabel('Time (Months)')
plt.ylabel('ADAS13')
plt.title('ADAS13 Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(viscodes[i], mmses[i], linewidth = '.15')

plt.xlabel('Time (Months)')
plt.ylabel('MMSE')
plt.title('MMSE Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(viscodes[i], ravltIms[i], linewidth = '.15')

plt.xlabel('Time (Months)')
plt.ylabel('RAVLT.immediate')
plt.title('RAVLT.immediate Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(viscodes[i], ravltLearn[i], linewidth = '.15')

plt.xlabel('Time (Months)')
plt.ylabel('RAVLT.learning')
plt.title('RAVLT.learning Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(viscodes[i], ravltForg[i], linewidth = '.15')

plt.xlabel('Time (Months)')
plt.ylabel('RAVLT.forgetting')
plt.title('RAVLT.forgetting Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(viscodes[i], ravltPercs[i], linewidth = '.15')

plt.xlabel('Time (Months)')
plt.ylabel('RAVLT.perc.forgetting')
plt.title('RAVLT.perc.forgetting Progression')
plt.show() 

for i in range(1, numArr):
     plt.plot(viscodes[i], faqs[i], linewidth = '.15')

plt.xlabel('Time (Months)')
plt.ylabel('FAQ')
plt.title('FAQ Progression')
plt.show()


# In[44]:


## Plots the clinical information vs the exam dates
for i in range(1, numArr):
     plt.plot(examdates[i], adas11s[i], linewidth = '.15')

plt.xlabel('Date')
plt.ylabel('ADAS11')
plt.title('ADAS11 Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(examdates[i], adas13s[i], linewidth = '.15')

plt.xlabel('Date')
plt.ylabel('ADAS13')
plt.title('ADAS13 Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(examdates[i], mmses[i], linewidth = '.15')

plt.xlabel('Date')
plt.ylabel('MMSE')
plt.title('MMSE Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(examdates[i], ravltIms[i], linewidth = '.15')

plt.xlabel('Date')
plt.ylabel('RAVLT.immediate')
plt.title('RAVLT.immediate Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(examdates[i], ravltLearn[i], linewidth = '.15')

plt.xlabel('Date')
plt.ylabel('RAVLT.learning')
plt.title('RAVLT.learning Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(examdates[i], ravltForg[i], linewidth = '.15')

plt.xlabel('Date')
plt.ylabel('RAVLT.forgetting')
plt.title('RAVLT.forgetting Progression')
plt.show()

for i in range(1, numArr):
     plt.plot(examdates[i], ravltPercs[i], linewidth = '.15')

plt.xlabel('Date')
plt.ylabel('RAVLT.perc.forgetting')
plt.title('RAVLT.perc.forgetting Progression')
plt.show() 

for i in range(1, numArr):
     plt.plot(examdates[i], faqs[i], linewidth = '.15')

plt.xlabel('Date')
plt.ylabel('FAQ')
plt.title('FAQ Progression')
plt.show()


# In[48]:


# Plots the hippocampus vs the exam dates
for i in range(1, numArr):
     plt.plot(examdates[i], hippocampuses[i], linewidth = '.15')

plt.xlabel('Date')
plt.ylabel('Hippocampus Size')
plt.title('Hippocampus Progression')
plt.show()


# In[59]:


## Plots the clinical information vs hippocampus size
for i in range(1, numArr):
     plt.scatter(adas11s[i], hippocampuses[i], 0.1)

plt.xlabel('ADAS11')
plt.ylabel('Hippocampus Size')
plt.title('ADAS11 v Hippocampus Size')
plt.show()

for i in range(1, numArr):
     plt.scatter(adas13s[i], hippocampuses[i], 0.1)

plt.xlabel('ADAS13')
plt.ylabel('Hippocampus Size')
plt.title('ADAS13 vs Hippocampus Size')
plt.show()

for i in range(1, numArr):
     plt.scatter(mmses[i], hippocampuses[i], 0.1)

plt.xlabel('MMSE')
plt.ylabel('Hippocampus Size')
plt.title('MMSE vs Hippocampus Size')
plt.show()

for i in range(1, numArr):
     plt.scatter(ravltIms[i], hippocampuses[i], 0.1)

plt.xlabel('RAVLT.immediate')
plt.ylabel('Hippocampus')
plt.title('RAVLT.immediate vs Hippocampus Size')
plt.show()

for i in range(1, numArr):
     plt.scatter(ravltLearn[i], hippocampuses[i], 0.1)

plt.xlabel('RAVLT.learning')
plt.ylabel('Hippocampus Size')
plt.title('RAVLT.learning vs Hippocampus Size')
plt.show()

for i in range(1, numArr):
     plt.scatter(ravltForg[i], hippocampuses[i], 0.1)

plt.xlabel('RAVLT.forgetting')
plt.ylabel('Hippocampus Size')
plt.title('RAVLT.forgetting vs Hippocampus Size')
plt.show()

for i in range(1, numArr):
     plt.scatter(ravltPercs[i], hippocampuses[i], 0.1)

plt.xlabel('RAVLT.perc.forgetting')
plt.ylabel('Hippocampus Size')
plt.title('RAVLT.perc.forgetting vs Hippocampus Size')
plt.show() 

for i in range(1, numArr):
     plt.scatter(faqs[i], hippocampuses[i], 0.1)

plt.xlabel('FAQ')
plt.ylabel('Hippocampus Size')
plt.title('FAQ Progression')
plt.show()


# In[ ]:




