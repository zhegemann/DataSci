#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
dfBase = pd.read_csv("ADNI1_baseline_info.csv") ## Reads data in from the data set
df3T = pd.read_csv("ADNI1_3T_ROI.csv") ## Reads data in from the data set
numBaseRIDs = dfBase.RID.nunique() ## Finds the number of unique RID's in dfBase
num3TRIDs = df3T.RID.nunique() ## Finds the number of unique RID's in df3T
print("Number of participants in ADNI1_baseline_info.csv: " +  str(numBaseRIDs))
print("Number of participants in ADNI1_3T_ROI.csv: " +  str(num3TRIDs))


# In[2]:


## Counts the number of elements in the intersection of the two RID columns in each CSV
len(set(dfBase['RID']) & set(df3T['RID']))


# In[3]:


## Merges the two dataframes based on the intersection of the RID column of each respective dataset
df = pd.merge(dfBase, df3T, 
                   on='RID', 
                   how='inner')


# In[4]:


## Finds the duplicates in the ADNI data set and prints them out 
print(df.duplicated().sum())
## We see there are no duplicated data in our set


# In[5]:


## Check the dataset for missing values (NaN) in columns
## and then report how many missing values per column

# Shows the number of columns with missing values, lists those columns, and then how many missing values per column
print(len(df.columns[df.isnull().any()]))
print(df.columns[df.isnull().any()])
df.isnull().sum()


# In[77]:


from sklearn.linear_model import LinearRegression
X = df.drop(columns=['RID', 'VISCODE', 'DX.bl', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY', 'APOE4', 'SITEID', 'EICV'])
Y = df['AGE']
Z = df.drop(columns=['RID', 'VISCODE', 'DX.bl', 'PTGENDER', 'PTEDUCAT', 'PTMARRY', 'APOE4', 'SITEID', 'EICV'])
x = np.array(X)
y = np.array(Y)
z = np.array(Z)
## Creates a multilinear regression model to predict ages
modelMultLinReg = LinearRegression().fit(x,y)


# In[7]:


## Generates results for our model
r_sq = modelMultLinReg.score(x,y)
print('Coefficient of determination: ', r_sq)
print('Intercept: ', modelMultLinReg.intercept_)
print('Slope: ', modelMultLinReg.coef_)


# In[8]:


# Predicts results based on x - feature data array
age_pred = modelMultLinReg.predict(x)

Z = Y - age_pred
from numpy import linalg as LA
print('The largest discrepency of predicted age to actual age is: ', LA.norm(Z, np.inf))


# In[9]:


from sklearn.preprocessing import PolynomialFeatures


# In[10]:


# Performs multi-polynomial regression on our dataset. Degree n = 5 yields the highest model score
poly = PolynomialFeatures(degree = 5)
poly_x = poly.fit_transform(x)
modelMultiPolyReg = LinearRegression().fit(poly_x,y)
r_sq2 = modelMultiPolyReg.score(poly_x,y)
print('Coefficient of determination: ', r_sq2)
print('Intercept: ', modelMultiPolyReg.intercept_)
print('Slope: ', modelMultiPolyReg.coef_)


# In[11]:


# Predicts results based on poly_x - feature data array for multi-polynomial regression
age_pred2 = modelMultiPolyReg.predict(poly_x)

Z2 = Y - age_pred2
print('The largest discrepency of predicted age to actual age is: ', LA.norm(Z2, np.inf))


# In[72]:


# We now perform PCA and SVD on our dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
scaler = StandardScaler()
xPCA = scaler.fit_transform(x)
PCA
pca = PCA()
pca.fit_transform(xPCA)
pca_variance = pca.explained_variance_
plt.figure(figsize = (8,6))
plt.bar(range(13), pca_variance, alpha = 0.5, align = 'center', label = 'individual variance')
plt.legend()
plt.ylabel('Variance Rate')
plt.xlabel('Principal Components')
plt.show()
# It looks as if 6 features explain the majority of our data


## Plots the first 6 figures
pca2 = PCA(n_components = 8)
pca2.fit(xPCA)
x_3d = pca2.transform(xPCA)
plt.figure(figsize = (8,6))
plt.scatter(x_3d[:,0], x_3d[:,5], c=df['AGE'])
plt.show()

## Plots the first 3 figures
pca3 = PCA(n_components = 3)
pca3.fit(xPCA)
x_3d = pca3.transform(xPCA)
plt.figure(figsize = (8,6))
plt.scatter(x_3d[:,0], x_3d[:,2], c=df['AGE'])
plt.show()

# x = scaler.fit_transform(x)
# PCA
# pca = PCA()
# pca.fit_transform(x)
# pca_variance = pca.explained_variance_
# plt.figure(figsize = (8,6))
# plt.bar(range(13), pca_variance, alpha = 0.5, align = 'center', label = 'individual variance')
# plt.legend()
# plt.ylabel('Variance Rate')
# plt.xlabel('Principal Components')
# plt.show()
# # It looks as if 6 features explain the majority of our data


# ## Plots the first 6 figures
# pca2 = PCA(n_components = 8)
# pca2.fit(x)
# x_3d = pca2.transform(x)
# plt.figure(figsize = (8,6))
# plt.scatter(x_3d[:,0], x_3d[:,5], c=df['AGE'])
# plt.show()

# ## Plots the first 3 figures
# pca3 = PCA(n_components = 3)
# pca3.fit(x)
# x_3d = pca3.transform(x)
# plt.figure(figsize = (8,6))
# plt.scatter(x_3d[:,0], x_3d[:,2], c=df['AGE'])
# plt.show()


# In[68]:


# Conduct SVD analysis
u,s,v = LA.svd(x)
print(s)


# In[16]:


plt.figure(figsize = (8,6))
plt.bar(range(13), s, alpha = 0.5, align = 'center', label = 'Singular Values')
plt.legend()
plt.ylabel('Sigma Value')
plt.xlabel('Singular Value')
plt.show()


# In[61]:


# Perform K-mean analysis
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init = 10, random_state = 0)
    kmeans.fit(z)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#This shows the optimial number of clusters is 4-6


# In[73]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(z)

plt.scatter(z[y_kmeans==0,0], z[y_kmeans==0,1], s=10, c='red', label='1')
plt.scatter(z[y_kmeans==1,0], z[y_kmeans==1,1], s=10, c='blue', label='2')
plt.scatter(z[y_kmeans==2,0], z[y_kmeans==2,1], s=10, c='green', label='3')
plt.scatter(z[y_kmeans==3,0], z[y_kmeans==3,1], s=10, c='cyan', label='4')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroid')
plt.title('Cluster of Study Participants')
plt.xlabel('Age')
plt.ylabel('Brain Dimensions')


# In[104]:


# # Conduct the NMF Analysis
# print(x.shape)
# from sklearn.decomposition import NMF
# nmf = NMF(n_components = 10)
# nmf.fit(x)


# In[158]:


# Attemptes to predict the last 234 participant DX.bl status using the first 500 as a training model
from sklearn.ensemble import RandomForestClassifier

y3 = df["DX.bl"]
yDX = y3.head(500)
yTail = y3.tail(234)

features = ["PTGENDER","AGE", "PTEDUCAT", "PTMARRY", "APOE4"]
X3 = pd.get_dummies(df[features])
XTr = X3.head(500)
xTail = X3.tail(234)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=1)
model.fit(XTr, yDX)
predictions = model.predict(xTail)

result = pd.DataFrame({'RID': df.RID.tail(234), 'DX.bl Predicted': predictions})
comparison_column = np.where(df['DX.bl'].tail(234) == result['DX.bl Predicted'], True, False)

print("Our accuracy for a Random Forest is:", np.count_nonzero(comparison_column) / len(comparison_column))


# In[168]:


#XG Boost
import sklearn
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
import xgboost as xgb


# In[ ]:




