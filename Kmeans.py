#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os #provides functions for interacting with the operating system
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from math import sqrt

from sklearn.cluster import KMeans, k_means


# In[2]:


raw_data=pd.read_csv('Desktop/Full_cus_add.csv')


# In[4]:


print(raw_data.shape)
raw_data.head()


# In[5]:


# Checking for null values

raw_data.isnull().sum()


# In[6]:


# Visualize the NULL observations


raw_data[raw_data['last_name'].isnull()]


# In[7]:


# Deleting the NULL values
raw_data = raw_data.dropna(subset = ['last_name'])

# Printing the shape
print(raw_data.shape)

# Visualize the NULL observations
raw_data.isnull().sum()


# In[10]:


raw_data = raw_data.dropna(subset = ['job_title'])
raw_data = raw_data.dropna(subset = ['job_industry_category'])
raw_data.isnull().sum()


# In[11]:


raw_data = raw_data.dropna(subset = ['age'])
raw_data = raw_data.dropna(subset = ['Age group'])
raw_data.isnull().sum()


# In[12]:


# Investigate all the elements whithin each Feature 

for column in raw_data:
    unique_vals = np.unique(raw_data[column])
    nr_values = len(unique_vals)
    if nr_values < 10:
        print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column, nr_values))


# In[13]:


raw_data.columns


# In[17]:



features = ['Gender',
       'job_title',
       'job_industry_category', 'wealth_segment', 'state',
       'property_valuation', 'Age group']

for f in features:
    sns.countplot(x = f, data = raw_data, palette = 'Set3')# hue = 'Good Loan')
    plt.xticks(rotation=45)
    plt.show()


# In[18]:


# Making categorical variables into numeric representation

print(raw_data.shape)

# keeping the columns we need - Drop the location columns for now, as we do not want them to impact our results (for now)
raw_data_1= raw_data[features]
print(raw_data_1.shape)

# Making categorical variables into numeric representation
new_raw_data = pd.get_dummies(raw_data_1, columns = features)

# Notes:
# We can also do this with Label Encoding and OneHotEncoder from the preprocessing library

print(new_raw_data.shape)
# print the shape

new_raw_data.head()


# In[19]:


# Running Kmeans

X_train = new_raw_data.values

# We wills start with 5 clusters

kmeans = KMeans(n_clusters=5, random_state=540)
kmeans = kmeans.fit(X_train)

# Prints the clusters it assigned to each observation
print("The clusters are: ", kmeans.labels_)

# Prints the Inertia
print("The Inertia is: ", kmeans.inertia_)


# Inertia is the sum of squared error for each cluster. Therefore the smaller the inertia the denser the cluster(closer together all the points are) 

# In[20]:


# How to find the best number if Ks?

# Running K means with multible Ks

no_of_clusters = range(2,20) #[2,3,4,5,6,7,8,9]
inertia = []


for f in no_of_clusters:
    kmeans = KMeans(n_clusters=f, random_state=2)
    kmeans = kmeans.fit(X_train)
    u = kmeans.inertia_
    inertia.append(u)
    print("The innertia for :", f, "Clusters is:", u)


# In[21]:


fig, (ax1) = plt.subplots(1, figsize=(16,6))
xx = np.arange(len(no_of_clusters))
ax1.plot(xx, inertia)
ax1.set_xticks(xx)
ax1.set_xticklabels(no_of_clusters, rotation='vertical')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia Score')
plt.title("Inertia Plot per k")


# In[23]:


# Running K means on 4 clusters

kmeans = KMeans(n_clusters=4, random_state=2)
kmeans = kmeans.fit(X_train)


kmeans.labels_

# "predictions" for new data
predictions = kmeans.predict(X_train)

# calculating the Counts of the cluster
unique, counts = np.unique(predictions, return_counts=True)
counts = counts.reshape(1,4)

# Creating a datagrame
countscldf = pd.DataFrame(counts, columns = ["Cluster 0","Cluster 1","Cluster 2", "Cluster 3"])

# display
countscldf


# In[25]:


new_raw_data.shape


# PCA is a method for compressing a lot of data into something that captures the essence of the original data.
# PCA takes a dataset with a lot of dimensions and flattens it into 2 or 3 dimensions so we can look at it. We need reduce 236 variables to optimal number. 95% explained variance should be the criterium when choosing the number of principal components. 

# In[27]:


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[28]:


X = X_train
y_num = predictions


# In[38]:


# Trying with Dimentionality reduction and then Kmeans

n_components = X.shape[1]

# Running PCA with all components
pca = PCA(n_components=n_components, random_state = 453)
X_r = pca.fit(X).transform(X)


# Calculating the 95% Variance
total_variance = sum(pca.explained_variance_)
print("Total Variance in our dataset is: ", total_variance)
var_95 = total_variance * 0.95
print("The 95% variance we want to have is: ", var_95)
print("")

# Creating a df with the components and explained variance
a = zip(range(0,n_components), pca.explained_variance_)
a = pd.DataFrame(a, columns=["PCA Comp", "Explained Variance"])

# Trying to hit 95%
print("Variance explain with 30 n_compononets: ", sum(a["Explained Variance"][0:30]))
print("Variance explain with 35 n_compononets: ", sum(a["Explained Variance"][0:35]))
print("Variance explain with 40 n_compononets: ", sum(a["Explained Variance"][0:40]))
print("Variance explain with 41 n_compononets: ", sum(a["Explained Variance"][0:41]))
print("Variance explain with 50 n_compononets: ", sum(a["Explained Variance"][0:50]))
print("Variance explain with 53 n_compononets: ", sum(a["Explained Variance"][0:53]))
print("Variance explain with 55 n_compononets: ", sum(a["Explained Variance"][0:55]))
print("Variance explain with 60 n_compononets: ", sum(a["Explained Variance"][0:60]))
print("Variance explain with 70 n_compononets: ", sum(a["Explained Variance"][0:70]))
print("Variance explain with 90 n_compononets: ", sum(a["Explained Variance"][0:90]))
print("Variance explain with 110 n_compononets: ", sum(a["Explained Variance"][0:110]))
print("Variance explain with 115 n_compononets: ", sum(a["Explained Variance"][0:115]))
print("Variance explain with 120 n_compononets: ", sum(a["Explained Variance"][0:120]))
print("Variance explain with 130 n_compononets: ", sum(a["Explained Variance"][0:130]))

# Plotting the Data
plt.figure(1, figsize=(14, 8))
plt.plot(pca.explained_variance_ratio_, linewidth=2, c="r")
plt.xlabel('n_components')
plt.ylabel('explained_ratio_')

# Plotting line with 95% e.v.
plt.axvline(115,linestyle=':', label='n_components - 95% explained', c ="blue")
plt.legend(prop=dict(size=12))

# adding arrow
plt.annotate('115 eigenvectors used to explain 95% variance', xy=(115, pca.explained_variance_ratio_[115]), 
             xytext=(58, pca.explained_variance_ratio_[10]),
            arrowprops=dict(facecolor='blue', shrink=0.05))

plt.show()



# In[39]:


# Running PCA again

pca = PCA(n_components=115, random_state = 453)
X_r = pca.fit(X).transform(X)

inertia = []

#running Kmeans

for f in no_of_clusters:
    kmeans = KMeans(n_clusters=f, random_state=2)
    kmeans = kmeans.fit(X_r)
    u = kmeans.inertia_
    inertia.append(u)
    print("The innertia for :", f, "Clusters is:", u)

# Creating the scree plot for Intertia - elbow method
fig, (ax1) = plt.subplots(1, figsize=(16,6))
xx = np.arange(len(no_of_clusters))
ax1.plot(xx, inertia)
ax1.set_xticks(xx)
ax1.set_xticklabels(no_of_clusters, rotation='vertical')
plt.xlabel('n_components Value')
plt.ylabel('Inertia Score')
plt.title("Inertia Plot per k")


# In[52]:


# Attachine the clusters back to our initial Dataset that has all the data
clusters = kmeans.labels_
raw_data_1['Clusters'] = clusters

# Creating a cluster Category

raw_data_1['Clusters Category'].loc[raw_data_1['Clusters'] == 0] = 'Cluster 1'
raw_data_1['Clusters Category'].loc[raw_data_1['Clusters'] == 1] = 'Cluster 2'
raw_data_1['Clusters Category'].loc[raw_data_1['Clusters'] == 2] = 'Cluster 3'
raw_data_1['Clusters Category'].loc[raw_data_1['Clusters'] == 3] = 'Cluster 4'
raw_data_1['Clusters Category'].loc[raw_data_1['Clusters'] == 4] = 'Cluster 5'



raw_data.head(50)


# In[48]:


raw_data.to_csv('Desktop/Output_kmeans', sep='\t')


# In[ ]:




