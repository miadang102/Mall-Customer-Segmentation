#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd #library for wrangling data 
import seaborn as sns #statistical visualisations
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import warnings 
warnings.filterwarnings('ignore') #remove the warnings


# In[3]:


df = pd.read_csv("C:/Users/maikh/OneDrive/Data Analytics/Portfolios/Mall Customer Segment - Python/Mall_Customers.csv")


# In[4]:


df.head() #first 5 rows


# # Univariate Analysis 

# In[5]:


df.describe()  #basic stats


# In[11]:


sns.distplot(df['Annual Income (k$)']) #histogram to view distribution


# In[12]:


df.columns


# In[15]:


column = ['Age', 'Annual Income (k$)',
       'Spending Score (1-100)']

#create hist figure for each variable 
for i in column: 
    plt.figure()
    sns.distplot(df[i])


# In[16]:


#kde PLOT
sns.kdeplot(df['Annual Income (k$)'], shade = True, hue = df['Gender'])


# In[17]:


#create KDE figure for each variable 
for i in column: 
    plt.figure()
    sns.kdeplot(df[i], shade = True, hue = df['Gender'])


# In[18]:


#create boxplots
for i in column: 
    plt.figure()
    sns.boxplot(data=df, x='Gender', y=df[i])
    


# In[20]:


#Male vs female stats
df['Gender'].value_counts #number 
df['Gender'].value_counts(normalize=True)  #percentage 


# # Bivariate Analysis 

# In[22]:


#scatter plot 
sns.scatterplot(data = df,x = 'Annual Income (k$)', y = 'Spending Score (1-100)')


# In[32]:


df = df.drop('CustomerID', axis = 1) #CustomerID not adding value so we drop


# In[33]:


sns.pairplot(df) #display all charts


# In[31]:


sns.pairplot(df, hue = 'Gender')


# In[36]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[37]:


df.corr()


# In[38]:


#create heat map to explore correlation
sns.heatmap(df.corr(), annot = True, cmap = "coolwarm")


# # Clustering - Univariate, Bivariate and Multivariate

# # Univariate Clustering 

# In[62]:


clustering1 = KMeans(n_clusters = 3)


# In[63]:


clustering1.fit(df[['Annual Income (k$)']])


# In[64]:


clustering1.labels_


# In[65]:


df['Income cluster'] = clustering1.labels_
df.head()


# In[66]:


df['Income cluster'].value_counts()


# In[67]:


clustering1.inertia_


# In[68]:


#Elbow method to determine ideal numbers of clusters 
#inertia of each cluster 

inertia_scores = []
for i in range(1,11): 
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)


# In[69]:


inertia_scores


# In[70]:


plt.plot(range(1,11), inertia_scores) #Elbow method
#The elbow starts at 3 --> change n_cluster to 3 


# In[71]:


df.columns


# In[72]:


df.groupby('Income cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# In[73]:


#Analysis: 0 cluster has a slightly higher mean age 
#Cluster 1 - lowest annual income - alignment with earlier EDA, it is observed
#that correlation bw age vs. annual income goes down as age increases
#Cluster 3 - smallest mean age, highest annual income and spending score


# # Bivariate clustering 

# In[89]:


clustering2 = KMeans(n_clusters=5)


# In[90]:


clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])


# In[91]:


clustering2.labels_


# In[92]:


df['Income and Spending cluster'] = clustering2.labels_
df.head()


# In[93]:


clustering2.inertia_


# In[94]:


#Elbow method to determine ideal numbers of clusters 
#inertia of each cluster 

inertia_scores2 = []
for i in range(1,11): 
    kmeans2 = KMeans(n_clusters = i)
    kmeans2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)


# In[95]:


inertia_scores2


# In[96]:


plt.plot(range(1,11), inertia_scores2)


# In[ ]:


#Analysis: The elbow seems to start at 5 --> Change n_cluster = 5


# In[97]:


sns.scatterplot(data = df,x = 'Annual Income (k$)', y = 'Spending Score (1-100)')


# In[101]:


#Find centroids of each cluster 
clustering2.cluster_centers_


# In[112]:


centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x', 'y']
centers


# In[ ]:


#Analysis: The scatter plot seemingly shows 5 clusters
#Which is pretty much align with n_clusters we found using elbow method


# In[132]:


#mark the centroid 
plt.scatter(x=centers['x'], y=centers['y'], s = 100, c = 'blue', marker = '*')
#Visualise clusters 
sns.scatterplot(data = df,x = 'Annual Income (k$)', y = 'Spending Score (1-100)', hue = 'Income and Spending cluster')
plt.savefig('clustering_bivariate.png')


# In[115]:


pd.crosstab(df['Income and Spending cluster'], df['Gender'])

#Analyse gender proportion of each cluster


# In[116]:


pd.crosstab(df['Income and Spending cluster'], df['Gender'], normalize = 'index')
 #percentage format


# In[117]:


#Average figures for age, annual income and spending score of each clusters

df.groupby(['Income and Spending cluster'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# # Multivariate Cluster

# In[119]:


from sklearn.preprocessing import StandardScaler 


# In[120]:


scale = StandardScaler()


# In[121]:


df.head()


# In[122]:


dff = pd.get_dummies(df)
dff.head()


# In[123]:


dff.columns


# In[125]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Female', 'Gender_Male']]


# In[126]:


dff.head()


# In[127]:


dff = scale.fit_transform(dff)


# In[129]:


inertia_scores3 = []
for i in range(1,11): 
    kmeans3 = KMeans(n_clusters = i)
    kmeans3.fit(dff)
    inertia_scores3.append(kmeans3.inertia_)
    


# In[130]:


plt.plot(range(1,11), inertia_scores3)


# In[131]:


df


# In[ ]:


#After conducting univariate, bivariate and multivariate analysis,
#the bivariate models seems to be the most telling/useful 
#Therefore, I save the database 5 clusters in Bivariate analysis

df.to_csv('Clustering.csv')


#Analysis: Target group would be cluster 2 which has a high spending score and high income
#54% of cluster 2 shoppers are female. 
#Thus, We should look for ways to attract these customers using marketing campaigns targeting popular items in this cluster
#Cluster 2 presentes an interesting opportunity to market to the customers for sales event on popular items 
#since they have low income BUT high spending score

