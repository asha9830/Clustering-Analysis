#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore")


# In[2]:


data= pd.read_csv('C:/Users/ashan/Desktop/Project_Files/CC GENERAL.csv')
data


# In[3]:


data.describe().T


# In[4]:


data.isnull().sum().sort_values(ascending=False)


# In[5]:


cat_col = data.select_dtypes(include=['object']).columns
num_col = data.select_dtypes(exclude=['object']).columns


# In[6]:


df_v=pd.DataFrame(data['TENURE'].value_counts())
plot = df_v.plot.pie(y='TENURE', figsize=(8, 8));


# In[7]:


sns.boxplot(x = 'TENURE', y = 'CREDIT_LIMIT', data = data,palette='Pastel1');


# In[8]:


sns.boxplot(x = 'TENURE', y = 'BALANCE_FREQUENCY', data = data,palette='autumn');


# In[9]:


sns.scatterplot(x='CREDIT_LIMIT', y='PRC_FULL_PAYMENT', data=data,color='purple');


# In[10]:


sns.scatterplot(x='BALANCE', y='PURCHASES', data=data,color='purple');


# In[11]:


data[num_col].hist(bins=15, figsize=(20, 15), layout=(5, 4));


# In[12]:


data[num_col].corr()


# In[13]:


plt.subplots(figsize=(20,15))
sns.heatmap(data[num_col].corr(),annot = True);


# In[14]:


from sklearn.impute import KNNImputer
imputer = KNNImputer()
imp_data = pd.DataFrame(imputer.fit_transform(data[num_col]),columns=data[num_col].columns)
imp_data.isna().sum()


# In[15]:


imp_data


# In[16]:


from pycaret.clustering import *


# In[17]:


clu = setup(imp_data, normalize = True, 
            pca = True,
            remove_multicollinearity= True,
            session_id = 123)


# In[18]:


models()


# In[19]:


kmeans = create_model('kmeans')


# In[20]:


kmean_results = assign_model(kmeans)
kmean_results.head()


# In[21]:


plot_model(kmeans)


# In[22]:


plot_model(kmeans, plot = 'elbow')


# In[23]:


plot_model(kmeans, plot = 'silhouette')


# In[24]:


plot_model(kmeans, plot = 'distribution') #to see size of clusters


# In[25]:


plot_model(kmeans, plot = 'distribution', feature = 'TENURE')


# In[26]:


plot_model(kmeans, plot = 'distribution', feature = 'PAYMENTS')


# In[27]:


plot_model(kmeans, plot ='distance')


# In[28]:


plot_model(kmeans, plot ='tsne')


# In[29]:


hierarchical_clust = create_model('hclust')


# In[30]:


hierarchical_results = assign_model(hierarchical_clust)
hierarchical_results.head()


# In[31]:


plot_model(hierarchical_clust)


# In[32]:


plot_model(hierarchical_clust, plot = 'distribution')


# In[33]:


plot_model(hierarchical_clust, plot ='tsne')


# In[34]:


Ap_clustering = create_model('ap')


# In[35]:


Ap_results = assign_model(Ap_clustering)
Ap_results.head()


# In[36]:


plot_model(Ap_clustering)


# In[37]:


plot_model(Ap_clustering, plot = 'distribution') #to see size of clusters


# In[38]:


plot_model(Ap_clustering, plot = 'distribution', feature = 'TENURE')


# In[39]:


plot_model(Ap_clustering, plot = 'distribution', feature = 'PAYMENTS')


# In[40]:


plot_model(Ap_clustering, plot ='distance')


# In[41]:


plot_model(Ap_clustering, plot ='tsne')


# In[42]:


Density_clust = create_model('dbscan')


# In[43]:


Density_results = assign_model(Density_clust)
Density_results.head()


# In[44]:


plot_model(Density_clust)


# In[45]:


plot_model(Density_clust, plot = 'distribution') #to see size of clusters


# In[46]:


plot_model(Density_clust, plot = 'distribution', feature = 'TENURE')

