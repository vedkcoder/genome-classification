#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import seaborn as sns

rna_data = pd.read_csv('data.tar.gz')
rna_data_nnull = rna_data.dropna()

gene_data = rna_data_nnull.iloc[:,1:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_data = sc.fit_transform(gene_data)

labels = pd.read_csv('labels.csv')
labels = labels.drop(labels.columns[0], axis= 1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

df = pd.DataFrame(scaled_data, columns = list(gene_data.columns))
df['labels'] = labels

random_df = df.sample(n = 50, random_state = 0)
labels = random_df['labels']

df_50 = random_df.drop('labels', axis = 1)

class PCA:
    
    def __init__(self, data, d):
        self.data = np.array(data, dtype = np.float64)
        self.d = d
        self.n = len(data)
        self.egvals = None
        self.egvectors = None
        
    def fit(self):
        
        print(len(self.data[0]))
        ### initialize identity matrix and eigenvalues
        I = np.identity(self.n)
        e = np.asarray([1 for _ in range(self.n)])
        et = np.transpose(np.atleast_2d(e))
        
        ### calculating centered data matrix
        delta_swirl = np.matmul((I - ((np.matmul(e,et)) / self.n)), self.data)
        print('delta',delta_swirl.shape)
        
        ### take covariance of the data
        cov = np.cov(np.transpose(delta_swirl))
        print(cov.shape)
        
        ### calculate eigenvalues and vectors of delta
        egvals, egvectors = np.linalg.eigh(cov)
        print(egvals, egvals.shape, egvectors.shape)
        
        ### selecting top d vectors
        indices = np.argsort(egvals)[::-1]
        egvals = egvals[indices]
        self.egvals = egvals
        egvectors = egvectors[:,indices]
        self.egvectors = egvectors
        pca_complete = np.dot(egvectors, np.diag(np.sqrt(np.abs(egvals))))
        
        egvectors_d = egvectors[:,:self.d]
        egvals_d = egvals[:self.d]
        delta_cap = np.dot(egvectors_d, np.diag(np.sqrt(np.abs(egvals_d))))
        
        return delta_cap, pca_complete

data_50 = np.array(df_50)
pca1 = PCA(data_50, 2)

vals, complete = pca1.fit()
egs1 = np.abs(pca1.egvals)
total_var1 = np.sum(egs1)
eg_cum = np.cumsum(egs1)
nos1 = np.argmax((eg_cum/ total_var1) >= .9) + 1

I = np.identity(len(data_50))
e = np.asarray([1 for _ in range(len(data_50))])
et = np.transpose(np.atleast_2d(e))
delta_swirl = np.matmul((I - ((np.matmul(e,et)) / len(data_50))), data_50)

vects_50 = pca1.egvectors[:nos1]
transformed_data50 = np.dot(delta_swirl, vects_50.transpose())

import scipy
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering

agc1 = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'complete')
clusters_pca = agc1.fit(transformed_data50)

z1 = linkage(clusters_pca.children_)

plt.figure(figsize=(10,10))
d1 = dendrogram(z1)
plt.show()

