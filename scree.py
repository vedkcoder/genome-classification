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
    
pca = PCA(scaled_data,2)
pca_vals, pca_complete= pca.fit()
pca_df = pd.DataFrame(pca_vals, columns= ['PCA1','PCA2'])
print(pca_df)

plt.figure(figsize=(10,10))
plt.scatter(x = pca_df['PCA1'], y = pca_df['PCA2'])
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

plt.figure(figsize=(7,7))
plt.scatter(x = pca_df['PCA1'], y = pca_df['PCA2'])
sns.regplot(data= pca_df, x = 'PCA1', y= 'PCA2')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

egs = np.abs(pca.egvals)
total_var = np.sum(np.abs(egs))

contri = (np.abs(egs) / total_var)*100
print(contri[0], contri[1])

eg_cum = np.cumsum(egs)
exp_var = egs / total_var

plt.figure(figsize = (10,10))
plt.plot(exp_var[:100],'-o')
plt.xlabel('No of Components')
plt.ylabel('Explained Variance')
plt.show()

selected_comp_no = np.argmax((eg_cum/ total_var) >= .75) + 1
print(selected_comp_no)

selected_comp_no1 = np.where(egs > 1)
comps = egs[:max(selected_comp_no1[0])]

