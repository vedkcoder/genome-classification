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

nos = np.argmax((eg_cum/ total_var) >= .9) + 1

I = np.identity(len(scaled_data))
e = np.asarray([1 for _ in range(len(scaled_data))])
et = np.transpose(np.atleast_2d(e))
delta_swirl = np.matmul((I - ((np.matmul(e,et)) / len(scaled_data))), scaled_data)

selected_vectors = pca.egvectors[:nos]
transformed_data = np.dot(delta_swirl, selected_vectors.transpose())

class kmeans:
    
    def __init__(self,k):
        self.data = None
        self.k = k
        self.T = None
        self.max_iter = 20
        
    def initialize_centroid(self):
        min_ = np.min(self.data, axis = 0)
        max_ = np.max(self.data, axis = 0)
        centroids = [np.random.uniform(min_, max_) for _ in range(self.k)]
            
        return centroids
        
    def assign_clusters(self,centroids):
        clusters = [[] for i in range(self.k)]
        
        for i, point in enumerate(self.data):
            centroid_num = np.argmin(np.sqrt(np.sum((point - centroids)**2, axis = 1)))
            
            clusters[centroid_num].append(i)
            
        return clusters
    
    def update_centroids(self,clusters):
        new_centroids = np.zeros((self.k, self.data.shape[1]))
        
        for i,cluster in enumerate(clusters):
            new_centroid = np.mean(self.data[cluster], axis = 0)
            
            new_centroids[i] = new_centroid
            
        return new_centroids
    
    def calculate(self,centroids, previous_centroids):
        diff = np.sum(np.subtract(previous_centroids, centroids))
        
        return diff
    
    def fit(self,data,T):
        self.data = data
        self.T = T
        
        current_centroids = self.initialize_centroid()
        
        iter = 0
        while iter < self.max_iter:
            clusters = self.assign_clusters(current_centroids)
            previous_centroids = current_centroids
            current_centroids = self.update_centroids(clusters)
            diff = self.calculate(current_centroids, previous_centroids)
            
            if diff < self.T:
                final_clusters = np.zeros((self.data.shape[0]))
                for i,cluster in enumerate(clusters):
                    for j in cluster:
                        final_clusters[j] = i
                
                return final_clusters
            iter += 1
        
        final_clusters = np.zeros((self.data.shape[0]))
        for i,cluster in enumerate(clusters):
            for j in cluster:
                final_clusters[j] = i
        return final_clusters
    
km = kmeans(5)
preds = km.fit(scaled_data,0.01)

labels = pd.read_csv('labels.csv')
labels = labels.drop(labels.columns[0], axis= 1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

def calculate_ser(data):
    
    clist= []
    for i in range(5):
        ck = data[data['preds']==i]
        clist.append(ck)
    
    ser = 0
    for ck in clist:
        i = 0
        df = ck
        yi = len(df[df['labels'] ==i])
        ni = len(df[df['labels'] != i])
        if yi == 0 and ni == 0:
            err_ck = 0
        else:
            err_ck = ni/(yi+ni)
        ser += err_ck 
        i += 1
        
    return ser

sers = []
for j in range(20):
    km = kmeans(5)
    preds = km.fit(scaled_data, 0.01)
    df = pd.DataFrame(scaled_data)
    df['labels'] = labels
    df['preds'] = preds
    sers.append(calculate_ser(df))
    
sers = pd.DataFrame(sers)
sers = sers.transpose()

plt.figure(figsize=(7,7))
sers.boxplot()
plt.xlabel('Iteration value')
plt.ylabel('Error')
plt.show()

