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

import scipy
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering

agc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'complete')
clusters = agc.fit(df_50)

z = linkage(clusters.children_)
plt.figure(figsize=(10,10))
d = dendrogram(z)
plt.show()

