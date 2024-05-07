from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")
df.head()

x = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
x[:5]


import scipy.cluster.hierarchy as shc

dendro = shc.dendrogram(shc.linkage(x,method="ward"))
plt.title('Dendrogram plot')
plt.ylabel("Euclidean Distances")
plt.xlabel("Customers")
plt.show()