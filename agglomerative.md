from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")
df.head()

x = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
x[:5]

model = AgglomerativeClustering(n_clusters = 5 , linkage ='ward')
y_pred = model.fit_predict(x)
y_pred[:]

c0 = x[y_pred==0, :]
c1 = x[y_pred==1, :]
c2 = x[y_pred==2, :]
c3 = x[y_pred==3, :]
c4 = x[y_pred==4, :]

print("points from cluster 0: \n", c0[:4])

plt.scatter(c0[:, 0], c0[:, 1])
plt.scatter(c1[:, 0], c1[:, 1])
plt.scatter(c2[:, 0], c2[:, 1])
plt.scatter(c3[:, 0], c3[:, 1])
plt.scatter(c4[:, 0], c4[:, 1])

plt.show()
