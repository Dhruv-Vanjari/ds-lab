import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")
df.head()

df.isnull().sum()

sns.pairplot(df)

x = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
x

from sklearn.cluster import KMeans
model = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred = model.fit_predict(x)
y_pred

c0 = x[y_pred==0, :]
c1 = x[y_pred==1, :]
c2 = x[y_pred==2, :]
c3 = x[y_pred==3, :]
c4 = x[y_pred==4, :]

print("c0\n", c0[:3])
print("c1\n", c1[:3])
print("c2\n", c2[:3])
print("c4\n", c3[:3])
print("c5\n", c4[:3])

# Plotting all the cluster points
plt.scatter(c0[:, 0], c0[:, 1], label='cluster 1')
plt.scatter(c1[:, 0], c1[:, 1], label='cluster 2')
plt.scatter(c2[:, 0], c2[:, 1], label='cluster 3')
plt.scatter(c3[:, 0], c3[:, 1], label='cluster 4')
plt.scatter(c4[:, 0], c4[:, 1], label='cluster 5')

# Plotting the Cluster centroids
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='black', s=300, marker='x')
plt.legend()
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (0 - 100)")

plt.show()