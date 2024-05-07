import matplotlib.pyplot as plt
import numpy as np

import random

num_samples = 5000
data_points = np.empty((0, 2))
labels = np.empty((0,), dtype=int)

ratio = np.empty((0,), dtype=int)

outside = 0
inside = 0

for i in range(1, num_samples+1):
    x = random.uniform(-1, 1);
    y = random.uniform(-1, 1);

    point = np.array([x, y])
    data_points = np.vstack([data_points, point])
    
    distance = x**2 + y**2

    if distance <= 1:
        inside+=1
        labels = np.append(labels, 1)
    else:
        labels = np.append(labels, 0)

    pi = 4 * inside / i

    ratio = np.append(ratio, pi)

print("Estimated Value of PI is : ", pi)

plt.figure(figsize=(8, 8))

plt.scatter(data_points[labels==1, 0], data_points[labels==1, 1], s=10, alpha=0.5)
plt.scatter(data_points[labels==0, 0], data_points[labels==0, 1], s=10, alpha=0.5)




plt.plot(ratio)