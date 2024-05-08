import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

ds=pd.read_csv('SALARY DATA.csv')
ds.head()

ds['Purchased'].unique()
x=ds.iloc[:,[2,3]].values
y=ds.iloc[:,4].values
y


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
print(ytrain)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)
print(xtrain[0:10])

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=42)
classifier.fit(xtrain,ytrain)


y_pred=classifier.predict(xtest)
y_pred



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_pred)
print('Confusion Matrix: \n',cm)

from sklearn.metrics import accuracy_score
print("Accuracy",accuracy_score(ytest,y_pred))

## Plotting

from matplotlib.colors import ListedColormap
x_set,y_set=xtest,ytest

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Assuming classifier is already defined

x_set, y_set = xtest, ytest

# Create a mesh grid
x1, x2 = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
    np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01)
)

# Contour plot
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))

# Set plot limits
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# Scatter plot for each class
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)

# Add labels and legend
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Show the plot
plt.show()




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming ds is your DataFrame
# input
x = ds.iloc[:, [2, 3]].values  # Age and EstimatedSalary
# output
y = ds.iloc[:, 4].values  # Purchased

# Split the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)
print(ytrain)

# Standardization
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

print(xtrain[0:10])


import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap

# Training the Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(xtrain, ytrain)

# Visualizing the decision boundary
x1_min, x1_max = xtrain[:, 0].min() - 1, xtrain[:, 0].max() + 1
x2_min, x2_max = xtrain[:, 1].min() - 1, xtrain[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                       np.arange(x2_min, x2_max, 0.01))

Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)

plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=ListedColormap(('red', 'green')))
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for i, j in enumerate(np.unique(ytrain)):
    plt.scatter(xtrain[ytrain == j, 0], xtrain[ytrain == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Naive Bayes Classifier Decision Boundary')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()