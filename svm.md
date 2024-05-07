import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Generating a synthetic dataset for binary classification
X = np.array([[4,2],[2,4],[2,3],[3,6],[4,4],[9,10],[6,8],[9,5],[8,7],[10,8]])
y = np.array([0,0,0,0,0,1,1,1,1,1])

# Visualize the data points
plt.scatter(X[y==0, 0], X[y==0, 0])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)


model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Plotting decision boundary
def plot_decision_boundary(X, y, model):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.show()

plot_decision_boundary(X_test, y_test,model)



from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, plot_confusion_matrix

# Calculating predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Calculating confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting confusion matrix
plot_confusion_matrix(svm_classifier, X_test, y_test)
plt.title('Confusion Matrix')
plt.show()

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:",recall)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)