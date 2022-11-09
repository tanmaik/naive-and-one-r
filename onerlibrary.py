from mlxtend.classifier import OneRClassifier
from mlxtend.data import iris_data
import numpy as np
from sklearn.model_selection import train_test_split

X, y = iris_data()

def discretize(X):
    X_discretized = X.copy()
    for col in range(X.shape[1]):
        for q, class_label in zip([1.0, 0.75, 0.5, 0.25], [3, 2, 1, 0]):
            threshold = np.quantile(X[:, col], q=q)
            X_discretized[X[:, col] <= threshold, col] = class_label
    return X_discretized.astype(int)

Xd = discretize(X)

Xd_train, Xd_test, y_train, y_test = train_test_split(Xd, y, random_state = 0, stratify = y, test_size=0.3)

oneR = OneRClassifier()

oneR.fit(Xd_train, y_train)

y_pred = oneR.predict(Xd_test)

unique = set(y_test)
confusion_matrix = np.zeros((len(unique), len(unique)), dtype=int)
for i in range(len(y_test)):
    confusion_matrix[y_test[i],y_pred[i]] = confusion_matrix[y_test[i],y_pred[i]] + 1

test_acc = np.mean(y_pred == y_test)  
print(f'\nTest accuracy {test_acc*100}%')

print("\nConfusion matrix: ")
print(confusion_matrix)
print()