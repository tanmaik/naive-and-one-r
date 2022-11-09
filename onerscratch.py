# one r classifer

from sklearn.datasets import load_iris
import numpy as np
from tqdm import tqdm 
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y = True)

def discretize(X):
    X_discretized = X.copy()
    for col in range(X.shape[1]):
        for q, class_label in zip([1.0, 0.75, 0.5, 0.25], [3, 2, 1, 0]):
            threshold = np.quantile(X[:, col], q=q)
            X_discretized[X[:, col] <= threshold, col] = class_label
    return X_discretized.astype(int)

Xd = discretize(X)

Xd_train, Xd_test, y_train, y_test = train_test_split(Xd, y, random_state = 0, stratify = y, test_size=0.3)

training_observation_num = len(Xd_train)
print("\nNumber of training observations:", training_observation_num)
rules = dict()

print("\nBuilding rule database...")
for feature in range(len(Xd_train[0])):
    rules[feature] = dict()
    for attribute_value in range(4):
        rules[feature][attribute_value] = dict()
        for observation_num in range(len(Xd_train)):
            if Xd_train[observation_num][feature] == attribute_value:
                if y_train[observation_num] not in rules[feature][attribute_value]:
                    rules[feature][attribute_value][y_train[observation_num]] = 0
                rules[feature][attribute_value][y_train[observation_num]] += 1

errors = []
for attribute in rules: 
    total_error = 0
    for attribute_value in rules[attribute]:
        total_error += sum(rules[attribute][attribute_value].values()) - max(rules[attribute][attribute_value].values())
    errors.append(total_error)
errors = [round(error / training_observation_num, 3) for error in errors]
print("Error rate for each attribute:", errors)

best_attribute = errors.index(min(errors))
print("Best attribute:", best_attribute)

print("\nNow establishing rules...")

rules = rules[best_attribute]
for attribute_value in rules:
    rules[attribute_value] = max(rules[attribute_value], key=rules[attribute_value].get)

print("Rules (where key = attribute_value and value = class_value):", rules)

print("\nNow testing...\n")


correct = 0
total = 0
confusion_matrix = np.zeros((len(set(y)), len(set(y))), dtype=int)
for i in range(len(Xd_test)):
    if rules[Xd_test[i][best_attribute]] == y_test[i]:
        correct += 1
    total += 1
    confusion_matrix[y_test[i], rules[Xd_test[i][best_attribute]]] = confusion_matrix[y_test[i], rules[Xd_test[i][best_attribute]]] + 1
    print("Actual:", y_test[i], "Predicted:", rules[Xd_test[i][best_attribute]], "Correct:", rules[Xd_test[i][best_attribute]] == y_test[i])


print("\nTest accuracy:", round(correct / total, 3) * 100, "%")
print("Confusion matrix:")
print(confusion_matrix)
print()
