# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Matplotlib and Scikit-learn.
2. 
2.Create a dataset containing employee age, salary, years of experience and churn status.

3.Split the dataset into training and testing data, then create and train the Decision Tree Classifier.

4.Predict whether an employee will leave the company or stay and display the accuracy.
  

## Program:
```py
# Program to implement Decision Tree Classifier
# for predicting Employee Churn.
# Developed by:
# RegisterNumber:

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Input data
# Features: Age, Salary, Experience
X = np.array([
    [22, 20000, 1],
    [25, 25000, 2],
    [28, 30000, 3],
    [30, 35000, 4],
    [35, 45000, 6],
    [40, 50000, 8],
    [45, 60000, 10],
    [24, 22000, 1],
    [32, 40000, 5],
    [38, 55000, 7],
    [27, 28000, 2],
    [42, 58000, 9]
])

# Output: 1 = Leave, 0 = Stay
Y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0])

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, Y_train)

# Predict test data
Y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print("Actual Churn Status:", Y_test)
print("Predicted Churn Status:", Y_pred)
print("Accuracy:", accuracy * 100, "%")

# Predict for a new employee
new_employee = np.array([[26, 24000, 2]])

prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("Employee Churn Status: Will Leave")
else:
    print("Employee Churn Status: Will Stay")

# Plot decision tree
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 7))
plot_tree(
    model,
    feature_names=['Age', 'Salary', 'Experience'],
    class_names=['Stay', 'Leave'],
    filled=True
)

plt.title("Employee Churn Prediction")
plt.show()

```

## Output:
~~~
Actual Churn Status: [0 1 0]
Predicted Churn Status: [0 1 0]
Accuracy: 100.0 %

Employee Churn Status: Will Leave
~~~

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
