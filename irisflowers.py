# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: different iris species

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifiers
knn = KNeighborsClassifier()
log_reg = LogisticRegression(max_iter=200)
rf = RandomForestClassifier()

# Train and test K-Nearest Neighbors (KNN)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f'K-Nearest Neighbors accuracy: {knn_accuracy * 100:.2f}%')

# Train and test Logistic Regression
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print(f'Logistic Regression accuracy: {log_reg_accuracy * 100:.2f}%')

# Train and test Random Forest Classifier
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest accuracy: {rf_accuracy * 100:.2f}%')

# Compare all models' accuracy
print("\nModel Comparison:")
print(f"K-Nearest Neighbors Accuracy: {knn_accuracy * 100:.2f}%")
print(f"Logistic Regression Accuracy: {log_reg_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")