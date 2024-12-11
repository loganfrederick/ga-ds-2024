import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X = iris.data[:, [0, 1]]  # Using sepal length and sepal width
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM model
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Print model performance
print("Iris Species Classification Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualize the decision boundary
def plot_decision_boundary(X, y, model, scaler):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    X_mesh_scaled = scaler.transform(X_mesh)
    Z = model.predict(X_mesh_scaled)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.4)
    
    # Create scatter plot with a different color for each class
    for i, species in enumerate(iris.target_names):
        idx = y == i
        plt.scatter(X[idx, 0], X[idx, 1], label=species, alpha=0.8)
    
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('SVM Classification of Iris Species')
    plt.legend()
    plt.show()

# Plot the results
plot_decision_boundary(X, y, svm_model, scaler) 