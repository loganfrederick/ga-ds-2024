"""
Why KNN excels in these scenarios:

1. Small to medium-sized datasets:
   - KNN must store and compare against ALL training data points
   - For N training samples, each prediction requires N distance calculations
   - With large datasets (e.g., millions of samples), this becomes computationally infeasible
   - Memory requirements also grow linearly with dataset size

2. Low-noise datasets:
   - KNN makes predictions based on local neighborhood voting
   - Noisy data points can significantly skew the voting in their local area
   - Example: If a single mislabeled point exists, it affects k predictions around it
   - Clean data ensures the "nearest neighbors" truly represent the correct class

3. Clear decision boundaries:
   - KNN assumes similar points (in feature space) should have similar labels
   - Works best when this "locality assumption" holds true
   - Example: In image recognition, pixels of a cat will be more similar to other cat images
     than to dog images in the feature space

4. Need for interpretability:
   - KNN's decisions are transparent: "This sample was classified as X because its
     k nearest neighbors were X"
   - No complex math or black-box transformations
   - You can literally show stakeholders the k similar cases that led to the decision
   - Critical for applications requiring trust and explanation (healthcare, finance)

5. Balanced datasets:
   - With imbalanced classes, majority class tends to dominate the k-nearest neighbors
   - Example: If 90% of data is class A, roughly 90% of any point's neighbors will be class A
   - This leads to over-prediction of the majority class
   - Balanced datasets ensure fair representation in the local neighborhood

The key insight is that KNN is a "lazy learner" that makes decisions based purely on local
information. This makes it powerful when local patterns are meaningful and the data is clean,
but vulnerable when these assumptions don't hold.
"""

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNearestNeighbors:
    def __init__(self, k=3):
        """
        Initialize KNN classifier
        k: number of nearest neighbors to consider (default=3)
        """
        self.k = k

    def fit(self, X, y):
        """
        Store training data - KNN is a lazy learner and doesn't actually 'train'
        X: training features
        y: training labels
        """
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between two points
        x1, x2: points in n-dimensional space
        
        The Euclidean distance is calculated as:
        sqrt((x1[0] - x2[0])^2 + (x1[1] - x2[1])^2 + ... + (x1[n] - x2[n])^2)
        where n is the number of dimensions/features
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Predict labels for test data
        X: test features
        Returns: predicted labels
        """
        predictions = []
        
        # Loop through each test sample
        for x in X:
            # Calculate distances between test sample and all training samples
            distances = []
            for x_train in self.X_train:
                dist = self.euclidean_distance(x, x_train)
                distances.append(dist)
            
            # Get indices of k-nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k-nearest neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Majority vote to determine prediction
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train KNN classifier
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)

    # Make predictions
    predictions = knn.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
