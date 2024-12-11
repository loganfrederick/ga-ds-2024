"""
XGBoost (eXtreme Gradient Boosting) is a powerful machine learning algorithm that data scientists 
often choose because:
1. It handles complex non-linear relationships well
2. Built-in handling of missing values
3. Generally provides better predictive accuracy than basic models
4. Includes regularization to prevent overfitting
5. Computationally efficient and supports parallel processing
6. Feature importance ranking built-in
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

# Load sample dataset (breast cancer classification)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# XGBoost Training Process:
# 1. Starts with an initial prediction (typically the mean for regression or log-odds for classification)
# 2. For each boosting round (n_estimators=100):
#    a. Calculates the residuals (difference between actual and predicted values)
#    b. Builds a decision tree to predict these residuals
#    c. Makes predictions using this tree
#    d. Updates the overall prediction by adding the tree's predictions * learning_rate (0.1)
#    e. Applies regularization to prevent overfitting
# 3. The eval_set allows monitoring of model performance on test data during training
# 4. 'logloss' metric measures the quality of probabilistic predictions
#    (lower values indicate better performance)

# Train the model
xgb_clf.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='logloss',
    verbose=False
)

# Make predictions
y_pred = xgb_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_clf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 5 important features:")
print(feature_importance.head(5))
