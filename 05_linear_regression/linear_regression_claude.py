"""
When to Use Linear Regression:
Linear regression is appropriate when:
1. You want to predict a continuous numerical output (dependent variable)
2. There is a roughly linear relationship between your input features and output
3. Your data meets these assumptions:
   - Linear relationship between variables
   - Independence of observations
   - Homoscedasticity (constant variance in errors)
   - Normal distribution of residuals
   - Little to no multicollinearity between independent variables

Common use cases:
- Sales forecasting based on advertising spend
- House price prediction based on square footage
- Salary prediction based on years of experience
- Temperature prediction based on altitude
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
# Initialize a LinearRegression object from scikit-learn
# This creates a model that will find the best-fitting line y = mx + b
model = LinearRegression()

# Fit the model to our training data using the least squares method
# X_train: Input features (independent variables)
# y_train: Target values (dependent variable)
# This step:
#   1. Calculates the optimal slope (m) and intercept (b)
#   2. Minimizes the sum of squared differences between predictions and actual y values
#   3. Stores these parameters internally in the model object
model.fit(X_train, y_train)

# Make predictions using our trained model
# model.predict() takes our test features (X_test) and:
#   1. Uses the learned coefficients (slope and intercept)
#   2. Applies the formula: y = mx + b for each X value
#   3. Returns an array of predicted y values (y_pred)
y_pred = model.predict(X_test)

# Print model parameters
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Slope: {model.coef_[0][0]:.2f}")
print(f"R² Score: {model.score(X_test, y_test):.2f}")

# Visualize the results
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X_test, y_pred, color='red', label='Linear regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()
