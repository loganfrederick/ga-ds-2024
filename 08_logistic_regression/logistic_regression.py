import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Generate synthetic credit data
np.random.seed(42)

def generate_credit_data(n_samples=1000):
    """Generate synthetic credit data with 13 features."""
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(60000, 20000, n_samples),
        'employment_length': np.random.normal(5, 3, n_samples),
        'debt_to_income': np.random.normal(0.3, 0.1, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'num_credit_lines': np.random.normal(5, 2, n_samples),
        'num_credit_inquiries': np.random.poisson(2, n_samples),
        'credit_utilization': np.random.normal(0.4, 0.2, n_samples),
        'num_late_payments': np.random.poisson(1, n_samples),
        'months_since_last_late': np.random.normal(24, 12, n_samples),
        'num_bankruptcies': np.random.poisson(0.1, n_samples),
        'num_tax_liens': np.random.poisson(0.05, n_samples),
        'months_since_last_delinquent': np.random.normal(36, 12, n_samples)
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable (good=1, bad=0) based on some rules
    probability = 1 / (1 + np.exp(-(
        -5 +
        0.3 * (df['credit_score'] - 700) / 50 +
        0.2 * (df['income'] - 60000) / 20000 +
        -0.4 * df['debt_to_income'] / 0.1 +
        -0.3 * df['num_late_payments'] +
        0.2 * df['employment_length'] / 3
    )))
    
    df['loan_status'] = (np.random.random(n_samples) < probability).astype(int)
    
    return df

def train_credit_model():
    """Train a logistic regression model for credit risk."""
    # Generate data
    df = generate_credit_data()
    
    # Separate features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    # The optimization process works by:
    # 1. Making predictions using the logistic function: p(y) = 1 / (1 + e^(-wx))
    # 2. Computing the error using log loss (cross-entropy)
    # 3. Adjusting coefficients using gradient descent:
    #    - Calculate gradient: grad = (prediction - actual) * feature_value
    #    - Update rule: w = w - learning_rate * gradient
    #    - This moves coefficients in direction that reduces error
    # In scikit-learn's implementation, this process is optimized using 
    # sophisticated algorithms like "LBFGS" (the default solver)
    # 4. Repeating until convergence or max iterations reached
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("\nModel Performance:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return model, scaler

def convert_to_credit_score(probability):
    """Convert probability to a 1-100 credit score."""
    return int(1 + 99 * probability)

def predict_credit_score(model, scaler, customer_data):
    """Predict credit score for a customer."""
    # Convert customer_data list to DataFrame with proper column names
    feature_names = [
        'age', 'income', 'employment_length', 'debt_to_income', 'credit_score',
        'num_credit_lines', 'num_credit_inquiries', 'credit_utilization',
        'num_late_payments', 'months_since_last_late', 'num_bankruptcies',
        'num_tax_liens', 'months_since_last_delinquent'
    ]
    customer_df = pd.DataFrame([customer_data], columns=feature_names)
    
    # Scale the input data
    scaled_data = scaler.transform(customer_df)
    
    # Get probability of good loan
    probability = model.predict_proba(scaled_data)[0][1]
    
    # Convert to credit score
    score = convert_to_credit_score(probability)
    return score

if __name__ == "__main__":
    # Train the model
    model, scaler = train_credit_model()
    
    # Example customer data
    example_customer = [
        35,      # age
        65000,   # income
        5,       # employment_length
        0.28,    # debt_to_income
        720,     # credit_score
        4,       # num_credit_lines
        1,       # num_credit_inquiries
        0.35,    # credit_utilization
        0,       # num_late_payments
        36,      # months_since_last_late
        0,       # num_bankruptcies
        0,       # num_tax_liens
        48       # months_since_last_delinquent
    ]
    
    # Predict score for example customer
    score = predict_credit_score(model, scaler, example_customer)
    print(f"\nExample Customer Credit Score: {score}")
    
    # Feature importance analysis
    feature_names = [
        'age', 'income', 'employment_length', 'debt_to_income', 'credit_score',
        'num_credit_lines', 'num_credit_inquiries', 'credit_utilization',
        'num_late_payments', 'months_since_last_late', 'num_bankruptcies',
        'num_tax_liens', 'months_since_last_delinquent'
    ]
    
    # Feature importance in logistic regression is measured using the absolute values
    # of the model's coefficients (model.coef_[0]). The magnitude of these coefficients 
    # indicates how strongly each feature influences the prediction:
    # - Larger absolute values = stronger influence
    # - Smaller absolute values = weaker influence
    # - Sign (+ or -) indicates direction of relationship
    #
    # Limitations of this approach:
    # 1. Features must be scaled for coefficients to be comparable
    # 2. Assumes features are independent
    # 3. Doesn't capture non-linear relationships
    # 4. Less sophisticated than other methods like those in tree-based models
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, abs(model.coef_[0]))
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance in Credit Risk Model')
    plt.tight_layout()
    plt.show()
