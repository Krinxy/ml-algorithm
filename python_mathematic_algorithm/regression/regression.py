import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


np.random.seed(0)

# Generate data
x = np.random.randn(100, 1)

y_linear = 2 * x + 1 + np.random.randn(100, 1)


threshold = 0
y_logistic = (2 * x + 1 + np.random.randn(100, 1) > threshold).astype(int).ravel()

# Regression Model
model_linear = LinearRegression()
model_linear.fit(x, y_linear)

# Regression Model
model_logistic = LogisticRegression()
model_logistic.fit(x, y_logistic)


x_points = np.linspace(-3, 3, 300).reshape(-1, 1)
y_linear_pred = model_linear.predict(x_points)

# Sigmoid function for logistic regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

logit_values = model_logistic.coef_ * x_points + model_logistic.intercept_
y_logistic_prob = sigmoid(logit_values)


plt.figure(figsize=(14, 6))

# Linear Regression Plot
plt.subplot(1, 2, 1)
plt.scatter(x, y_linear, color='blue', label='Data Points')
plt.plot(x_points, y_linear_pred, color='red', linewidth=2, label='Linear Regression')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Logistic Regression Plot
plt.subplot(1, 2, 2)
plt.scatter(x, y_logistic, color='blue', label='Data Points')
plt.plot(x_points, y_logistic_prob, color='red', linewidth=2, label='Logistic Regression')
plt.axhline(y=0.5, color='gray', linestyle='-', label='Decision Boundary')
plt.title('Logistic Regression')
plt.xlabel('X')
plt.ylabel('Probability')
plt.legend()

plt.tight_layout()
plt.show()
