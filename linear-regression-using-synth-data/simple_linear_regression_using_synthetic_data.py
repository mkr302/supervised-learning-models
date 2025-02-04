import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate synthetic data
np.random.seed(42)  # For reproducibility
X = 2.5 * np.random.rand(100, 1) + 1  # House sizes (in 1000 sq ft)
y = 5 + 3 * X + np.random.randn(100, 1) * 0.5  # House prices (in $100K)

# Step 2: Visualize the data
plt.scatter(X, y, color="blue", label="Actual Data")
plt.xlabel("House Size (1000 sq ft)")
plt.ylabel("House Price ($100K)")
plt.title("House Size vs. Price")
plt.legend()
plt.show()

# Step 3: Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Coefficient: {model.coef_[0][0]:.2f}")

# Step 7: Plot the regression line
plt.scatter(X_test, y_test, color="blue", label="Actual Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("House Size (1000 sq ft)")
plt.ylabel("House Price ($100K)")
plt.title("Linear Regression: House Size vs. Price")
plt.legend()
plt.show()
