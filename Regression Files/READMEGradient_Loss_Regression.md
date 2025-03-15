
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Sample input data
X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

# Create and fit the model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities on the training data
probabilities = model.predict_proba(X)
predicted_labels = model.predict(X)

# Calculate the cross-entropy loss
cross_entropy_loss = log_loss(y, probabilities)

print("Predicted Probabilities:", probabilities)
print("Predicted Labels:", predicted_labels)
print("Cross-Entropy Loss:", cross_entropy_loss)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample input data
X = [[40, 0], [55, 1], [42, 1], [60, 0], [38, 0], [65, 1]]  # Age and Family History
y = [0, 1, 1, 0, 0, 1]  # 0: No Cancer, 1: Cancer

# Create and fit the model
model = LogisticRegression()
model.fit(X, y)

# New data to predict
new_data = [[50, 1], [45, 0]]  # Age and Family History for two individuals

# Predict the probability of cancer
probabilities = model.predict_proba(new_data)
predicted_labels = model.predict(new_data)

print("Predicted Probabilities:", probabilities)
print("Predicted Labels:", predicted_labels)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample input data
X = np.array([[40, 0], [55, 1], [42, 1], [60, 0], [38, 0], [65, 1]])  # Age and Family History
y = np.array([0, 1, 1, 0, 0, 1])  # 0: No Cancer, 1: Cancer

# Create and fit the model
model = LogisticRegression()
model.fit(X, y)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Age')
plt.ylabel('Family History')
plt.title('Logistic Regression Decision Boundary')
plt.show()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample input data
X = np.array([[40, 0], [55, 1], [42, 1], [60, 0], [38, 0], [65, 1]])  # Age and Family History
y = np.array(["No", "Yes", "Yes", "No", "No", "Yes"])  # No: No Cancer, Yes: Cancer

# Create and fit the model
model = LogisticRegression()
model.fit(X, y)

# Map labels to numeric values
label_map = {"No": 0, "Yes": 1}
mapped_labels = np.array([label_map[label] for label in y])

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array([label_map[label] for label in Z])
Z = Z.reshape(xx.shape)

xx = np.array(xx, dtype=float)
yy = np.array(yy, dtype=float)
Z = np.array(Z, dtype=float)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=mapped_labels, edgecolors='k')
plt.xlabel('Age')
plt.ylabel('Family History')
plt.title('Logistic Regression Decision Boundary')
plt.show()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import numpy as np
import matplotlib.pyplot as plt

# Sample input data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Initialize parameters
theta0 = 0  # Initial intercept
theta1 = 0  # Initial slope

# Hyperparameters
learning_rate = 0.05
num_iterations = 100

# Perform gradient descent
cost_history = []
theta0_history = []
theta1_history = []

for i in range(num_iterations):
    # Calculate predictions
    y_pred = theta0 + theta1 * X

    # Calculate gradients
    gradient0 = (1/len(X)) * np.sum(y_pred - y)
    gradient1 = (1/len(X)) * np.sum((y_pred - y) * X)

    # Update parameters
    theta0 -= learning_rate * gradient0
    theta1 -= learning_rate * gradient1

    # Calculate cost (mean squared error)
    cost = (1/(2*len(X))) * np.sum((y_pred - y)**2)

    # Store parameter and cost history for visualization
    theta0_history.append(theta0)
    theta1_history.append(theta1)
    cost_history.append(cost)

# Visualize parameter updates
plt.plot(theta0_history, label='theta0')
plt.plot(theta1_history, label='theta1')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Gradient Descent: Parameter Updates')
plt.legend()
plt.show()

# Visualize cost function
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Gradient Descent: Cost Function')
plt.show()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import numpy as np
import matplotlib.pyplot as plt

# Sample input data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Initialize parameters
theta0 = 0  # Initial intercept
theta1 = 0  # Initial slope

# Hyperparameters
learning_rate = 0.05
num_iterations = 100

# Perform stochastic gradient descent
cost_history = []
theta0_history = []
theta1_history = []

for i in range(num_iterations):
    for j in range(len(X)):
        # Select a random data point
        random_index = np.random.randint(0, len(X))
        x = X[random_index]
        y_true = y[random_index]

        # Calculate prediction
        y_pred = theta0 + theta1 * x

        # Calculate gradients
        gradient0 = y_pred - y_true
        gradient1 = (y_pred - y_true) * x

        # Update parameters
        theta0 -= learning_rate * gradient0
        theta1 -= learning_rate * gradient1

        # Calculate cost (mean squared error)
        cost = 0.5 * (y_pred - y_true)**2

        # Store parameter and cost history for visualization
        theta0_history.append(theta0)
        theta1_history.append(theta1)
        cost_history.append(cost)

# Visualize parameter updates
plt.plot(theta0_history, label='theta0')
plt.plot(theta1_history, label='theta1')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Stochastic Gradient Descent: Parameter Updates')
plt.legend()
plt.show()

# Visualize cost function
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Stochastic Gradient Descent: Cost Function')
plt.show()

            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    