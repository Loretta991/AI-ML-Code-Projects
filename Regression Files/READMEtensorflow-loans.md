
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import tensorflow as tf
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv(loans.csv')

# Preprocess the data
X = df.drop(['loan_status'], axis=1)
y = df['loan_status']
y = np.array([1 if i=='Charged Off' else 0 for i in y])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the neural network architecture
input_size = X_train.shape[1]
hidden_size = 32
output_size = 1

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, input_shape=(input_size,), activation='sigmoid'),
    tf.keras.layers.Dense(output_size, activation='sigmoid')
])

# Define the loss function and optimizer
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Train the model
batch_size = 64
epochs = 100

for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        with tf.GradientTape() as tape:
            y_pred = model(batch_X)
            loss = loss_fn(batch_y, y_pred)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if epoch % 10 == 0:
        print("Epoch:", epoch, " Loss:", loss.numpy())

# Evaluate the model on the testing set
y_pred = model(X_test)
y_pred = np.round(y_pred.numpy())
accuracy = np.mean(y_pred == y_test)
print("Test Accuracy:", accuracy)
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    