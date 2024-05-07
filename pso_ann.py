import numpy as np
import pyswarms as ps
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# Define the 3-3-2-1 artificial neural network. To run the experiments for other architectures, simply replace the numbers below
def fitness_function(params):
    # Unpack the parameters
    n_inputs = 3
    n_hidden1 = 3
    n_hidden2 = 2
    n_outputs = 1
    # Reshape the parameters for the neural network
    W1 = params[0:n_inputs*n_hidden1].reshape((n_inputs, n_hidden1))
    b1 = params[n_inputs*n_hidden1:n_inputs*n_hidden1+n_hidden1]
    W2 = params[n_inputs*n_hidden1+n_hidden1:n_inputs*n_hidden1+n_hidden1+n_hidden1*n_hidden2].reshape((n_hidden1, n_hidden2))
    b2 = params[n_inputs*n_hidden1+n_hidden1+n_hidden1*n_hidden2:n_inputs*n_hidden1+n_hidden1+n_hidden1*n_hidden2+n_hidden2]
    W3 = params[n_inputs*n_hidden1+n_hidden1+n_hidden1*n_hidden2+n_hidden2:].reshape((n_hidden2, n_outputs))
    b3 = params[-1]
    
    # Create the neural network
    nn = MLPRegressor(hidden_layer_sizes=(n_hidden1, n_hidden2), activation='relu', solver='adam', random_state=42)
    
    # Set the weights and biases
    nn.coefs_ = [W1, W2, W3]
    nn.intercepts_ = [b1, b2, b3]
    
    # Train the neural network
    nn.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = nn.predict(X_test)
    
    # Compute mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

#sample data collected from 10 patients
voltage = [380, 545, 368, 359, 568, 572, 50, 539, 31, 396] #in mV
age = [51, 45, 20, 24, 38, 45, 31, 54, 23, 25]
bmi = [22.2, 21.9, 18.9, 19.1, 23.4, 24.0, 22., 24.3, 18.3, 19.3]
X = np.column_stack((voltage, age, bmi))

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the number of particles and iterations
n_particles = 50
n_iterations = 100

# Define bounds for the parameters
bounds = ([-1] * (3*3 + 3 + 3*2 + 2 + 2*1 + 1), [1] * (3*3 + 3 + 3*2 + 2 + 2*1 + 1))

# Initialize the optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=3*3 + 3 + 3*2 + 2 + 2*1 + 1, bounds=bounds)

# Perform optimization
best_cost, best_params = optimizer.optimize(fitness_function, iters=n_iterations)

# Print the best parameters and cost
print("Best parameters:", best_params)
print("Best cost:", best_cost)
