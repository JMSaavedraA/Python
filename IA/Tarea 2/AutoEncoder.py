import numpy as np
from sklearn.datasets import fetch_openml

def initNN(input_size, hidden_size1, hidden_size2, output_size):
    
    # Initialize weights and biases for the first hidden layer
    W1 = 0.1*np.random.randn(hidden_size1, input_size)
    b1 = np.zeros((hidden_size1, 1))
    
    # Initialize weights and biases for the second hidden layer
    W2 = 0.1*np.random.randn(hidden_size2, hidden_size1)
    b2 = np.zeros((hidden_size2, 1))
    
    # Initialize weights and biases for the output layer
    W3 = 0.1*np.random.randn(output_size, hidden_size2)
    b3 = np.zeros((output_size, 1))
    return W1, b1, W2, b2, W3, b3

def forward(X,W1,b1,W2,b2,W3,b3):
    # Forward pass through the first hidden layer
    z1 = W1 @ X + b1
    y1 = sigmoid(z1)
    
    # Forward pass through the second hidden layer
    z2 = W2 @ y1 + b2
    y2 = sigmoid(z2)
    
    # Forward pass through the output layer
    z3 = W3 @ y2 + b3
    y3 = sigmoid(z3)
    return y3, y2, y1

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X):
    return X * (1 - X)

def backward(X,W1,b1,W2,b2,W3,b3, y1, y2, y3, learning_rate):
    # Backpropagation through the output layer
    delta3 = (y3 - X) * sigmoid_derivative(y3)
    
    # Backpropagation through the second hidden layer
    delta2 = np.dot(W3.T, delta3) * sigmoid_derivative(y2)
    
    # Backpropagation through the first hidden layer
    delta1 = np.dot(W2.T, delta2) * sigmoid_derivative(y1)
    
    # Calculate gradients
    W3_grad =  np.dot(delta3, y2.T)
    b3_grad = np.sum(delta3, axis=1, keepdims=True)
    
    W2_grad = np.dot(delta2, y1.T)
    b2_grad = np.sum(delta2, axis=1, keepdims=True)
    
    W1_grad = np.dot(delta1, X.T)
    b1_grad = np.sum(delta1, axis=1, keepdims=True)
    
    return W1_grad, b1_grad, W2_grad, b2_grad, W3_grad, b3_grad
    

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True)
X, y = mnist['data'], mnist['target']
X = X.astype(np.float32)
y = y.astype(np.uint8)

# Normalize the input data
X /= 255.0

# Convert X and y to numpy arrays
X = np.array(X).T
y = np.array(y)




# Set up neural network parameters
input_size = X.shape[0]
hidden_size1 = 128
hidden_size2 = 16
output_size = input_size  # Output size is same as input size for autoencoder
batch_size = 256
learning_rate = 1e-5
epochs = 10000

def train(X, learning_rate, epochs, batch_size):
    num_samples = X.shape[1]
    W1, b1, W2, b2, W3, b3 = initNN(input_size, hidden_size1, hidden_size2, output_size)
    stages = epochs // 100
    loss = np.zeros(stages)
    for i in range(stages):
        for j in range(98):
            subsample = np.random.choice(num_samples, batch_size, replace=False)
            x_batch = X[:, subsample]
            y3, y2, y1 = forward(x_batch, W1, b1, W2, b2, W3, b3)
            W1_grad, b1_grad, W2_grad, b2_grad, W3_grad, b3_grad = backward(x_batch, W1, b1, W2, b2, W3, b3, y1, y2, y3, learning_rate)
            W1 -= learning_rate * W1_grad
            b1 -= learning_rate * b1_grad
            W2 -= learning_rate * W2_grad
            b2 -= learning_rate * b2_grad
            W3 -= learning_rate * W3_grad
            b3 -= learning_rate * b3_grad
        y3, y2, y1 = forward(X, W1, b1, W2, b2, W3, b3)
        W1_grad, b1_grad, W2_grad, b2_grad, W3_grad, b3_grad = backward(X, W1, b1, W2, b2, W3, b3, y1, y2, y3, learning_rate)
        W1 -= learning_rate * W1_grad
        b1 -= learning_rate * b1_grad
        W2 -= learning_rate * W2_grad
        b2 -= learning_rate * b2_grad
        W3 -= learning_rate * W3_grad
        b3 -= learning_rate * b3_grad
        loss[i] = np.mean(np.square(X - y3))
        print(f"Epoch {100*(i+1)}, Loss: {loss:.4f}")
    return W1, b1, W2, b2, W3, b3, y3, loss


def ADAM(grad,momentum, inertia, old_step, old_speed, momentum_t, inertia_t,stepsize):
    p  = momentum * old_step + (1.0 - momentum) * grad
    speed  = inertia * old_speed + (1.0 - inertia) * (grad**2)
    p = p / (1.0 - momentum_t)
    speed = speed / (1.0 - inertia_t)
    step = - (stepsize / (np.sqrt(abs(speed)) + 1e-6))* p
    return step, speed