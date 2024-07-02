import numpy as np
from sklearn.datasets import fetch_openml

class NeuralNetwork:
    def __init__(self,input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = 1e-5
        # Initialize weights and biases for the first hidden layer
        self.W1 = 0.1*np.random.randn(hidden_size1, input_size)
        self.b1 = np.zeros((hidden_size1, 1))
        
        # Initialize weights and biases for the second hidden layer
        self.W2 = 0.1*np.random.randn(hidden_size2, hidden_size1)
        self.b2 = np.zeros((hidden_size2, 1))
        
        # Initialize weights and biases for the output layer
        self.W3 = 0.1*np.random.randn(output_size, hidden_size2)
        self.b3 = np.zeros((output_size, 1))
        
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivative(self, X):
        return X * (1 - X)
    
    def forward(self, X):
        # Forward pass through the first hidden layer
        self.z1 = self.W1 @ X + self.b1
        self.y1 = self.sigmoid(self.z1)
        
        # Forward pass through the second hidden layer
        self.z2 = self.W2 @ self.y1 + self.b2
        self.y2 = self.sigmoid(self.z2)
        
        # Forward pass through the output layer
        self.z3 = self.W3 @ self.y2 + self.b3
        self.y3 = self.sigmoid(self.z3)
    
    def backward(self, X):
        # Backpropagation through the output layer
        self.delta3 = (self.y3 - X) * self.sigmoid_derivative(self.y3)
        
        # Backpropagation through the second hidden layer
        self.delta2 = np.dot(self.W3.T, self.delta3) * self.sigmoid_derivative(self.y2)
        
        # Backpropagation through the first hidden layer
        self.delta1 = np.dot(self.W2.T, self.delta2) * self.sigmoid_derivative(self.y1)
        
        # Calculate gradients
        self.W3_grad =  np.dot(self.delta3, self.y2.T)
        self.b3_grad = np.sum(self.delta3, axis=1, keepdims=True)
        
        self.W2_grad = np.dot(self.delta2, self.y1.T)
        self.b2_grad = np.sum(self.delta2, axis=1, keepdims=True)
        
        self.W1_grad = np.dot(self.delta1, X.T)
        self.b1_grad = np.sum(self.delta1, axis=1, keepdims=True)
        
        self.W1 -= self.learning_rate * self.W1_grad
        self.b1 -= self.learning_rate * self.b1_grad
        self.W2 -= self.learning_rate * self.W2_grad
        self.b2 -= self.learning_rate * self.b2_grad
        self.W3 -= self.learning_rate * self.W3_grad
        self.b3 -= self.learning_rate * self.b3_grad
    
    def loss(self, X):
        return np.mean(np.square(X - self.y3))
        
    def train(self, X, epochs, batch_size, learning_rate=1e-3):
        num_samples = X.shape[1]
        
        self.forward(X)
        self.backward(X)
        self.learning_rate = learning_rate
        stages = epochs // 100
        self.losses = np.zeros(100)
        for i in range(100):
            for j in range(stages):
                subsample = np.random.choice(num_samples, batch_size, replace=False)
                x_batch = X[:, subsample]
                self.forward(x_batch)
                self.backward(x_batch)
            self.learning_rate = max(1e-4, self.learning_rate * 0.5)
            self.forward(X)
            self.backward(X)
            self.losses[i] = self.loss(X)
            thisEpoch = stages*(i + 1)
            print(f"Epoch {thisEpoch}, Loss: {self.losses[i]:.4f}")
            

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist['data']
X = X.astype(np.float32)

# Normalize the input data
X /= 255.0

# Convert X and y to numpy arrays
X = np.array(X).T


# Set up neural network parameters
input_size = X.shape[0]
hidden_size1 = 128
hidden_size2 = 32
output_size = input_size  # Output size is same as input size for autoencoder
batch_size = 256
learning_rate = 1e-3
epochs = 10000

model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

model.train(X, epochs, batch_size, learning_rate)

