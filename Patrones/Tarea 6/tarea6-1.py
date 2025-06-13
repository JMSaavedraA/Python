import numpy as np

class perceptronLinearClassifier:
    def __init__(self, learning_rate=0.01, n_epochs=100):
        """
        Initializes the classifier.
        Args:
            learning_rate (float): The learning rate (eta).
            n_epochs (int): The number of passes over the training data.
        """
        self.eta = learning_rate
        self.n_epochs = n_epochs
        self.b = None  # Weights vector

    def train(self, X, y):
        """
        Trains the linear classifier.
        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values (labels), shape (n_samples,).
                            Assumed to be -1 or 1.
        """
        n_samples, n_features = X.shape

        # Initialize weights if not already done (e.g., for incremental training)
        if self.b is None:
            self.b = np.zeros(n_features)

        # Ensure y is a numpy array
        y_arr = np.asarray(y)

        for epoch in range(self.n_epochs):
            linear_scores = np.dot(X, self.b)
            predicted_classes_c = np.sign(linear_scores)
            misclassified_mask = (y_arr != predicted_classes_c)
            if not np.any(misclassified_mask):
                break
            y_for_update = misclassified_mask * y_arr
            self.b = self.b + self.eta * np.dot(X.T, y_for_update)
            
        return self

    def predict(self, X):
        """
        Predicts class labels for samples in X.
        Args:
            X (np.ndarray): Data to predict, shape (n_samples, n_features).
        Returns:
            np.ndarray: Predicted class labels (-1 or 1), shape (n_samples,).
        """
        if self.b is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Calculate linear scores: b' * X
        linear_scores = np.dot(X, self.b)
        
        predictions = np.sign(linear_scores)

        # np.sign(0) is 0.
        predictions[predictions == 0] = 1
        return predictions