import numpy as np

class LogisticReg():
    """
    Logistic Regression with L2 regularization.

    Parameters:
    - learning_rate (float): The learning rate for gradient descent.
    - n_iter (int): The number of iterations for gradient descent.
    - reg_rate (float): The regularization rate (lambda) for L2 regularization.
    """

    def __init__(self, learning_rate=0.1, n_iter=5000, reg_rate=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.reg_rate = reg_rate
        self.coef = None

    def __sigmoid(self, z):
        """Sigmoid activation function to map any real number to (0, 1)."""
        return 1 / (1 + np.exp(-z))
        
    def __cost(self, x, coef, y):
        """
        Compute the cost function for logistic regression with L2 regularization.

        Parameters:
        - x (numpy array): Input features (including bias term).
        - coef (numpy array): Coefficients/weights of the model.
        - y (numpy array): Target labels.

        Returns:
        - cost (float): The cost value.
        """
        m = x.shape[0]
        z = self.__sigmoid(x @ coef)
        cost = -y * np.log(z) - (1-y) * np.log(1-z)
        reg_term = self.reg_rate * (coef ** 2)
        reg_term[0] = 0  # Do not regularize the bias term
        return sum(cost)/m + sum(reg_term)/(2*m)

    def fit(self, X, Y):
        """
        Fit the logistic regression model to the input data.

        Parameters:
        - X (numpy array): Input features (without bias term).
        - Y (numpy array): Target labels.

        Returns:
        - self (LogisticReg): The fitted model instance.
        """
        # Make sure input data is numpy array
        X = np.array(X)
        Y = np.array(Y)

        X = np.c_[np.ones(X.shape[0]), X]
        n_samples, n_features = X.shape
        self.coef = np.zeros(n_features)

        for iteration in range(self.n_iter):
            # Calculate gradient for gradient descent
            gradient = X.T @ (self.__sigmoid(X @ self.coef) - Y) / n_samples
            # Calculate gradient of regularization term
            reg_gradient = (self.reg_rate/ n_samples) * self.coef 
            # Change bias regularization gradient to zero
            reg_gradient[0] = 0
            # Update coefficients using gradient descent with L2 regularization
            self.coef -= self.learning_rate * (gradient + reg_gradient)    
            print(f"Iteration : {iteration+1} Cost : {self.__cost(X , self.coef, Y)}")
        
        return self

    def predict(self, X, prob=False):
        """
        Make predictions on new data.

        Parameters:
        - X (numpy array): Input features (without bias term).
        - prob (bool): If True, return probabilities of positive class, else return class labels.

        Returns:
        - predictions (numpy array): Predicted probabilities or class labels.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return self.__sigmoid(X @ self.coef) if prob else np.where(X @ self.coef > 0.5, 1, 0) 
  