import numpy as np


class MLP:
    """Multi-layer Perceptron classifier ,
    Train a neural network by optimizng log-loss function using stochastic 
    gradient descesnt.
    
    Parameters: 
    -----------
    hidden_layer_sizes : List[int]
        Sizes of the hidden layers in the neural network. The ith element represents
        the number of neurons in the ith hidden layer

    activation : str
        Activation function to be used in hidden layers ('logistic', 'tanh', 'relu').

    learning_rate : float
        Learning rate for gradient descent.

    epochs : int
        Number of training iterations.

    batch_size : int
        Number of samples per batch for sgd optimization.

    tol : float
        Tolerance value for detecting convergence.

    regularization_rate : float
        Strength of L2 regularization term, set 0 to disable regularization.
    """
        
    def __init__(self, hidden_layer_sizes, activation='logistic', learning_rate=0.01, epochs=100,
                 batch_size=200, tol=1e-8, regularization_rate=0.1):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.reg_rate=regularization_rate

    def __activation(self, x):
        """
        Apply the chosen activation function element-wise to the input.
        
        Parameters:
        - x (array-like): Input data.

        Returns:
        - result (array-like): Output after applying the activation function.
        """
        if self.activation == 'logistic':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Invalid activation function")

    def __deriv_activation(self, Y):
        """
        Calculate the derivative of the selected activation function.
        
        Parameters:
        - Y (array-like): Output from the activation function.

        Returns:
        - derivative (array-like): Derivative of the activation function.
        """
        if self.activation == 'logistic':
            return (Y) * (1-Y)
        elif self.activation == 'tanh':
            return 1 - np.square(Y)
        elif self.activation == 'relu':
            return 1 * (Y > 0)

    def __softmax(self, x):
        """
        Apply the softmax function along the specified axis (to use for output layer).
        
        Parameters:
        - x (array-like): Input data.

        Returns:
        - softmax_result (array-like): Output after applying the softmax function.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def __init_weights(self, n_features, n_classes):
        """
        Initialize weights and biases for the neural network layers.
        
        Parameters:
        - n_features (int): Number of input features.
        - n_classes (int): Number of output classes.
        """
        self.weights = []
        self.biases = []
        prev_size = n_features

        for size in self.hidden_layer_sizes:
            self.weights.append(np.random.randn(prev_size, size))
            self.biases.append(np.zeros((1, size)))
            prev_size = size

        self.weights.append(np.random.randn(prev_size, n_classes))
        self.biases.append(np.zeros((1, n_classes)))

    def __feed_forward(self, X):
        """
        Perform forward propagation through the neural network.
        
        Parameters:
        - X (array-like): Input data.

        Returns:
        - activations (list of arrays): array of activation outputs for each neuron.
        """
        activations = [X]
        for i in range(len(self.hidden_layer_sizes)):
            layer_input = activations[-1].dot(self.weights[i]) + self.biases[i]
            activations.append(self.__activation(layer_input))

        # Calculate output with softmax function
        output_layer_input = activations[-1].dot(self.weights[-1]) + self.biases[-1]
        activations.append(self.__softmax(output_layer_input))

        return activations

    def __backpropagation(self, X, y, activations):
        """
        Perform backpropagation to calculate gradients for weight and bias updates.
        
        Parameters:
        - X (array-like): Input data.
        - y (array-like): Target labels.
        - activations (list of arrays): List of activation outputs for each layer.

        Returns:
        - gradient_weights (list of arrays): Gradients for weight updates.
        - gradient_biases (list of arrays): Gradients for bias updates.
        """
        num_samples = X.shape[0]
        gradient_weights = [np.zeros_like(w) for w in self.weights]
        gradient_biases = [np.zeros_like(b) for b in self.biases]

        # calculate the error and gradient for output layer
        error = activations[-1] - y
        output_layer_reg_term = (self.reg_rate/num_samples)*self.weights[-1] 
        gradient_weights[-1] = (activations[-2].T.dot(error) / num_samples) + output_layer_reg_term
        gradient_biases[-1] = np.sum(error, axis=0, keepdims=True) / num_samples

        # calculate the error and gradient for hidden layers
        for i in range(len(self.hidden_layer_sizes), 0, -1):
            error = error.dot(self.weights[i].T) * self.__deriv_activation(activations[i])
            reg_term = (self.reg_rate/num_samples)*self.weights[i-1] 
            gradient_weights[i-1] = (activations[i-1].T.dot(error) / num_samples) + reg_term
            gradient_biases[i-1] = (np.sum(error, axis=0, keepdims=True) / num_samples)

        return gradient_weights, gradient_biases

    def __one_hot_encode_labels(self, y):
        """
        Perform one-hot encoding on target labels.
        
        Parameters:
        - y (array-like): Target labels.

        Returns:
        - one_hot_y (array-like): One-hot encoded labels.
        """
        one_hot_y = np.zeros((y.size, y.max() + 1))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y

    def __prepare_batches(self, X, y):
        """
        Split data into batches for mini-batch gradient descent.
        
        Parameters:
        - X (array-like): Input data.
        - y (array-like): Target labels.

        Returns:
        - x_batches (list of arrays): List of input data batches.
        - y_batches (list of arrays): List of target label batches.
        """
        batches_num = np.ceil(X.shape[0] // self.batch_size)
        x_batches = np.array_split(X, batches_num)
        y_batches = np.array_split(y, batches_num)
        return x_batches, y_batches

    def fit(self, X, y):
        """
        Train the MLP classifier on the given data.

        Parameters:
        - X (array-like): Training data of shape (n_samples, n_features).
        - y (array-like): Target labels of shape (n_samples,).

        Returns:
        - self (MLP): The trained MLP classifier.

        """
        X = np.array(X) # Making sure that input data is a numpy array
        y = np.array(y) # Making sure that target is a numpy array
        y = self.__one_hot_encode_labels(y)

        n_samples, n_features = X.shape
        n_classes = y.shape[1]

        self.__init_weights(n_features, n_classes)
        x_batches, y_batches = self.__prepare_batches(X,y)
        
        perv_loss , n_iter_no_chang= 0, 0

        for epoch in range(1, self.epochs+1):
            epoch_loss = 0
            for X, y in zip(x_batches, y_batches):
                activations = self.__feed_forward(X)
                gradient_weights, gradient_biases = self.__backpropagation(X, y, activations)
                
                # calclulate loss function
                epoch_loss += -np.sum(y * np.log(activations[-1])) / n_samples
                # calculate reqularization term
                epoch_loss += self.reg_rate * np.sum([np.sum(np.square(w)) for w in self.weights]) / n_samples

                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * gradient_weights[i]
                    self.biases[i] -= self.learning_rate * gradient_biases[i]

            print(f"Epoch : {epoch} Loss : {epoch_loss}")
            
            if abs(perv_loss-epoch_loss) < self.tol:
                n_iter_no_chang += 1
            else:
                n_iter_no_chang = 0

            if n_iter_no_chang == 3:
                    print(f"Converge in epoch {epoch}")
                    break
            
            perv_loss = epoch_loss

        return self

    def predict(self, X):
        """
        Predict the class labels for given input data.

        Parameters:
        - X (array-like): Input data of shape (n_samples, n_features).

        Returns:
        - predictions (array): Predicted class labels.
        """
        activations = self.__feed_forward(X)
        return np.argmax(activations[-1], axis=1)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.datasets import make_blobs, make_circles, make_moons
    from mlxtend.plotting import plot_decision_regions


    x, y = make_moons(n_samples=1200, random_state=8, noise=0.2)
    print(x.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
    model = MLP((3, 3), activation='logistic', learning_rate=1, epochs=2000, tol=0, regularization_rate=0)
    model.fit(x_train, y_train)
    
    plot_decision_regions(x, y, model)
    plt.show()
