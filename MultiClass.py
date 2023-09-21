import numpy as np

class MC_Classifier:
    """
        One Versus One Multi-Class Classification using Binary Classification models.
        
        Parameters:
        - classifier (class): Binary classification model class used for creating individual models.
            Note : Input classifier should be the class of classifier model, not instance of it
    """
    def __init__(self, classifier):
        self.models = []
        self.classifier = classifier

    def fit(self, X, y, **kwargs):
        """
        Train binary classification models for multi-class classification.
        
        Parameters:
        - X (array-like): Training data of shape (n_samples, n_features).
        - y (array-like): Target labels of shape (n_samples,).
        - **kwargs: Additional keyword arguments passed to the binary classification models.

        Returns:
        - self (MC_Classifier): The trained multi-class classifier
        """
        self.n_classes = len(np.unique(y))
        X = np.array(X)
        y = np.array(y)

        labels_comb = []
        for i in range(self.n_classes):
            for j in range(i+1, self.n_classes):
                labels_comb.append((i,j))

        for comb in labels_comb:
            y_bin = y[(y==comb[0]) | (y==comb[1])]
            y_bin[y_bin==comb[0]] = 0
            y_bin[y_bin==comb[1]] = 1
            X_bin = X[(y==comb[0]) | (y==comb[1])]

            model = self.classifier(**kwargs)
            model.fit(X_bin,y_bin)

            self.models.append((comb, model))

        return self

    def predict(self, X):
        """
        Predict class labels for the given input data.

        Parameters:
        - X (array-like): Input data of shape (n_samples, n_features).

        Returns:
        - predictions (array): Predicted class labels.
        """
        votes = np.zeros((len(X), self.n_classes))
        for comb, model in self.models:
            prediction = model.predict(X)
            votes[np.where(prediction == 0), comb[0]] += 1
            votes[np.where(prediction == 1), comb[1]] += 1

        return np.argmax(votes, axis=1)
        