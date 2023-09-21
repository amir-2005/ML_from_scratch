import numpy as np


class LinReg():

    def __init__(self, method="NE", reg_method="None", learning_rate=0.1, reg_rate=0.1, n_iter=50000):
        if method not in ["GD", "NE"]:
            raise ValueError(
                "The training method must be GD (Gradient Descent) or NE (Normal Equation)")
        if reg_method not in ["Ridge", "Lasso", "None"]:
            raise ValueError(
                "The regularization method must be Lasso or Ridge or None")

        self._method = method
        self._reg_method = reg_method
        self._learning_rate = learning_rate
        self._reg_rate = reg_rate
        self.n_iter = n_iter
        self._coef = None

    def __reg_gradient(self, coef, n):
        # Select regularization method according to user's choice
        if self._reg_method == "None":
            return np.zeros(coef.shape[0])
        if self._reg_method == "Ridge":
            coef[0] = 0 # Set bias planty to zero
            return self._reg_rate * coef / n
        if self._reg_method == "Lasso":
            coef[0] = 0 # Set bias planty to zero
            return self._reg_rate * np.sign(coef) / n

    def __normal_eq(self, X, Y):
        # Finding coefficients with Normal Equaltion
        self._coef = np.linalg.inv(X.T @ X) @ X.T @ Y

    def __gradient_des(self, X, Y):
        # Store number of samples and features in variables
        n_samples, n_features = X.shape
        # Initialize weights randomly from normal distribution
        self._coef = np.zeros(n_features)

        # Finding coefficients with Gradient Descent
        for _ in range(self.n_iter):
            copy = np.copy(self._coef)
            gradient = (X.T @ (X @ self._coef - Y)) + self.__reg_gradient(self._coef, n_samples)
            self._coef -= self._learning_rate * gradient
            print(sum(abs(copy - self._coef)))

    def fit(self, X_train, Y_train):
        # Make sure inputs are numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train).flatten()
        # Add x_0 = 1 to each instance for the bias term
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]

        # Select method for training model according to user's choice
        if self._method == "NE":
            self.__normal_eq(X_train, Y_train)
        if self._method == "GD":
            self.__gradient_des(X_train, Y_train)

        return self

    def predict(self, X):
        # raise error if fit method is'nt called
        if type(self._coef) == type(None):
            raise NotImplementedError("The model has not yet been build")

        # Make sure inputs are numpy arrays
        X = np.array(X)
        # Add x_0 = 1 to each instance for the bias term
        X = np.c_[np.ones(X.shape[0]), X]

        return X @ self._coef


if __name__ == "__main__":

    from sklearn.datasets import fetch_california_housing, make_regression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import Normalizer

    data = make_regression(n_samples=3000, n_features=3, noise=10, coef=True, random_state=3)
    x = data[0]
    y = data[1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

    model1 = LinReg(method="GD", reg_method="Lasso", learning_rate=1e-5, n_iter=1000)
    model1.fit(x_train, y_train)

    y_pred = model1.predict(x_test)
    y_pred = model1.predict(x_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(r2_score(y_test, y_pred))
    print(model1._coef)

    print("\n")

    model2 = LinReg(method="NE")
    model2.fit(x_train, y_train)

    y_pred = model2.predict(x_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(r2_score(y_test, y_pred))
    print(model2._coef)

    print("\n", data[2])
