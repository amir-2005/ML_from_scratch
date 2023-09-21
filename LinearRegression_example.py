import numpy as np
from LinearRegression import LinReg
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

data = make_regression(n_samples=1000, n_features=3, noise=0.2, coef=True, random_state=3)
x = data[0]
y = data[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

model1 = LinReg(method="GD", reg_method="Lasso", learning_rate=1e-5, n_iter=1000)
model1.fit(x_train, y_train)

y_pred = model1.predict(x_test)

print('\n')
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

print("\n", "Real coefficients :", data[2])