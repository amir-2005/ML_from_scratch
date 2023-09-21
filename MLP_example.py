import matplotlib.pyplot as plt
from MLP import MLP
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_moons
from mlxtend.plotting import plot_decision_regions

x, y = make_moons(n_samples=1200, random_state=8, noise=0.2)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
model = MLP((3, 3), activation='tanh', learning_rate=1, epochs=2000, tol=1e-7, regularization_rate=1e-5)
model.fit(x_train, y_train)
    
print("\n", classification_report(y_test, model.predict(x_test)))
plot_decision_regions(x, y, model)
plt.show()