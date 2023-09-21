import matplotlib.pyplot as plt
from MultiClass import MC_Classifier
from LogisticRegression import LogisticReg
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions

X, y = load_iris(return_X_y=True)
X = X[:,[2,3]]

MC = MC_Classifier(LogisticReg)
MC.fit(X, y, reg_rate=1e-5, learning_rate=1, n_iter=1000)

plot_decision_regions(X, y, MC, zoom_factor=5)
plt.show()