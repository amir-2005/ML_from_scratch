import numpy as np
from scipy.optimize import LinearConstraint, Bounds, minimize


class SVM():
    def __init__(self, C=1, kernel="rbf", gamma=0.01, degree=3, Kernel_C=0):
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.Kernel_C = Kernel_C
        if kernel in ["linear", "rbf", "poly", "logistic"]:
            if kernel == "rbf":
                self.kernel = self._rbf
            elif kernel == "linear":
                self.kernel = self._linear
            elif kernel == "poly":
                self.kernel = self._poly
            elif kernel == "logistic":
                self.kernel = self._logistic
        else:
            raise(ValueError("Invalid kernel, shoud be linear, rbf, poly or logistic")) 
        
        self.sv = None
        self.sv_y = None
        self.sv_alpha = None
        self.b = None
        
    
    def _linear(self, x1, x2):
        return np.outer(x1, x2) + self.Kernel_C
    
    def _rbf(self, x1, x2):
        squared_dist = np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1)[None, :] - 2 * np.dot(x1 , x2.T)
        return np.exp(- self.gamma * squared_dist)
    
    def _poly(self, x1, x2):
        return (np.dot(x1 , x2.T) + self.Kernel_C) ** self.degree
    
    def _logistic(self, x1, x2):
        return np.tanh(self.gamma * np.dot(x1 , x2.T) + self.Kernel_C)
    
    def _objective(self, alpha, X, y):
        return 0.5 * np.sum(np.outer(alpha*y, alpha*y) *  self.kernel(X, X) ) - np.sum(alpha)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y = np.where(y>0, 1, -1)
        
        alpha_init = np.zeros(X.shape[0])
        
        result = minimize(fun=self._objective, 
                             x0=alpha_init, 
                             args=(X,y), 
                             constraints=LinearConstraint(y,0,0), 
                             bounds= Bounds(0, self.C),
                             method = "SLSQP"   
                            )
        
        indices = result.x > 1e-5
        self.sv = X[indices]
        self.sv_y = y[indices]
        self.sv_alpha = result.x[indices]
        
        self.sv_y = self.sv_y.reshape(-1, 1)  
        self.sv_alpha = self.sv_alpha.reshape(-1, 1)  
        
        self.b = np.mean(self.sv_y - np.sum(self.sv_y * self.sv_alpha * self.kernel(self.sv, self.sv), axis=1))


    def predict(self, X):
        result = np.sum(self.sv_y * self.sv_alpha * self.kernel(self.sv, X), axis=0) + self.b
        return np.where(result<0, 0, 1)
    
    
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.pyplot as plt    

    X,y = make_moons(200, noise=0.2)
    
    model = SVM(kernel="rbf", C = 100, gamma=0.5)
    model.fit(X,y)
    
    plot_decision_regions(X, y, model)
    plt.show()
