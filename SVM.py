import numpy as np
from scipy.optimize import LinearConstraint, Bounds, minimize
from cvxopt import matrix, solvers


class SVM():
    
    def __init__(self, C=1, kernel="rbf", gamma=0.01, degree=3, Kernel_C=0, solver_library="cvxopt", verbose=True):
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
        
        
        if solver_library=="cvxopt":
            self.solver = "cvxopt"
        elif solver_library=="scipy":
            self.solver = "scipy"
        else :
            raise(ValueError("Invalid solver library, shoud be cvxopt or scipy"))
        
        self.verbose = verbose
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
    
    def _objective(self, alpha, X, y):  # only used for scipy optimization
        return 0.5 * np.sum(np.outer(alpha*y, alpha*y) *  self.kernel(X, X) ) - np.sum(alpha)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y = np.where(y>0, 1, -1)
        
        
        if self.solver=="scipy":
            
            alpha_init = np.zeros(X.shape[0])
            result = minimize(fun=self._objective, 
                                x0=alpha_init, 
                                args=(X,y), 
                                constraints=LinearConstraint(y,0,0), 
                                bounds= Bounds(0, self.C),
                                method = "SLSQP"   
                                )
            alphas = result.x
        
        elif self.solver == "cvxopt":
            
            n = X.shape[0]
            P = matrix(np.outer(y, y) *  self.kernel(X, X))
            q = matrix(-np.ones((n, 1)))
            A = matrix(y.reshape(1,-1).astype(float))
            b = matrix(0.0)
            G = matrix(np.vstack((-np.eye(n), np.eye(n))))
            h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
            
            solvers.options['show_progress'] = self.verbose
            result = solvers.qp(P, q, G, h, A, b)
            alphas = np.array(result["x"]).flatten()
            
        indices = alphas > 1e-5
        self.sv = X[indices]
        self.sv_y = y[indices]
        self.sv_alpha = alphas[indices]
        
        self.sv_y = self.sv_y.reshape(-1, 1)  
        self.sv_alpha = self.sv_alpha.reshape(-1, 1)  
        
        self.b = np.mean(self.sv_y - np.sum(self.sv_y * self.sv_alpha * self.kernel(self.sv, self.sv), axis=1))

        return self

    def predict(self, X):
        result = np.sum(self.sv_y * self.sv_alpha * self.kernel(self.sv, X), axis=0) + self.b
        return np.where(result<0, 0, 1)
