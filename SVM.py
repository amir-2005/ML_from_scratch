import numpy as np
from scipy.optimize import LinearConstraint, Bounds, minimize
from cvxopt import matrix, solvers


class SVM():
    """
    Support Vector Machine (SVM) classifier.

    This class implements a flexible SVM classifier that supports various kernels
    and optimization solvers.

    Args:
        C (float, optional): Regularization parameter. The strength of the regularization
            is inversely proportional to C. Must be strictly positive. Defaults to 1.
        kernel (str, optional): Specifies the kernel type to be used in the algorithm.
            It must be one of 'linear', 'rbf', 'poly', 'logistic'. Defaults to 'rbf'.
        gamma (float, optional): Kernel coefficient for 'rbf', 'poly', and 'logistic'.
            Defaults to 0.01.  Ignored by 'linear' kernel.
        degree (int, optional): Degree of the polynomial kernel function ('poly').
            Ignored by all other kernels. Defaults to 3.
        Kernel_C (float, optional):  Independent term in kernel functions.  This corresponds
            to the 'c' in (x.T y + c)^degree or exp(-gamma ||x - x'||^2) + c . Defaults to 0.
        solver_library (str, optional): Specifies the solver library to be used.  Must be
            either "cvxopt" (using the convex optimization package) or "scipy" (using
            scipy.optimize.minimize). Defaults to "cvxopt".
        verbose (bool, optional):  If True, print verbose output from the solver. Defaults to True.

    Attributes:
        sv (numpy.ndarray): Support vectors.
        sv_y (numpy.ndarray): Labels of support vectors.
        sv_alpha (numpy.ndarray): Lagrange multipliers corresponding to support vectors.
        b (float): Bias term.
        kernel: kernel function.
        solver: optimization solver used ("cvxopt" or "scipy")
    """
    
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
        self.sv = None  # Support vectors
        self.sv_y = None # Labels of support vectors
        self.sv_alpha = None # Lagrange multipliers of support vectors
        self.b = None    # Bias term
        
    
    def _linear(self, x1, x2):
        """Linear kernel."""
        return np.outer(x1, x2) + self.Kernel_C
    
    def _rbf(self, x1, x2):
        """Radial Basis Function (RBF) kernel."""
        squared_dist = np.sum(x1**2, axis=1)[:, None] + np.sum(x2**2, axis=1)[None, :] - 2 * np.dot(x1 , x2.T)
        return np.exp(- self.gamma * squared_dist)
    
    def _poly(self, x1, x2):
        """Polynomial kernel."""
        return (np.dot(x1 , x2.T) + self.Kernel_C) ** self.degree
    
    def _logistic(self, x1, x2):
        """Logistic (Sigmoid) kernel."""
        return np.tanh(self.gamma * np.dot(x1 , x2.T) + self.Kernel_C)
    
    def _objective(self, alpha, X, y):  # only used for scipy optimization
        """Objective function for the dual SVM problem (for scipy solver)."""
        return 0.5 * np.sum(np.outer(alpha*y, alpha*y) *  self.kernel(X, X) ) - np.sum(alpha)
    
    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.

        Args:
            X (array-like of shape (n_samples, n_features)): Training vectors.
            y (array-like of shape (n_samples,)): Target values (class labels).  Values
               should be convertible to 1 and -1.

        Returns:
            self: Returns self.
        """
        X = np.array(X)
        y = np.array(y)
        y = np.where(y>0, 1, -1)  # Ensure labels are +1 or -1
        
        
        if self.solver=="scipy":
            # Scipy optimization
            alpha_init = np.zeros(X.shape[0])
            result = minimize(fun=self._objective, 
                                x0=alpha_init, 
                                args=(X,y), 
                                constraints=LinearConstraint(y,0,0),  # Constraint: y.T @ alpha = 0
                                bounds= Bounds(0, self.C),  # Bounds: 0 <= alpha_i <= C
                                method = "SLSQP"   # Sequential Least Squares Programming
                                )
            alphas = result.x  # Extract the solution (Lagrange multipliers)
        
        elif self.solver == "cvxopt":
            # CVXOPT optimization
            n = X.shape[0]
            P = matrix(np.outer(y, y) *  self.kernel(X, X)) 
            q = matrix(-np.ones((n, 1))) 
            A = matrix(y.reshape(1,-1).astype(float))
            b = matrix(0.0)  
            G = matrix(np.vstack((-np.eye(n), np.eye(n)))) 
            h = matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
            
            solvers.options['show_progress'] = self.verbose
            result = solvers.qp(P, q, G, h, A, b)  # Solve the quadratic program
            alphas = np.array(result["x"]).flatten()  # Extract the solution (Lagrange multipliers)
            
        
        indices = alphas > 1e-5 
        self.sv = X[indices] 
        self.sv_y = y[indices] 
        self.sv_alpha = alphas[indices]
        
        self.sv_y = self.sv_y.reshape(-1, 1) 
        self.sv_alpha = self.sv_alpha.reshape(-1, 1) 
        
        # Compute the bias term (b)
        #  b = mean(y_sv - sum(alpha_sv * y_sv * K(x_sv, x_sv)))  for all support vectors
        self.b = np.mean(self.sv_y - np.sum(self.sv_y * self.sv_alpha * self.kernel(self.sv, self.sv), axis=1, keepdims=True))

        return self

    def predict(self, X):
        """
        Perform classification on samples in X.

        Args:
            X (array-like of shape (n_samples, n_features)):  Test vectors.

        Returns:
            numpy.ndarray: Predicted class labels (0 or 1).
        """
        
        result = np.sum(self.sv_y * self.sv_alpha * self.kernel(self.sv, X), axis=0) + self.b
        return np.where(result<0, 0, 1)  # Convert to 0/1 predictions