import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class PolynomialRegression(BaseEstimator, RegressorMixin):
    def __init__(self, degree=1, alpha=0.0):
        self.degree = degree      # stepen polinoma M
        self.alpha = alpha        # lambda za L2 regularizaciju

    def _phi(self, X):
        x = np.asarray(X, dtype=float).reshape(-1)          # (N,) ili (N,1) -> (N,)
        return np.vander(x, N=self.degree + 1, increasing=True)  # kolone x^0..x^M

    def fit(self, X, y):
        Phi = self._phi(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        N = Phi.shape[0]
        A = Phi.T @ Phi
        b = Phi.T @ y
        if self.alpha > 0:                       # Ridge: (Phi^T Phi + N*lambda*I) w = Phi^T y
            I = np.eye(self.degree + 1)
            I[0, 0] = 0                          # slobodni clan w0 se ne kaznjava
            A = A + N * self.alpha * I
        try:
            self.coef_ = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:            # singularna matrica
            self.coef_ = np.linalg.lstsq(A, b, rcond=None)[0]
        return self

    def predict(self, X):
        return self._phi(X) @ self.coef_

    def error(self, X, y):                       # Err(w) = 1/(2N) * suma kvadrata
        r = np.asarray(y, dtype=float).reshape(-1) - self.predict(X)
        return 0.5 * np.mean(r ** 2)

    def score(self, X, y):                       # MSE = 1/N * suma kvadrata
        r = np.asarray(y, dtype=float).reshape(-1) - self.predict(X)
        return np.mean(r ** 2)