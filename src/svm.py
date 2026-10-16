"""Linear soft-margin SVM via Pegasos (NumPy only) + random Fourier features."""
from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(self, lambda_=1e-4, n_iters=20_000, random_state=42):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.w = None
        self.b = 0.0
        self.objective_ = []
        self.classes_ = None

    @staticmethod
    def _to_pm1(y):
        y = np.asarray(y)
        cls = np.unique(y)
        return np.where(y == cls[-1], 1.0, -1.0), cls

    def _objective(self, X, y_pm):
        m = y_pm * (X @ self.w + self.b)
        return float(0.5 * self.lambda_ * self.w @ self.w + np.mean(np.maximum(0, 1 - m)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y_pm, self.classes_ = self._to_pm1(y)
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.objective_ = []
        step = max(self.n_iters // 20, 1)
        for t in range(1, self.n_iters + 1):
            i = rng.integers(n)
            eta = 1.0 / (self.lambda_ * t)
            if y_pm[i] * (X[i] @ self.w + self.b) < 1:
                self.w = (1 - eta * self.lambda_) * self.w + eta * y_pm[i] * X[i]
                self.b += eta * y_pm[i]
            else:
                self.w = (1 - eta * self.lambda_) * self.w
            if t % step == 0:
                self.objective_.append(self._objective(X, y_pm))
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X):
        s = self.decision_function(X)
        return np.where(s >= 0, self.classes_[-1], self.classes_[0])

    def margin_width(self) -> float:
        n = float(np.linalg.norm(self.w))
        return 2.0 / n if n > 0 else float("inf")

    def support_mask(self, X, y):
        y_pm, _ = self._to_pm1(y)
        return (y_pm * self.decision_function(X)) <= 1


def rff_map(X, D=400, gamma=1e-3, random_state=42):
    """Random Fourier features approximating the RBF kernel."""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    W = rng.normal(0, np.sqrt(2 * gamma), size=(X.shape[1], D))
    b = rng.uniform(0, 2 * np.pi, size=D)
    return np.sqrt(2.0 / D) * np.cos(X @ W + b)
