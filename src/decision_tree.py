"""Decision tree from scratch (NumPy only). Classification + regression."""
from __future__ import annotations

import numpy as np


class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.proba = None
        self.n = 0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(self, task="classification", criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, random_state=42):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root = None
        self.feature_importances_ = None
        self._n_features = 0

    def _impurity(self, y) -> float:
        if self.task == "regression":
            return float(np.mean((y - y.mean()) ** 2)) if len(y) else 0.0
        _, c = np.unique(y, return_counts=True)
        p = c / c.sum()
        if self.criterion == "entropy":
            return float(-np.sum(p * np.log2(p + 1e-15)))
        return float(1.0 - np.sum(p ** 2))  # gini

    def _best_split(self, X, y):
        n, d = X.shape
        if n < 2:
            return None
        parent = self._impurity(y)
        best = None
        for j in range(d):
            xj = X[:, j]
            order = np.argsort(xj, kind="mergesort")
            xs, ys = xj[order], y[order]
            # candidate cut points between distinct values
            diff = xs[1:] != xs[:-1]
            if not np.any(diff):
                continue
            cuts = np.where(diff)[0]
            if len(cuts) > 255:  # cap for speed on big data
                cuts = cuts[np.linspace(0, len(cuts) - 1, 255).astype(int)]
            for c in cuts:
                nl, nr = c + 1, n - c - 1
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                yl, yr = ys[: c + 1], ys[c + 1:]
                gain = parent - (nl * self._impurity(yl) + nr * self._impurity(yr)) / n
                if gain > self.min_impurity_decrease and (best is None or gain > best[2]):
                    best = (j, float((xs[c] + xs[c + 1]) / 2), float(gain))
        return best

    def _grow(self, X, y, depth: int) -> Node:
        node = Node()
        node.n = len(y)
        if self.task == "classification":
            counts = np.array([np.sum(y == k) for k in self._classes])
            node.value = self._classes[int(np.argmax(counts))]
            node.proba = counts / max(counts.sum(), 1)
        else:
            node.value = float(np.mean(y))
        imp = self._impurity(y)
        if ((self.max_depth is not None and depth >= self.max_depth)
                or len(y) < self.min_samples_split or imp == 0.0):
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        j, t, gain = split
        left = X[:, j] <= t
        if left.sum() == 0 or left.sum() == len(y):
            return node
        self.feature_importances_[j] += gain * len(y)
        node.feature, node.threshold = j, t
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._n_features = X.shape[1]
        self._classes = np.unique(y) if self.task == "classification" else None
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._grow(X, y, depth=0)
        s = self.feature_importances_.sum()
        if s > 0:
            self.feature_importances_ /= s
        return self

    def _predict_one(self, x, node: Node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in np.asarray(X, dtype=float)])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for x in X:
            node = self.root
            while not node.is_leaf():
                node = node.left if x[node.feature] <= node.threshold else node.right
            out.append(node.proba)
        return np.vstack(out).astype(float)
