from __future__ import annotations

"""MGAGC migrated from the official MGAGC.ipynb implementation."""

from dataclasses import asdict

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans

from core.algorithm import Algorithm

from .config import MGAGCConfig


class _GranularBall:
    def __init__(self, points: np.ndarray, center: np.ndarray, parent: "_GranularBall | None" = None) -> None:
        self.points, self.center, self.parent = points, center, parent


def generate_anchors(X: np.ndarray, min_points: int) -> np.ndarray:
    """Official adaptive granular-ball generation; returns anchor columns."""
    pending = [_GranularBall(X, X.mean(axis=0))]
    accepted: list[_GranularBall] = []
    while pending:
        ball = pending.pop(0)
        if ball.points.shape[0] <= min_points:
            accepted.append(ball)
            continue
        labels = KMeans(n_clusters=2, random_state=0).fit_predict(ball.points)
        children: list[_GranularBall] = []
        parent_distances = np.linalg.norm(ball.points - ball.center, axis=1)
        parent_variance = np.mean((parent_distances - parent_distances.mean()) ** 2)
        parent_ratio = 0.0 if ball.parent is None else ball.points.shape[0] / ball.parent.points.shape[0]
        parent_ad = parent_ratio / parent_variance if parent_variance > 0 else np.inf
        child_ad = 0.0
        valid = True
        for label in range(2):
            points = ball.points[labels == label]
            if points.shape[0] <= min_points:
                valid = False
                break
            center = points.mean(axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            variance = np.mean((distances - distances.mean()) ** 2)
            child_ad += (points.shape[0] / ball.points.shape[0] * 2) / variance if variance > 0 else np.inf
            children.append(_GranularBall(points, center, ball))
        if valid and child_ad / 2 >= parent_ad:
            pending.extend(children)
        else:
            accepted.append(ball)
    if not accepted:
        raise RuntimeError("MGAGC generated no granular-ball anchors")
    return np.asarray([ball.center for ball in accepted], dtype=float).T


class MGAGC(Algorithm):
    def __init__(self, config: MGAGCConfig, anchors: np.ndarray | None = None) -> None:
        self.config, self.anchors = config, anchors

    @staticmethod
    def _weighted_distance(X: np.ndarray, A: np.ndarray, w: np.ndarray) -> np.ndarray:
        Xw, Aw = X * w[:, None], A * w[:, None]
        return np.maximum(np.sum(Xw**2, axis=0)[:, None] + np.sum(Aw**2, axis=0)[None, :] - 2 * Xw.T @ Aw, 0)

    @staticmethod
    def _components(Z: np.ndarray, labels: bool = False) -> int | np.ndarray:
        n, m = Z.shape; graph = np.zeros((n + m, n + m), dtype=np.uint8)
        graph[:n, n:] = Z > 1e-6; graph[n:, :n] = graph[:n, n:].T
        count, component_labels = connected_components(graph, directed=False, return_labels=True)
        return component_labels[:n] if labels else int(count)

    def _init_z(self, X: np.ndarray, A: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
        distances = self._weighted_distance(X, A, w); Z = np.zeros_like(distances)
        for i, indices in enumerate(np.argsort(distances, axis=1)[:, :k]):
            values = 1 / (distances[i, indices] + 1e-6); Z[i, indices] = values / values.sum()
        return Z

    @staticmethod
    def _update_f(Z: np.ndarray, n_clusters: int) -> np.ndarray:
        dr, dc = Z.sum(axis=1), Z.sum(axis=0)
        B = Z / np.sqrt(dr[:, None] + 1e-10) / np.sqrt(dc[None, :] + 1e-10)
        U, _, Vt = np.linalg.svd(B, full_matrices=False)
        return np.vstack((U[:, :n_clusters] / np.sqrt(dr[:, None] + 1e-10), Vt[:n_clusters].T / np.sqrt(dc[:, None] + 1e-10)))

    def _update_z(self, X: np.ndarray, A: np.ndarray, F: np.ndarray, w: np.ndarray, lambda_: float, k: int) -> np.ndarray:
        n, m = X.shape[1], A.shape[1]; Fd, Fa = F[:n], F[n:]
        feature_distance = self._weighted_distance(X, A, w)
        spectral_distance = np.maximum(np.sum(Fd**2, axis=1)[:, None] + np.sum(Fa**2, axis=1)[None, :] - 2 * Fd @ Fa.T, 0)
        distances = feature_distance + lambda_ * spectral_distance; Z = np.zeros((n, m))
        for i, indices in enumerate(np.argsort(distances, axis=1)[:, :k]):
            local = distances[i, indices]
            eta = (1 + local.sum() / (2 * self.config.beta)) / k
            values = np.maximum((-local / 2 + self.config.beta * eta) / self.config.beta, 0)
            if values.sum() > 0: Z[i, indices] = values / values.sum()
        return Z

    @staticmethod
    def _update_w(X: np.ndarray, A: np.ndarray, Z: np.ndarray) -> np.ndarray:
        residual = np.empty(X.shape[0])
        for feature in range(X.shape[0]):
            residual[feature] = np.sum((X[feature, :, None] - A[feature, None, :]) ** 2 * Z)
        inverse = 1 / np.maximum(residual, 1e-12)
        return inverse / inverse.sum()

    @staticmethod
    def _objective(X: np.ndarray, A: np.ndarray, Z: np.ndarray, F: np.ndarray, w: np.ndarray, beta: float, lambda_: float) -> float:
        Xw, Aw = X * w[:, None], A * w[:, None]
        term1 = float(np.sum(((Xw.T[:, None, :] - Aw.T[None, :, :]) ** 2).sum(axis=2) * Z))
        dr, dc = Z.sum(axis=1), Z.sum(axis=0); n, m = Z.shape
        laplacian = np.zeros((n + m, n + m)); laplacian[:n, :n] = np.diag(dr); laplacian[:n, n:] = -Z; laplacian[n:, :n] = -Z.T; laplacian[n:, n:] = np.diag(dc)
        return term1 + beta * float(np.sum(Z**2)) + lambda_ * float(np.trace(F.T @ (laplacian + np.eye(n + m) * 1e-8) @ F))

    def fit(self, X: np.ndarray) -> "MGAGC":
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not all(values.shape) or not np.all(np.isfinite(values)):
            raise ValueError("X must be a finite non-empty 2-D array")
        A = self.anchors if self.anchors is not None else generate_anchors(values, self.config.min_points)
        if A.ndim != 2 or A.shape[0] != values.shape[1] or A.shape[1] < 1:
            raise ValueError("anchors must have shape (n_features, n_anchors)")
        data, k = values.T, min(self.config.k, A.shape[1]); w = np.ones(data.shape[0]) / data.shape[0]
        Z = self._init_z(data, A, w, k); F = self._update_f(Z, self.config.n_clusters); lambda_ = self.config.lambda_init
        best_Z, best_w, best_lambda = Z.copy(), w.copy(), lambda_
        best_objective = old_objective = self._objective(data, A, Z, F, w, self.config.beta, lambda_)
        for _ in range(self.config.max_iter):
            w = self._update_w(data, A, Z); Z = self._update_z(data, A, F, w, lambda_, k); F = self._update_f(Z, self.config.n_clusters)
            components = self._components(Z)
            if components == self.config.n_clusters:
                best_Z, best_w, best_lambda = Z.copy(), w.copy(), lambda_; break
            objective = self._objective(data, A, Z, F, w, self.config.beta, lambda_)
            lambda_ = float(np.clip(lambda_ * (2.0 if components < self.config.n_clusters else 0.5), 1e-6, 1e6))
            if objective < best_objective:
                best_objective, best_Z, best_w, best_lambda = objective, Z.copy(), w.copy(), lambda_
            if abs(old_objective - objective) < self.config.tol:
                break
            old_objective = objective
        self.labels_, self.anchors_, self.Z_, self.feature_weights_, self.lambda_final_ = self._components(best_Z, labels=True), A, best_Z, best_w, best_lambda
        return self

    def get_params(self) -> dict[str, object]:
        return asdict(self.config)
