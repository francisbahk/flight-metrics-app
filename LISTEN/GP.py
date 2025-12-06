import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from typing import Optional

class GaussianProcessUtilityModel:
    """GP utility model u(x) ~ GP for pairwise comparisons."""

    def __init__(self, kernel=None, rng: Optional[np.random.Generator] = None):
        if kernel is None:
            kernel = C(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self._trained = False
        self._rng = rng or np.random.default_rng()
        self.X_train = None
        self.y_train = None

    def fit_on_duels(self, X_delta: np.ndarray, y01: np.ndarray) -> None:
        """Fit GP on pairwise comparisons (regress delta -> outcome)."""
        print("USING GP ")
        if X_delta.size == 0:
            return
        if X_delta.ndim == 1:
            X_delta = X_delta.reshape(-1, 1)
        y01 = y01.ravel()
        # If all labels identical, flip one for GP stability
        if y01.size >= 2 and (np.all(y01 == 0) or np.all(y01 == 1)):
            idx = int(np.argmax(np.linalg.norm(X_delta, axis=1)))
            y01 = y01.copy()
            X_delta = X_delta.copy()
            y01[idx] = 1 - y01[idx]
            X_delta[idx] = -X_delta[idx]
        if np.unique(y01).size < 2:
            return
        self.gp.fit(X_delta, y01)
        self.X_train = X_delta
        self.y_train = y01
        self._trained = True

    def posterior_mean_util(self, X: np.ndarray) -> np.ndarray:
        """Posterior mean utility for each item."""
        if not self._trained:
            return np.zeros(X.shape[0])
        mean, _ = self.gp.predict(X, return_std=True)
        return mean

    def posterior_std_util(self, X: np.ndarray) -> np.ndarray:
        """Posterior std of utility for each item."""
        if not self._trained:
            return np.ones(X.shape[0])
        _, std = self.gp.predict(X, return_std=True)
        return std

    def _sample_thetas(self, n_samples: int) -> np.ndarray:
        """Sample utility functions from GP posterior (approximate)."""
        if not self._trained:
            return np.zeros((n_samples, 1))
        # Sample utilities at training points
        mean, cov = self.gp.predict(self.X_train, return_cov=True)
        samples = self._rng.multivariate_normal(mean, cov, size=n_samples)
        return samples

    def sample_utilities(self, X: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample utilities from GP posterior at X."""
        print('USING GP ')
        if not self._trained:
            return np.zeros((X.shape[0], n_samples))
        mean, cov = self.gp.predict(X, return_cov=True)
        samples = self._rng.multivariate_normal(mean, cov, size=n_samples)
        return samples.T  # (n_items, n_samples)

    def predict_proba(self, winner_idx: int, challenger_idx: int, feat: np.ndarray) -> float:
        """P(challenger beats winner) under GP model."""
        if not self._trained:
            return 0.5
        u_winner = self.posterior_mean_util(feat[winner_idx].reshape(1, -1))[0]
        u_challenger = self.posterior_mean_util(feat[challenger_idx].reshape(1, -1))[0]
        # Use logistic link for probability
        from scipy.special import expit
        return float(expit(u_challenger - u_winner))

    def predict_proba_delta(self, delta: np.ndarray) -> float:
        """P(A beats B) given delta = features(A) - features(B)"""
        if not self._trained:
            return 0.5
        from scipy.special import expit
        mean = self.posterior_mean_util(delta.reshape(1, -1))[0]
        return float(expit(mean))

    def compute_score(self, winner_idx: int, challenger_idx: int, feat: np.ndarray, 
                     acquisition: str = "eubo", n_samples: int = 100) -> float:
        """Compute acquisition score for comparing challenger to winner."""
        if not self._trained:
            return self._rng.random()
        x_winner = feat[winner_idx].reshape(1, -1)
        x_challenger = feat[challenger_idx].reshape(1, -1)
        u_winner_mean = self.posterior_mean_util(x_winner)[0]
        u_challenger_mean = self.posterior_mean_util(x_challenger)[0]
        if acquisition == "eubo":
            u_samples = self.sample_utilities(feat[[winner_idx, challenger_idx]], n_samples)
            p_challenger_best = np.mean(u_samples[1] > u_samples[0])
            score = p_challenger_best * u_challenger_mean + (1 - p_challenger_best) * u_winner_mean
        elif acquisition == "ucb":
            u_challenger_std = self.posterior_std_util(x_challenger)[0]
            beta = 2.0
            score = u_challenger_mean + beta * u_challenger_std
        elif acquisition == "thompson":
            theta_sample = self._sample_thetas(1)[0]
            score = theta_sample[challenger_idx]
        elif acquisition == "info_gain":
            p_win = self.predict_proba(winner_idx, challenger_idx, feat)
            entropy = -p_win * np.log(p_win + 1e-10) - (1-p_win) * np.log(1-p_win + 1e-10)
            score = entropy * (1 + u_challenger_mean)
        elif acquisition == "random":
            score = self._rng.random()
        else:
            score = u_challenger_mean
        return float(score)

    def ready(self) -> bool:
        return self._trained