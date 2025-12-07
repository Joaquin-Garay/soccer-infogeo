# mixture.py

"""
Mixture model.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Sequence, Tuple

from .distributions import ExponentialFamily
from scipy.special import logsumexp
from sklearn.cluster import KMeans, kmeans_plusplus

_EPS = 1e-9


# -------------------- Mixture Model --------------------
class MixtureModel:
    def __init__(self, components: list[ExponentialFamily],
                 weights: Optional[np.ndarray] = None,
                 init: Optional[str] = None,
                 rng: Optional[int] = None,
                 ):

        self._components = components
        self._rng = np.random.RandomState(42) if rng is None else np.random.RandomState(rng)

        allowed = {"k-means++", "k-means", "random_from_data", "random"}
        if init is None:
            self._init = "k-means++"
        elif init in allowed:
            self._init = init
        else:
            raise ValueError(f"init must be one of {sorted(allowed)}")

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.ndim != 1 or weights.size != self.n_components:
                raise ValueError("Components and weights mismatch.")
            if np.any(weights <= 0):
                raise ValueError("All weights must be > 0.")
            self._weights = weights / weights.sum()
            self._is_initialized = True
        else:
            self._weights = None
            self._is_initialized = False

    def _initialize(self,
                    X: np.ndarray,
                    sample_weight: np.ndarray,
                    ) -> None:

        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        eps = 10 * np.finfo(X.dtype).eps

        # Build the posterior matrix for the chosen strategy
        post = self._build_posteriors(X, sample_weight, self._rng, n_samples)

        # Fallback: if any cluster got zero responsibility, switch to
        #    fully random responsibilities for that run
        if (post.sum(axis=0) == 0).any():
            post = self._rng.random((n_samples, self.n_components))

        # Fit components & weights once
        self._fit_components_from_posteriors(X, post, sample_weight, eps)
        self._is_initialized = True

    def _build_posteriors(self,
                          X: np.ndarray,
                          sample_weight: np.ndarray,
                          rng: np.random.RandomState,
                          n_samples: int,
                          ) -> np.ndarray:
        """Return an (n_samples × K) responsibility matrix for the init method."""
        post = np.zeros((n_samples, self.n_components), dtype=float)
        match self._init:
            case "k-means++":
                _, idx = kmeans_plusplus(
                    X, self.n_components, random_state=rng, sample_weight=sample_weight
                )
                post[idx, np.arange(self.n_components)] = 1.0
                return post

            case "k-means":
                labels = KMeans(
                    n_clusters=self.n_components,
                    init="random",
                    n_init=10,
                    max_iter=10,
                    random_state=rng,
                    # sample_weight=sample_weight,
                ).fit_predict(X)
                post[np.arange(n_samples), labels] = 1.0  # as in hard clustering
                return post

            case "random_from_data":
                idx = rng.choice(n_samples, self.n_components, replace=False)
                post[idx, np.arange(self.n_components)] = 1.0
                return post

            case "random":
                # Empty -> triggers fallback to random responsibilities
                return post

            case _:
                raise ValueError(f"Unknown init method: {self._init!r}")

    def _fit_components_from_posteriors(self,
                                        X: np.ndarray,
                                        post: np.ndarray,
                                        sample_weight: np.ndarray,
                                        eps: float,
                                        ) -> None:
        """Fit each component and update mixture weights once."""
        for j, dist in enumerate(self._components):
            dist.fit(X, sample_weight=post[:, j] * sample_weight)

        self._weights = post.sum(axis=0) + eps
        self._weights /= self._weights.sum()

    # ---- Getter and Setters ----
    @property
    def weights(self):
        if self._weights is None:
            raise RuntimeError("Mixture weights are not initialized yet.")
        return self._weights.copy()

    @property
    def components(self):
        return self._components

    @property
    def n_components(self) -> int:
        return len(self._components)

    @property
    def init(self):
        return self._init

    def get_posteriors(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        post, _, _, _ = self._e_step(X)
        return post

    def get_data_ll(self, X: np.ndarray):
        """
        Data Log-likelihood
        """
        X = np.asarray(X, dtype=float)
        _, log_likelihood, _, _ = self._e_step(X)
        return log_likelihood

    def get_expected_ll(self, X: np.ndarray):
        """
        Expected Complete-Data Log-likelihood (EM Q function)
        """
        X = np.asarray(X, dtype=float)
        _, _, expected_log_likelihood, _ = self._e_step(X)
        return expected_log_likelihood

    # ---- Densities ----
    def log_pdf_components(self, X: np.ndarray) -> np.ndarray:
        """
        Returns log p(x_i | k) for all i,k
        Shape: (N, K)
        """
        X = np.asarray(X, dtype=float)
        return np.column_stack([c.log_pdf(X) for c in self._components])

    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Return log p(x)
        Shape: (N,)
        """
        X = np.asarray(X, dtype=float)
        log_pi = np.log(self._weights)  # (K,)
        return logsumexp(self.log_pdf_components(X) + log_pi, axis=1)

    def pdf(self, X):
        return np.exp(self.log_pdf(X))

    # ---- Expectation Maximization Algorithm ----
    def _e_step(self,
                X: np.ndarray,
                ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """
        Compute the posterior_{ij} as prior_j * exp_family(x_i|theta_j) / sum_k (prior_k * exp_family(x_i|theta_k))
        Compute the expected complete-data log-likelihood (EM-Algorithm Q function) as sum_i sum_j (log prior_j + log exp_family(x_i|theta_j)) * posterior_{ij}
        Compute the data log likelihood as sum_i log(sum_j prior_j exp_family(x_i|theta_j))
        """
        # take log to prevent underflow
        eps = np.finfo(float).tiny
        log_prior = np.log(self._weights + eps)  # (K,) := log prior_j
        log_p = self.log_pdf_components(X)  # (N, K) := log exp_family(x_i|theta_j)
        log_numerator = log_prior + log_p  # (N, K) := log prior_j + log exp_family(x_i|theta_j)
        log_denominator = logsumexp(log_numerator, axis=1,
                                    keepdims=True)  # (N, 1) := log (sum_k (prior_k * exp_family(x_i|theta_k)) ) = log p(x_i)
        log_posterior = log_numerator - log_denominator  # (N, K) := log(posterior_{ij})
        posterior = np.exp(log_posterior)  # responsibilities (N, K)

        log_likelihood = float(
            log_denominator.sum())  # data log likelihood as sum_i log(sum_j prior_j exp_family(x_i|theta_j))

        expected_log_likelihood = float(np.sum(log_numerator * posterior))  # expected complete-data log-likelihood

        return posterior, log_likelihood, expected_log_likelihood, log_denominator.flatten()

    def fit(self,
            X: np.ndarray,
            sample_weight: Sequence[float] = None,
            tol: float = 1e-4,
            max_iter: int = 1000,
            verbose: bool = False,
            m_step: str = "classic",
            c_step: bool = False,
            ) -> Tuple[Sequence[float], int]:
        """
        Perform the Expectation-Maximization algorithm to fit a mixture model.
        It stops as soon as the absolute difference between two iterations is below the tolerance.
        """
        X = np.asarray(X, dtype=float)
        n_obs = X.shape[0]

        if sample_weight is None:
            sample_weight = np.ones(n_obs) / n_obs
        if not self._is_initialized:
            self._initialize(X, sample_weight)

        logger = []
        it = 0
        for it in range(max_iter):

            # E-step: Compute the posterior
            posterior, _, _, log_likelihood_arr = self._e_step(X)
            logger.append(float(np.dot(sample_weight, log_likelihood_arr)))  # sample-weighted data log likelihood

            # C-step: One-hot encoding of posterior matrix
            if c_step:
                idx = np.argmax(posterior, axis=1)  # shape (N,)
                one_hot = np.zeros_like(posterior, dtype=float)
                one_hot[np.arange(posterior.shape[0]), idx] = 1.0
                if np.any(one_hot.sum(axis=0) == 0):
                    # there is an empty cluster
                    if verbose:
                        empty = np.where(one_hot.sum(axis=0) == 0)[0]
                        print(f"Empty clusters: {empty} ")
                    break
                else:
                    posterior = one_hot

            # M-step: Maximize sample-weighted data log likelihood
            # update priors
            self._weights = np.average(posterior, axis=0, weights=sample_weight)  # (K,)
            # lift the priors when one of them is below 1 basis points
            if np.min(self._weights) <= _EPS:
                if verbose:
                    print(f"lifting priors...")
                self._weights = (self._weights + _EPS) / (1 + self.n_components * _EPS)
            # update distribution parameters
            for j, comp in enumerate(self._components):
                comp.fit(X, sample_weight=sample_weight * posterior[:, j], case=m_step)

            # check convergence
            if it > 10 and abs(logger[-1] - logger[-2]) < tol:
                if verbose:
                    print(f"Converged at iter {it}: LL={logger[-1]:.2f}, Delta LL={logger[-1] - logger[-2]:.2e}")
                break
        else:
            if verbose:
                print("Reached max_iter without full convergence.")

        return logger, it

    # ---- Display ----
    @staticmethod
    def _format_component(idx: int, w: float | None, comp) -> str:
        w_str = f"{w:0.3f}" if w is not None else "—"
        return f"  ├─ ({idx}) w={w_str}  {comp!r}"

    def __repr__(self) -> str:
        header = f"{self.__class__.__name__}(n_components={self.n_components})"
        if self._components is None:
            return header + "  [no components]"

        lines = [
            self._format_component(j,
                                   None if self._weights is None else self._weights[j],
                                   comp)
            for j, comp in enumerate(self._components)
        ]
        # Use a unicode corner for the last line
        if lines:
            lines[-1] = lines[-1].replace("├─", "└─", 1)
        return "\n".join([header, *lines])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for get_posteriors(X)."""
        return self.get_posteriors(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard labels via argmax of posterior responsibilities."""
        return np.argmax(self.get_posteriors(X), axis=1)

    def score(self, X: np.ndarray) -> float:
        """Average log-likelihood per sample (sklearn-style)."""
        X = np.asarray(X, dtype=float)
        return float(self.log_pdf(X).mean())
