import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.special import i0, i1, logsumexp
from functools import cached_property


# -------------------- Base --------------------
class ExponentialFamily(ABC):
    """Abstract base for exponential-family distributions."""

    @abstractmethod
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        """Return log-density log p(X)."""

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Default pdf via exp(log_pdf)."""
        return np.exp(self.log_pdf(np.asarray(X, dtype=float)))

    @abstractmethod
    def fit_MLE(self, X:np.ndarray, weights: np.ndarray | None):
        pass


# -------------------- Univariate Gaussian --------------------
class UnivariateGaussian(ExponentialFamily):
    """
    N(mean, variance) in natural form:
        θ = ( -μ/σ² , -1/(2σ²) )
    dual / mean-value form:
        η = ( μ , μ² + σ² )
    """

    def __init__(self, mean: float = 0.0, variance: float = 1.0):
        if variance <= 0:
            raise ValueError("variance must be > 0.")
        self.mean = float(mean)
        self.variance = float(variance)
        self._update_params()

    def _update_params(self) -> None:
        self.natural_param = np.array([
            -self.mean / self.variance,
            -1.0 / (2.0 * self.variance)
        ])
        self.dual_param = np.array([
            self.mean,
            self.mean**2 + self.variance
        ])

    # ---- setters ----
    def set_params(self, mean: float, variance: float) -> None:
        if variance <= 0:
            raise ValueError("variance must be > 0.")
        self.mean, self.variance = float(mean), float(variance)
        self._update_params()

    def set_natural_params(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        theta1, theta2 = theta
        if theta2 >= 0:
            raise ValueError("Second natural parameter must be negative.")
        self.mean = -0.5 * theta1 / theta2
        self.variance = -0.5 / theta2
        self._update_params()

    def set_dual_params(self, eta: np.ndarray) -> None:
        eta = np.asarray(eta, dtype=float)
        eta1, eta2 = eta
        var = eta2 - eta1**2
        if var <= 0:
            raise ValueError("eta2 - eta1^2 must be > 0.")
        self.mean = eta1
        self.variance = var
        self._update_params()

    # ---- densities ----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return -0.5 * ((X - self.mean) ** 2) / self.variance - 0.5 * np.log(2 * np.pi * self.variance)

    # pdf inherited from base

    def __repr__(self) -> str:
        return f"UnivariateGaussian(mean={self.mean:.5g}, variance={self.variance:.5g})"

# -------------------- Multivariate Gaussian --------------------
class MultivariateGaussian(ExponentialFamily):
    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        super().__init__()
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self._validate()
        self._cache()

    @property
    def d(self) -> int:
        return self.mean.size

    def _validate(self):
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if self.mean.shape[0] != self.covariance.shape[0]:
            raise ValueError("Mean and covariance shapes mismatch.")
        if not np.allclose(self.covariance, self.covariance.T):
            raise ValueError("Covariance must be symmetric.")
        # Positive definiteness checked in _cache via Cholesky

    def _cache(self):
        try:
            self._chol = np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError as e:
            raise ValueError("Covariance must be positive-definite.") from e
        self._log_det = 2 * np.sum(np.log(np.diag(self._chol)))

    def set_params(self, mean: np.ndarray, covariance: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self._validate()
        self._cache()

    # ----- densities -----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        diff = X - self.mean  # (d,) or (N,d)
        if diff.ndim == 1:  # single data point
            diff = diff[np.newaxis, :]  # (1,d)
        # Solve L y = diff^T => y = L^{-1} diff^T
        y = np.linalg.solve(self._chol, diff.T)  # (d,N)
        quad = np.sum(y * y, axis=0)  # (N,)
        return -0.5 * (self.d * np.log(2 * np.pi) + self._log_det + quad)

    # pdf inherited from base

    # ----- Calibration -----
    def fit_MLE(self, X: np.ndarray, weights: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        if weights is None:
            weights = np.ones(X.shape[0], dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)
        w_sum = weights.sum()
        mu = np.average(X, axis=0, weights=weights)
        diff = X - mu
        # Broadcasting weights to columns; (N,1) * (N,d) -> weighted rows
        weighted_diff = weights[:, np.newaxis] * diff
        cov = weighted_diff.T @ diff / w_sum
        # Numerical jitter if near-singular
        cov += 1e-9 * np.eye(cov.shape[0])

        self.mean = mu
        self.covariance = cov
        self._validate()
        self._cache()

    def __repr__(self):
        return f"MultivariateGaussian(d={self.d})"


class VonMises(ExponentialFamily):
    def __init__(self, loc=0.0, kappa=0.01):
        super().__init__()
        if kappa <= 0:
            raise ValueError("Concentration parameter kappa must be positive.")
        self.loc = float(loc)
        self.kappa = float(kappa)
        self._update_params()

    @staticmethod
    def _mean_length(x):
        return i1(x) / i0(x)

    def _update_params(self):
        self.natural_param = np.array([self.kappa * np.cos(self.loc),
                                       self.kappa * np.sin(self.loc)])
        A = self._mean_length(self.kappa)
        self.dual_param = np.array([A * np.cos(self.loc),
                                    A * np.sin(self.loc)])

    # ----- densities -----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.kappa * np.cos(X - self.loc) - np.log(2 * np.pi * i0(self.kappa))

    # pdf inherited from base

    # ----- Calibration -----
    def fit_MLE(self, X: np.ndarray, weights: np.ndarray | None = None):
        X = np.asarray(X, dtype=float)
        if weights is None:
            weights = np.ones(X.shape[0], dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)
        const = np.log(2 * np.pi)

        def neg_ll(params):
            loc, kappa = params
            if kappa <= 0:
                return np.inf
            ll = np.sum(weights * (kappa * np.cos(X - loc) - np.log(i0(kappa)) - const))
            return -ll  # minimize negative

        C = np.sum(weights * np.cos(X))
        S = np.sum(weights * np.sin(X))
        initials = np.array([np.arctan2(S, C), 1.0])
        bnds = ((-np.pi, np.pi), (1e-8, None))

        result = minimize(
            fun=neg_ll,
            x0=initials,
            method="L-BFGS-B",
            bounds=bnds
        )
        self.loc, self.kappa = result.x
        self._update_params()
        return result

    def __repr__(self):
        return f"VonMises(loc={self.loc:.3f}, kappa={self.kappa:.3f})"


class MixtureModel:
    def __init__(self, components: list[ExponentialFamily], weights: np.ndarray):
        if len(components) != weights.size:
            raise ValueError("Components and weights mismatch.")
        if not np.isclose(weights.sum(), 1):
            raise ValueError("Weights must sum to 1.")
        self.components = components
        self.weights = weights

    def log_pdf_components(self, X: np.ndarray) -> np.ndarray:
        """
        Returns log p(x_i | k) for all i,k
        Shape: (N, K)
        """
        X = np.asarray(X, dtype=float)
        return np.column_stack([c.log_pdf(X) for c in self.components])

    def pdf(self, x):
        return sum(w * c.pdf(x) for w, c in zip(self.weights, self.components))

    def fit_classical_EM(self, X, tol=1e-6, max_iter=200, verbose=False):
        X = np.asarray(X)
        N = X.shape[0]
        K = len(self.components)

        logger = []
        for it in range(max_iter):
            # E-step: Compute the posterior
            # take log to prevent underflow
            log_prior = np.log(self.weights)       # (K,)
            log_p = self.log_pdf_components(X)     # (N, K)
            log_numerator =  log_prior +  log_p    # (N, K)
            log_denominator = logsumexp(log_numerator, axis=1, keepdims=True)  # (N, 1)
            log_posterior = log_numerator - log_denominator    # (N, K)
            posterior = np.exp(log_posterior) # responsibilities (N, K)

            ll = np.sum(log_p * posterior)
            logger.append(ll)

            # M-step: Maximize weighted log-likelihood
            # update priors
            self.weights = posterior.sum(axis=0) / N # effective counts (K,)
            # update distribution parameters
            for k, comp in enumerate(self.components):
                comp.fit_MLE(X, posterior[:, k])

            # ---------- Check convergence ----------
            if it > 0 and abs(logger[-1] - logger[-2]) < tol:
                if verbose:
                    print(f"Converged at iter {it}: Delta LL={logger[-1] - logger[-2]:.3e}")
                break
        else:
            if verbose:
                print("Reached max_iter without full convergence.")
        return logger