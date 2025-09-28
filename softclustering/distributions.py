# distributions.py
"""
Distributions for exponential family models.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.special import i0e, i1e
from typing import Tuple, Optional

_EPS = 1e-9


# -------------------- Base --------------------
class ExponentialFamily(ABC):
    """Abstract base for exponential-family distributions."""

    # ---- Densities ----
    @abstractmethod
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        """Return log-density log p(X)."""
        pass

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Default pdf via exp(log_pdf)."""
        return np.exp(self.log_pdf(np.asarray(X, dtype=float)))

    # ---- Getters and Setter ----
    @property
    @abstractmethod
    def params(self):
        """ Get parameters in the ordinary form."""
        pass

    @property
    @abstractmethod
    def dual_param(self):
        """ Get dual parameter vector. Mean of sufficient statistic vector."""
        pass

    # ---- Calibration ----
    @abstractmethod
    def fit(self, X: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            case: Optional[str] = None):
        pass

    # ---- Utility methods ----
    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        if np.sum(w) == 0:
            raise ValueError("All supplied weights are zero.")
        return w / w.sum()

    def _input_process(self, X: np.ndarray,
                       weights: Optional[np.ndarray] = None,
                       ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        if weights is None:
            weights = np.ones(X.shape[0], dtype=float) / X.shape[0]
        else:
            weights = self._normalize_weights(weights)
        return X, weights


# -------------------- Univariate Gaussian --------------------
class UnivariateGaussian(ExponentialFamily):
    """
    N(mean, variance) in natural form:
        theta = ( -mu/sigma^2, -1/(2*sigma^2) )
    dual / mean-value form:
        eta =   ( mu , mu^2 + sigma^2 )
    """

    def __init__(self, mean: float = 0.0, variance: float = 1.0):
        self._mean = float(mean)
        self._variance = float(variance)
        self._natural_param = None
        self._dual_param = None
        self._validate()
        self._update_params()

    def _validate(self) -> None:
        if self._variance <= 0:
            raise ValueError("variance must be > 0.")
        if self._natural_param is not None and self._natural_param[1] >= 0:
            raise ValueError("Second natural parameter must be negative.")
        if self._dual_param is not None and self._dual_param[1] <= self._dual_param[0] ** 2:
            raise ValueError("eta2 - eta1^2 must be > 0.")

    def _update_params(self) -> None:
        self._natural_param = np.array([
            -self._mean / self._variance,
            -1.0 / (2.0 * self._variance)
        ])
        self._dual_param = np.array([
            self._mean,
            self._mean ** 2 + self._variance
        ])

    # ---- Getters and Setters ----
    @property
    def params(self) -> Tuple[float, float]:
        return self._mean, self._variance

    @params.setter
    def params(self, value: Tuple[float, float]):
        self._mean, self._variance = value
        self._validate()
        self._update_params()

    @property
    def natural_param(self):
        return self._natural_param.copy()

    @natural_param.setter
    def natural_param(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
            raise ValueError("natural_param must be a length-2 vector.")
        self._natural_param = theta
        self._mean = -0.5 * theta[0] / theta[1]
        self._variance = -0.5 / theta[1]
        self._validate()
        self._update_params()

    @property
    def dual_param(self):
        return self._dual_param.copy()

    @dual_param.setter
    def dual_param(self, eta: np.ndarray) -> None:
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (2,):
            raise ValueError("dual_param must be a length-2 vector.")
        self._dual_param = eta
        self._mean = eta[0]
        self._variance = eta[1] - eta[0] ** 2
        self._validate()
        self._update_params()

    @staticmethod
    def from_dual_to_ordinary(eta: np.ndarray) -> Tuple[float, float]:
        return float(eta[0]), float(eta[1] - eta[0] ** 2)

    # ---- densities ----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return -0.5 * ((X - self._mean) ** 2) / self._variance - 0.5 * np.log(2 * np.pi * self._variance)

    # pdf inherited from base

    # ---- Calibration ----
    def fit(self,
            X: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            case: str = "classic",
            ) -> None:

        X, sample_weight = self._input_process(X, sample_weight)
        match case:
            case "bregman":
                # compute dual/expectation parameters using sufficient statistics.
                eta = np.array([np.average(X, weights=sample_weight),
                                np.average(X ** 2, weights=sample_weight)])
                self.dual_param = eta
            case _:
                mu = np.average(X, weights=sample_weight)
                diff = X - mu
                variance = np.inner(sample_weight * diff, diff)

                self._mean = mu
                self._variance = variance
                self._validate()
                self._update_params()

    def __repr__(self) -> str:
        return f"UnivariateGaussian(mean={self._mean:.3f}, variance={self._variance:.3f})"


# -------------------- Multivariate Gaussian --------------------
class MultivariateGaussian(ExponentialFamily):
    """
    Multivariate Gaussian N(mu, sigma), with mu = mean vector and sigma = covariance matrix
    """

    def __init__(self, mean: Optional[np.ndarray] = None,
                 covariance: Optional[np.ndarray] = None):
        super().__init__()
        self._mean = np.zeros(2) if mean is None else np.asarray(mean, dtype=float)
        self._covariance = np.eye(self._mean.size) if covariance is None else np.asarray(covariance, dtype=float)
        self._validate()
        self._cache()

    def _validate(self):
        if self._covariance.shape[0] != self._covariance.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if self._mean.shape[0] != self._covariance.shape[0]:
            raise ValueError("Mean and covariance shapes mismatch.")
        if not np.allclose(self._covariance, self._covariance.T):
            raise ValueError("Covariance must be symmetric.")
        # Positive definiteness checked in _cache via Cholesky

    def _cache(self):
        try:
            self._chol = np.linalg.cholesky(self._covariance)
        except np.linalg.LinAlgError as e:
            raise ValueError("Covariance must be positive-definite.") from e
        self._log_det = 2 * np.sum(np.log(np.diag(self._chol)))

    # ---- Getters and Setter ----
    @property
    def d(self) -> int:
        return self._mean.size

    @property
    def params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._mean, self._covariance

    @params.setter
    def params(self, value):
        mean, covariance = value
        self._mean = np.asarray(mean, dtype=float)
        self._covariance = np.asarray(covariance, dtype=float)
        self._validate()
        self._cache()

    @property
    def natural_param(self):
        theta_1 = np.linalg.solve(self._covariance, self._mean)
        theta_2 = -0.5 * np.linalg.inv(self._covariance)
        return np.concatenate([theta_1, theta_2.ravel()])

    @natural_param.setter
    def natural_param(self, theta: np.ndarray):
        raise NotImplementedError("Setting natural_param is not implemented.")

    @property
    def dual_param(self) -> np.ndarray:
        mu = self._mean
        second_moment = (self._covariance + np.outer(mu, mu)).flatten()
        return np.concatenate([mu, second_moment.ravel()])

    @dual_param.setter
    def dual_param(self, eta: np.ndarray):
        d = self.d
        mu = eta[:d]  # E[x]
        second_moments = eta[d:].reshape((d, d))  # E[x x^T]
        cov = second_moments - np.outer(mu, mu)  # covariance = E[x x^T] – mu mu^T
        cov = 0.5 * (cov + cov.swapaxes(-1, -2))  # ensure symmetric matrix
        cov += _EPS * np.eye(cov.shape[0])  # Numerical jitter if near-singular
        self._mean, self._covariance = mu, cov

    @staticmethod
    def get_sufficient_stat(X: np.ndarray) -> np.ndarray:
        """
        Get the sufficient statistic vector e.g. case d=2: [x y x^2 xy yx y^2]
        :return: array of shape (N,d+d^2)
        """
        N = X.shape[0]
        d = X.shape[1]
        outer = np.einsum('ij,ik->ijk', X, X)  # (N,d,d)
        return np.concatenate([X, outer.reshape(N, d ** 2)], axis=1)

    @staticmethod
    def from_dual_to_ordinary(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized conversion from dual parameters to (mean, covariance).
        Returns:
            mu: array of shape (n, d)
            cov: array of shape (n, d, d)
        """
        eta = np.asarray(eta, dtype=float)
        if eta.ndim == 1:
            eta = eta[None, :]  # promote single vector to batch

        n, L = eta.shape
        # solve d from L = d + d^2  => d = (-1 + sqrt(1+4L)) / 2
        d = int(-0.5 + 0.5 * np.sqrt(1 + 4 * L))
        if d + d * d != L:
            raise ValueError(f"Invalid eta length {L}; cannot infer integer d.")

        mu = eta[:, :d]  # (n, d)
        second_moments = eta[:, d:].reshape(n, d, d)  # (n, d, d)
        # covariance = E[xx^T] - mu mu^T, broadcasting outer product
        cov = second_moments - mu[:, :, None] * mu[:, None, :]  # (n, d, d)
        cov = 0.5 * (cov + cov.swapaxes(-1, -2))  # ensure symmetric matrix
        cov = cov + _EPS * np.eye(d)

        return mu, cov

    # ----- densities -----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        diff = X - self._mean  # (d,) or (N,d)
        if diff.ndim == 1:  # single data point
            diff = diff[np.newaxis, :]  # (1,d)
        # Solve L y = diff^T => y = L^{-1} diff^T
        y = np.linalg.solve(self._chol, diff.T)  # (d,N)
        quad = np.sum(y * y, axis=0)  # (N,)
        return -0.5 * (self.d * np.log(2 * np.pi) + self._log_det + quad)

    # pdf inherited from base

    # ----- Calibration -----
    def fit(self,
            X: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            case: str = "classic",
            ) -> None:
        X, sample_weight = self._input_process(X, sample_weight)
        match case:
            case "bregman":
                # Compute MLE via minimization of Bregman divergence
                # form sufficient stats and average
                suf_stat = self.get_sufficient_stat(X)  # shape (N, d + d^2)
                dual = np.average(suf_stat, axis=0, weights=sample_weight)  # length d + d^2
                # update params
                self.dual_param = dual
                self._validate()
                self._cache()
            case _:
                # Compute MLE via analytical solution of ordinary-coordinates parameters
                mu = np.average(X, axis=0, weights=sample_weight)
                diff = X - mu
                # Broadcasting weights to columns; (N,1) * (N,d) -> weighted rows
                weighted_diff = sample_weight[:, np.newaxis] * diff
                cov = weighted_diff.T @ diff
                cov += _EPS * np.eye(cov.shape[0])  # Numerical jitter if near-singular

                self._mean = mu
                self._covariance = cov
                self._validate()
                self._cache()

    def __repr__(self):
        mean_str = np.array2string(self._mean, precision=3, separator=' ', suppress_small=True)
        cov_rows = [np.array2string(row, precision=3, separator=' ', suppress_small=True) for row in self._covariance]
        cov_str = '[' + ', '.join(cov_rows) + ']'
        return f"MultivariateGaussian(d={self.d}, mean={mean_str}, cov={cov_str})"


# -------------------- Von Mises --------------------
class VonMises(ExponentialFamily):
    """
    Von Mises distribution for directional data.
    X must be in sufficient statistic form: [cos x, sin x].
    """

    def __init__(self, loc: float = 0.0, kappa: float = 1.0):
        super().__init__()
        self._loc = float(loc)
        self._kappa = float(kappa)
        self._natural_param = None
        self._dual_param = None
        self._A = None
        self._MAX_KAPPA = 50.0
        self._MAX_A = self._mean_length(self._MAX_KAPPA)  # A(50) = 0.9899489673784978
        self._validate()
        self._update_params()

    def _mean_length(self, kappa):
        """
        Mean resultant length define as i1(k)/i0(k) where i1 and i0 are the modified Bessel
        functions of first kind of order 1 and 0, respectively
        """
        kappa = np.clip(kappa, 1e-6, self._MAX_KAPPA)
        return i1e(kappa) / i0e(kappa)

    @staticmethod
    def _inv_mean_length(r: float):
        """
        A^{-1} approximation given by Best and Fisher (1981).
        """
        if r < 0.53:
            return 2 * r + r ** 3 + (5 * r ** 5) / 6
        elif r < 0.85:
            return -0.4 + 1.39 * r + 0.43 / (1 - r)
        else:
            return 1 / (r ** 3 - 4 * r ** 2 + 3 * r)

    @staticmethod
    def _inv_mean_length_v2(r: float):
        """
        A^{-1} approximation given by Banerjee (2005).
        """
        return r * (2 - r ** 2) / (1 - r ** 2)

    def _validate(self):
        if self._kappa <= 0:
            raise ValueError("Concentration parameter kappa must be positive.")

    def _update_params(self):
        self._natural_param = np.array([self._kappa * np.cos(self._loc),
                                        self._kappa * np.sin(self._loc)])
        self._A = self._mean_length(self._kappa)
        self._dual_param = np.array([self._A * np.cos(self._loc),
                                     self._A * np.sin(self._loc)])

    # ---- Getters and Setters ----
    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, value):
        self._loc = float(value)
        self._update_params()

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        self._kappa = value
        self._validate()
        self._update_params()

    @property
    def params(self) -> Tuple[float, float]:
        return self._loc, self._kappa

    @property
    def mean_length(self):
        return self._A

    @property
    def natural_param(self) -> np.ndarray:
        return self._natural_param.copy()

    @natural_param.setter
    def natural_param(self, theta: np.ndarray):
        if theta.shape != (2,):
            raise ValueError("natural_param must be a length-2 vector.")
        self._natural_param = theta
        self._loc = np.arctan2(theta[1], theta[0])
        self._kappa = np.minimum(np.linalg.norm(theta, ord=None), self._MAX_KAPPA)
        self._A = self._mean_length(self._kappa)
        self._validate()
        self._dual_param = np.array([self._A * np.cos(self._loc),
                                     self._A * np.sin(self._loc)])

    @property
    def dual_param(self) -> np.ndarray:
        return self._dual_param.copy()

    @dual_param.setter
    def dual_param(self, eta: np.ndarray):
        self._dual_param = eta
        self._loc = np.arctan2(eta[1], eta[0])
        self._A = np.minimum(np.linalg.norm(eta, ord=None), self._MAX_A)
        self._kappa = self._inv_mean_length(self._A)
        self._validate()
        self._natural_param = np.array([self._kappa * np.cos(self._loc),
                                        self._kappa * np.sin(self._loc)])

    @staticmethod
    def from_dual_to_ordinary(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert dual params eta = [eta1, eta2] to (loc, kappa) for one or many etas.
        If eta has shape (2,), returns array([loc, kappa]).
        If eta has shape (n,2), returns shape (n,2) with each row [loc, kappa].
        """
        eta = np.asarray(eta, float)
        single = (eta.ndim == 1)
        if single:
            eta = eta[np.newaxis, :]

        loc = np.arctan2(eta[:, 1], eta[:, 0])  # shape (n,)
        A = np.minimum(np.linalg.norm(eta, axis=1), 0.9899489673784978)  # A(50) = 0.9899489673784978

        # vectorized Best–Fisher inversion:
        #   if A<0.53: 2A + A^3 + 5A^5/6
        #   elif A<0.85: -0.4 + 1.39A + 0.43/(1−A)
        #   else: 1/(A^3 − 4A^2 + 3A)
        kappa = np.where(  # shape (n,)
            A < 0.53,
            2 * A + A ** 3 + (5 * A ** 5) / 6,
            np.where(
                A < 0.85,
                -0.4 + 1.39 * A + 0.43 / (1 - A),
                1.0 / (A ** 3 - 4 * A ** 2 + 3 * A)
            )
        )

        return loc, kappa  # shape (n,) and (n,)

    # ----- densities -----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_partition = np.log(2 * np.pi * i0e(self._kappa)) + self._kappa
        return X @ self._natural_param - log_partition

    # pdf inherited from base

    # ----- Calibration -----
    def fit(self,
            X: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            case: str = "classic",
            ) -> None:

        X, sample_weight = self._input_process(X, sample_weight)
        match case:
            case "bregman":
                # Compute dual/expectation parameters using sufficient statistics.
                eta = np.average(X, axis=0, weights=sample_weight)
                # self.dual_param = eta
                loc = np.arctan2(eta[1], eta[0])
                R = np.minimum(np.linalg.norm(eta, ord=None), self._MAX_A)
                kappa = self._inv_mean_length(R)

                self._loc, self._kappa = loc, kappa
                self._validate()
                self._update_params()
            case "approximation":
                eta = np.average(X, axis=0, weights=sample_weight)
                loc = np.arctan2(eta[1], eta[0])
                R = np.minimum(np.linalg.norm(eta, ord=None), self._MAX_A)
                kappa = self._inv_mean_length_v2(R)

                self._loc, self._kappa = loc, kappa
                self._validate()
                self._update_params()

            case _:
                # Compute MLE with numerical optimizer
                const = np.log(2 * np.pi)

                def neg_ll(params):
                    loc, kappa = params
                    if kappa <= 0:
                        return np.inf
                    # i0e(kappa) = exp(-kappa)*i0(kappa)
                    # -log(i0(kappa)) = -log(i0e(kappa)) - kappa
                    ll = np.sum(sample_weight * (kappa * (np.cos(loc) * X[:, 0]
                                                          + np.sin(loc) * X[:, 1])
                                                 - np.log(i0e(kappa)) - kappa - const))
                    return -ll  # minimize negative

                C = np.sum(sample_weight * X[:, 0])
                S = np.sum(sample_weight * X[:, 1])
                initials = np.array([np.arctan2(S, C), self._kappa])
                bnds = ((-np.pi, np.pi), (1e-6, 50.0))
                result = minimize(
                    fun=neg_ll,
                    x0=initials,
                    method="L-BFGS-B",
                    bounds=bnds
                )
                self._loc, self._kappa = result.x
                self._validate()
                self._update_params()

    def __repr__(self):
        return f"VonMises(loc={self._loc * 180 / np.pi:.1f} deg, kappa={self._kappa:.3f})"


# ----- Custom Bregman Distribution -----
class CustomBregman(ExponentialFamily):
    """
    p(x) = Gauss(x|theta_gaus)^alpha * VonMises(x|theta_vm)^beta
    """

    def __init__(self, coef_gaus: float, coef_vm: float):
        super().__init__()
        self._vonmises = VonMises()
        self._gaussian = MultivariateGaussian()
        self._coef_gauss = coef_gaus
        self._coef_vm = coef_vm

    @property
    def params(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], float, float]:
        return self._gaussian.params, self._vonmises.params, self._coef_gauss, self._coef_vm

    @property
    def dual_param(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        return self._gaussian.dual_param, self._vonmises.dual_param, self._coef_gauss, self._coef_vm

    def log_pdf(self, X: np.ndarray):
        #TODO: Need to normalize this pdf to integrate one.
        X_gauss = np.asarray(X, dtype=float)[:, :2]
        X_vm = np.asarray(X, dtype=float)[:, 2:]
        log_gauss = self._gaussian.log_pdf(X_gauss)
        log_vm = self._vonmises.log_pdf(X_vm)
        return self._coef_gauss * log_gauss + self._coef_vm * log_vm

    def fit(self,
            X: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            case: Optional[str] = None,
            ) -> None:
        X_gauss = np.asarray(X, dtype=float)[:, :2]
        X_vm = np.asarray(X, dtype=float)[:, 2:]
        self._gaussian.fit(X_gauss, sample_weight=sample_weight, case=case)
        self._vonmises.fit(X_vm, sample_weight=sample_weight, case=case)

    @property
    def gaussian(self):
        return self._gaussian

    @property
    def vonmises(self):
        return self._vonmises
