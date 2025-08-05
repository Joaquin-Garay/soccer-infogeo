import numpy as np
import warnings
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.special import i0e, i1e, logsumexp
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor

# -------------------- Base --------------------
class ExponentialFamily(ABC):
    """Abstract base for exponential-family distributions."""

    # ---- Densities ----
    @abstractmethod
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        """Return log-density log p(X)."""

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
        """ Get dual parameter vector. Mean of sufficient statistc vector."""
        pass

    # ---- Calibration ----
    @abstractmethod
    def fit_with_mle(self, X: np.ndarray, weights: np.ndarray | None = None):
        pass

    @abstractmethod
    def fit_with_min_bregman(self, X: np.ndarray, weights: np.ndarray | None = None):
        pass

    # ---- Utility methods ----
    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        if w.sum() == 0:
            raise ValueError("All supplied weights are zero.")
        return w / w.sum()

    def _input_process(self, X: np.ndarray, weights: np.ndarray | None = None):
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
    def params(self) -> np.ndarray:
        return np.array([self._mean, self._variance]).copy()

    @params.setter
    def params(self, mean: float, variance: float):
        self._mean, self._variance = float(mean), float(variance)
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
    def from_dual_to_ordinary(eta: np.ndarray) -> list[float]:
        return [float(eta[0]), float(eta[1] - eta[0] ** 2)]

    # ---- densities ----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return -0.5 * ((X - self._mean) ** 2) / self._variance - 0.5 * np.log(2 * np.pi * self._variance)

    # pdf inherited from base

    @staticmethod
    def kl_div(mean, var, mean_0, var_0):
        """
        Compute the KL divergence of a Univariate Gaussian distribution.
        KL[N(mu, var) : N(mu0, var0)] = D_phi[eta : eta_0]
        """
        return 0.5 * (np.log(var_0 / var) + (var + (mean - mean_0) ** 2) / var_0 - 1)

    # ---- Calibration ----
    def fit_with_mle(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)
        mu = np.average(X, weights=weights)
        diff = X - mu
        variance = np.inner(weights * diff, diff)

        self._mean = mu
        self._variance = variance
        self._validate()
        self._update_params()

    def fit_with_min_bregman(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)
        # compute dual/expectation parameters using sufficient statistics.
        eta = np.array([np.average(X, weights=weights),
                        np.average(X ** 2, weights=weights)])
        self.dual_param = eta

    def __repr__(self) -> str:
        return f"UnivariateGaussian(mean={self._mean:.3f}, variance={self._variance:.3f})"


# -------------------- Multivariate Gaussian --------------------
class MultivariateGaussian(ExponentialFamily):
    def __init__(self, mean: np.ndarray | None = None, covariance: np.ndarray | None = None):
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
    def params(self):
        return [self._mean, self._covariance]

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
        pass

    @property
    def dual_param(self):
        mu = self._mean
        second_moment = (self._covariance + np.outer(mu, mu)).flatten()
        return np.concatenate([mu, second_moment.ravel()])

    @dual_param.setter
    def dual_param(self, eta: np.ndarray):
        d = self.d
        mu = eta[:d]  # E[x]
        second_moments = eta[d:].reshape((d, d))  # E[x x^T]
        cov = second_moments - np.outer(mu, mu)  # covariance = E[x x^T] – mu mu^T
        cov += 1e-9 * np.eye(cov.shape[0])  # Numerical jitter if near-singular
        self._mean, self._covariance = mu, cov

    @staticmethod
    def get_sufficient_stat(X: np.ndarray) -> np.ndarray:
        """
        Get the sufficient statistic vector e.g. case d=2: [x y x^2 xy yx y^2]
        :param X:
        :return: array (N,d+d^2)
        """
        N = X.shape[0]
        d = X.shape[1]
        outer = np.einsum('ij,ik->ijk', X, X)  # (N,d,d)
        return np.concatenate([X, outer.reshape(N, d ** 2)], axis=1)

    @staticmethod
    def from_dual_to_ordinary(eta: np.ndarray) -> list[np.ndarray]:
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
        cov = cov + 1e-9 * np.eye(d)

        # if original input was 1D, unwrap outputs
        if cov.shape[0] == 1:
            return [mu[0], cov[0]]
        return [mu, cov]

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

    @staticmethod
    def kl_div(mean, cov, mean_0, cov_0):
        """
        Compute the KL divergence of a Multivariate Gaussian distribution.
        KL[N(mu, Sigma) : N(mu0, Sigma0)] = D_phi[eta : eta_0]
        """
        d = mean.shape[0]
        diff = mean - mean_0

        if np.linalg.det(cov) <= 1e-9:
            cov += 1e-9 * np.eye(cov.shape[0])
        if np.linalg.det(cov_0) <= 1e-9:
            cov_0 += 1e-9 * np.eye(cov_0.shape[0])

        # quadratic term
        solve_diff = np.linalg.solve(cov_0, diff)
        quadratic_term = np.inner(diff, solve_diff)

        # trace term
        solve_trace = np.linalg.solve(cov, cov_0)
        trace_term = np.trace(solve_trace)

        # log determinants
        log_det = np.log(np.linalg.det(cov))
        log_det_0 = np.log(np.linalg.det(cov_0))

        return 0.5 * (log_det_0 - log_det + quadratic_term + trace_term - d)

    # ----- Calibration -----
    def fit_with_mle(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)
        mu = np.average(X, axis=0, weights=weights)
        diff = X - mu
        # Broadcasting weights to columns; (N,1) * (N,d) -> weighted rows
        weighted_diff = weights[:, np.newaxis] * diff
        cov = weighted_diff.T @ diff
        cov += 1e-9 * np.eye(cov.shape[0])  # Numerical jitter if near-singular

        self._mean = mu
        self._covariance = cov
        self._validate()
        self._cache()

    def fit_with_min_bregman(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)  # normalizes weights
        # form sufficient stats and average
        suf_stat = self.get_sufficient_stat(X)  # shape (N, d + d^2)
        dual = np.average(suf_stat, axis=0, weights=weights)  # length d + d^2
        # update params
        self.dual_param = dual
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

    def __init__(self, loc=0.0, kappa=0.01):
        super().__init__()
        self._loc = float(loc)
        self._kappa = float(kappa)
        self._natural_param = None
        self._dual_param = None
        self._A = None
        self._validate()
        self._update_params()

    @staticmethod
    def _mean_length(kappa):
        kappa = np.clip(kappa, 1e-6, 100.0)
        return i1e(kappa) / i0e(kappa)

    @staticmethod
    def _inv_mean_length(r: float):
        """
        A^{-1} approximation given by Best and Fisher (1981).
        :param r:
        :return:
        """
        if r < 0.53:
            return 2 * r + r ** 3 + (5 * r ** 5) / 6
        elif r < 0.85:
            return -0.4 + 1.39 * r + 0.43 / (1 - r)
        else:
            return 1 / (r ** 3 - 4 * r ** 2 + 3 * r)

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
    def params(self):
        return np.array([self._loc, self._kappa]).copy()

    @property
    def mean_length(self):
        return self._A

    @property
    def natural_param(self):
        return self._natural_param.copy()

    @natural_param.setter
    def natural_param(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2,):
            raise ValueError("natural_param must be a length-2 vector.")
        self._natural_param = theta
        self._loc = np.arctan2(theta[1], theta[0])
        # A(kappa=100) = 0.9948
        self._kappa = np.minimum(np.linalg.norm(theta), 100.0)
        self._A = self._mean_length(self._kappa)
        self._validate()
        self._dual_param = np.array([self._A * np.cos(self._loc),
                                     self._A * np.sin(self._loc)])

    @property
    def dual_param(self):
        return self._dual_param.copy()

    @dual_param.setter
    def dual_param(self, eta: np.ndarray):
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (2,):
            raise ValueError("dual_param must be a length-2 vector.")
        self._dual_param = eta
        self._loc = np.arctan2(eta[1], eta[0])
        # A(kappa=100) = 0.9948
        self._A = np.minimum(np.linalg.norm(eta), 0.9948)
        self._kappa = self._inv_mean_length(self._A)
        self._validate()
        self._natural_param = np.array([self._kappa * np.cos(self._loc),
                                        self._kappa * np.sin(self._loc)])

    @staticmethod
    def get_sufficient_stat(X: np.ndarray):
        return X

    @staticmethod
    def from_dual_to_ordinary(eta: np.ndarray) -> list[np.ndarray]:
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
        A = np.minimum(np.linalg.norm(eta, axis=1), 0.9948)

        # vectorized Best–Fisher inversion:
        #   if A<0.53: 2A + A^3 + 5A^5/6
        #   elif A<0.85: -0.4 + 1.39A + 0.43/(1−A)
        #   else: 1/(A^3 − 4A^2 + 3A)
        kappa = np.where(
            A < 0.53,
            2 * A + A ** 3 + (5 * A ** 5) / 6,
            np.where(
                A < 0.85,
                -0.4 + 1.39 * A + 0.43 / (1 - A),
                1.0 / (A ** 3 - 4 * A ** 2 + 3 * A)
            )
        )

        out = [loc, kappa]  # shape (n,2)
        return [loc[0], kappa[0]] if single else out

    # ----- densities -----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_partition = np.log(2 * np.pi * i0e(self._kappa)) + self._kappa
        return X @ self._natural_param - log_partition

    # pdf inherited from base

    @staticmethod
    def kl_div(loc, kappa, loc_0, kappa_0):
        """
        Compute the KL divergence of a von Mises distribution using ordinary coordinate system.
        KL[vM(loc, kappa) : vM(loc0, kappa0)] = D_phi[eta : eta_0]
        """
        log_term = np.log(i0e(kappa_0) / i0e(kappa)) + kappa_0 - kappa
        A = (i1e(kappa) / i0e(kappa))
        return kappa * A + log_term - kappa_0 * A * np.cos(loc_0 - loc)

    # ----- Calibration -----
    def fit_with_mle(self, X: np.ndarray, weights: np.ndarray | None = None):
        X, weights = self._input_process(X, weights)
        const = np.log(2 * np.pi)

        def neg_ll(params):
            loc, kappa = params
            if kappa <= 0:
                return np.inf
            # i0e(kappa) = exp(-kappa)*i0(kappa)
            # -log(i0(kappa)) = -log(i0e(kappa)) - kappa
            ll = np.sum(weights * (kappa * (np.cos(loc) * X[:, 0] \
                                            + np.sin(loc) * X[:, 1]) \
                                   - np.log(i0e(kappa)) - kappa - const))
            return -ll  # minimize negative

        C = np.sum(weights * X[:, 0])
        S = np.sum(weights * X[:, 1])
        initials = np.array([np.arctan2(S, C), 1.0])
        bnds = ((-np.pi, np.pi), (1e-8, None))

        result = minimize(
            fun=neg_ll,
            x0=initials,
            method="L-BFGS-B",
            bounds=bnds
        )
        self._loc, self._kappa = result.x
        self._update_params()
        return result

    def fit_with_mle_proxy(self, X: np.ndarray, weights: np.ndarray | None = None):
        X, weights = self._input_process(X, weights)

        C = np.sum(weights * X[:, 0])
        S = np.sum(weights * X[:, 1])
        loc = np.arctan2(S, C)
        R = min(np.sqrt(C * C + S * S), 0.99)
        kappa = R * (2 - R * R) / (1 - R * R)

        self._loc, self._kappa = loc, kappa
        self._update_params()
        return loc, kappa

    def fit_with_min_bregman(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)

        # compute dual/expectation parameters using sufficient statistics.
        eta = np.average(X, axis=0, weights=weights)
        self.dual_param = eta

    def __repr__(self):
        return f"VonMises(loc={self._loc * 180 / np.pi:.1f}º, kappa={self._kappa:.3f})"


# -------------------- Mixture Model --------------------
class MixtureModel:
    def __init__(self, components: list[ExponentialFamily], weights: np.ndarray | None = None):
        self._components = components
        self.n_clusters = len(components)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.ndim != 1 or weights.size != self.n_clusters:
                raise ValueError("Components and weights mismatch.")
            if np.any(weights <= 0):
                raise ValueError("All weights must be > 0.")
            self._weights = weights / weights.sum()
            self._is_initialized = True
        else:
            self._weights = None
            self._is_initialized = False

    def _initialize(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        labels = KMeans(
                        n_clusters=self.n_clusters,
                        init='k-means++',
                        max_iter=1,
                        random_state=0
                    ).fit_predict(X)
        N = X.shape[0]
        posteriors = np.zeros((N, self.n_clusters))
        posteriors[np.arange(N), labels] = 1.0
        # if there's an empty cluster, then uniform posterior probability
        counts = posteriors.sum(axis=0)
        zeros = counts == 0

        for j, dist in enumerate(self._components):
            if not zeros[j]:
                dist.fit_with_mle(X, weights=posteriors[:, j])
            else:
                dist.fit_with_mle(X, weights=None)
        if ~np.any(zeros):
            self._weights = counts / counts.sum()
        else:
            self._weights = np.ones(self.n_clusters) / float(self.n_clusters)
        self._is_initialized = True

    # ---- Getter ----
    def get_weights(self):
        return self._weights

    def get_components(self):
        return self._components

    def get_posteriors(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        post, _ , _ = self._e_step(X)
        return post

    # ---- Densities ----
    def log_pdf_components(self, X: np.ndarray) -> np.ndarray:
        """
        Returns log p(x_i | k) for all i,k
        Shape: (N, K)
        """
        X = np.asarray(X, dtype=float)
        return np.column_stack([c.log_pdf(X) for c in self._components])

    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_pi = np.log(self._weights)
        return logsumexp(self.log_pdf_components(X) + log_pi, axis=1)

    def pdf(self, X):
        return np.exp(self.log_pdf(X))

    # ---- Expectation Maximization Algorithm ----
    def _e_step(self, X):
        # E-step: Compute the posterior
        # take log to prevent underflow
        eps = np.finfo(float).tiny
        log_prior = np.log(self._weights + eps)  # (K,)
        log_p = self.log_pdf_components(X)  # (N, K)
        log_numerator = log_prior + log_p  # (N, K)
        log_denominator = logsumexp(log_numerator, axis=1, keepdims=True)  # (N, 1)
        log_posterior = log_numerator - log_denominator  # (N, K)
        posterior = np.exp(log_posterior)  # responsibilities (N, K)
        log_likelihood = log_denominator.sum()
        expected_log_likelihood = np.sum(log_numerator * posterior)
        return posterior, log_likelihood, expected_log_likelihood

    def fit_em_classic(self, X, weight=None, tol=1e-6, max_iter=200, verbose=False):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        if not self._is_initialized:
            self._initialize(X)
        if weight is None:
            weight = np.ones(N)
        logger = []
        for it in range(max_iter):
            # E-step: Compute the posterior
            posterior, _ , log_likelihood = self._e_step(X)
            logger.append(log_likelihood)

            # M-step: Maximize weighted log-likelihood
            # update priors
            self._weights = np.average(posterior, axis=0, weights=weight)  # (K,)
            # update distribution parameters
            for k, comp in enumerate(self._components):
                comp.fit_with_mle(X, weight * posterior[:, k])

            # verbose
            if verbose:
                print(f"Data log-likelihood at iter {it}: {logger[-1]:.2f}")
            # check convergence
            if it > 0 and abs(logger[-1] - logger[-2]) < tol:
                if verbose:
                    print(f"Converged at iter {it}: Delta LL={logger[-1] - logger[-2]:.3e}")
                break
        else:
            if verbose:
                print("Reached max_iter without full convergence.")
        return logger

    def fit_em_bregman(self, X, tol=1e-6, weight=None, max_iter=200, verbose=False):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        if not self._is_initialized:
            self._initialize(X)
        if weight is None:
            weight = np.ones(N)
        logger = []
        for it in range(max_iter):
            # E-step: Compute the posterior
            posterior, _ , log_likelihood = self._e_step(X)
            logger.append(log_likelihood)

            # M-step: Maximize weighted log-likelihood
            # update priors
            self._weights = np.average(posterior, axis=0, weights=weight)  # (K,)
            # update distribution parameters
            for k, comp in enumerate(self._components):
                comp.fit_with_min_bregman(X, weight * posterior[:, k])

            # verbose
            if verbose:
                print(f"Data log-likelihood at iter {it}: {logger[-1]:.2f}")
            # check convergence
            if it > 0 and abs(logger[-1] - logger[-2]) < tol:
                if verbose:
                    print(f"Converged at iter {it}: Delta LL={logger[-1] - logger[-2]:.3e}")
                break
        else:
            if verbose:
                print("Reached max_iter without full convergence.")
        return logger

    def fit_em_vonmises_approx(self, X, weight=None, tol=1e-6, max_iter=200, verbose=False):
        if not all(isinstance(c, VonMises) for c in self._components):
            raise TypeError("All components must be VonMises.")

        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        if not self._is_initialized:
            self._initialize(X)
        if weight is None:
            weight = np.ones(N)
        logger = []
        for it in range(max_iter):
            # E-step: Compute the posterior
            posterior, _ , log_likelihood = self._e_step(X)
            logger.append(log_likelihood)

            # M-step: Maximize weighted log-likelihood
            # update priors
            self._weights = np.average(posterior, axis=0, weights=weight)  # (K,)
            # update distribution parameters
            for k, comp in enumerate(self._components):
                comp.fit_with_mle_proxy(X, weight * posterior[:, k])

            # ---------- Check convergence ----------
            if it > 0 and abs(logger[-1] - logger[-2]) < tol:
                if verbose:
                    print(f"Converged at iter {it}: Delta LL={logger[-1] - logger[-2]:.3e}")
                break
        else:
            if verbose:
                print("Reached max_iter without full convergence.")
        return logger

    # ---- Performance metrics ---
    def bic_score(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        ll = self.log_pdf(X).sum()
        dist_params = self.get_components()[0].params
        p = self.n_clusters - 1
        for param in dist_params:
            if isinstance(param, float):
                p += 1
            else:
                p += param.size * self.n_clusters
        return np.log(X.shape[0]) * p - 2 * ll

    def aic_score(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        ll = self.log_pdf(X).sum()
        dist_params = self.get_components()[0].params
        p = self.n_clusters - 1
        for param in dist_params:
            if isinstance(param, float):
                p += 1
            else:
                p += param.size * self.n_clusters
        return 2 * p - 2 * ll

    def hard_predict(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        posteriors, _ , _ = self._e_step(X)
        labels = np.argmax(posteriors, axis=1)
        return labels

    def kl_ch_score(self, X):
        """
        Compute the Calinski–Harabasz score but instead of Euclidean distance,
        use KL divergences.
        CH = between-cluster separation (BC) / within-cluster dispersion (WC)
        BC = sum_j^K prior_j * KL[eta_j : centroid]
        WC = sum_j^K sum_i^N posterior_{ij} * KL[x_i : eta_j]
        """
        # precompute functions and values
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        K = self.n_clusters
        KL = self.get_components()[0].kl_div  # to have access to KL divergence of the dist.
        transform = self.get_components()[0].from_dual_to_ordinary
        priors = self.get_weights()  # shape (K,)
        _ , _ , expected_log_ll = self._e_step(X)

        # compute BC
        dual_parameters = []
        for dist in self.get_components():
            dual_parameters.append(dist.dual_param)
        clus_dual = np.stack(dual_parameters)
        clus_ord_1, clus_ord_2 = transform(clus_dual)
        centroid_dual = np.average(clus_dual, axis=0, weights=self.get_weights())
        centroid_ord_1, centroid_ord_2 = transform(centroid_dual)
        bc = 0
        for j in range(K):
            bc += priors[j] * KL(clus_ord_1[j], clus_ord_2[j], centroid_ord_1, centroid_ord_2)

        # compute wc
        wc = -1*expected_log_ll # negative log likelihood weighted by posteriors

        return (bc / (K - 1)) / (wc / (N - K))

    # ---- Display ----
    @staticmethod
    def _format_component(idx: int, w: float | None, comp) -> str:
        w_str = f"{w:0.3f}" if w is not None else "—"
        return f"  ├─ ({idx}) w={w_str}  {comp!r}"

    def __repr__(self) -> str:
        header = f"{self.__class__.__name__}(n_clusters={self.n_clusters})"
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


def two_layer_scheme(loc_data: np.ndarray, dir_data: np.ndarray, K_loc: int, K_dir: int, choose="classic", gmm = None):
    """
    Perform a two-layers clustering scheme, starting with a K_loc-GMM, and then, for each Gaussian cluster performs a K_dir-vMMM.
    vMMM: von Mises Mixture Model. GMM: Gaussian Mixture Model.
    :param loc_data: Event location data. Coordinates (x,y)
    :param dir_data: Event direction data as (Cos(alpha), Sin(alpha)
    :return:
    """
    dir_data = np.array(dir_data, dtype=float)
    loc_data = np.array(loc_data, dtype=float)
    if dir_data.shape[0] != loc_data.shape[0]:
        raise ValueError("Sample size don't match.")
    N = loc_data.shape[0]
    if gmm is None:
        gmm_components = [MultivariateGaussian() for _ in range(K_loc)]
        gmm = MixtureModel(gmm_components)
        match choose:
            case "bregman":
                _ = gmm.fit_em_bregman(loc_data)
            case _:
                _ = gmm.fit_em_classic(loc_data)

    # get the posteriors of the first layer
    posteriors = gmm.get_posteriors(loc_data)
    vmmm_list = []

    for loc_cluster in range(K_loc):
        # first layer posterior is fixed
        loc_posterior = posteriors[:, loc_cluster]
        # obtain initial weights of the second layer
        #labels = KMeans(n_clusters=K_dir,
        #                init='k-means++',
        #                max_iter=1,
        #                random_state=0).fit_predict(dir_data, sample_weight=loc_posterior)
        #one_hot = np.zeros((N, K_dir))
        #one_hot[np.arange(N), labels] = 1.0
        #initial_weight = one_hot.sum(axis=0) / one_hot.sum()
        vmmm_components = [VonMises() for _ in range(K_dir)]
        vmmm = MixtureModel(vmmm_components, None)

        match choose:
            case "bregman":
                _ = vmmm.fit_em_bregman(dir_data, weight=loc_posterior)
            case "classic":
                _ = vmmm.fit_em_classic(dir_data, weight=loc_posterior)
            case "soccermix":
                _ = vmmm.fit_em_vonmises_approx(dir_data, weight=loc_posterior)
            case _:
                raise ValueError("Invalid choice")
        vmmm_list.append(vmmm)

    return gmm, vmmm_list

def consolidate(actions):
    #actions.fillna(0, inplace=True)

    #Consolidate corner_short and corner_crossed
    corner_idx = actions.type_name.str.contains("corner")
    actions["type_name"] = actions["type_name"].mask(corner_idx, "corner")

    #Consolidate freekick_short, freekick_crossed, and shot_freekick
    freekick_idx = actions.type_name.str.contains("freekick")
    actions["type_name"] = actions["type_name"].mask(freekick_idx, "freekick")

    #Consolidate keeper_claim, keeper_punch, keeper_save, keeper_pick_up
    keeper_idx = actions.type_name.str.contains("keeper")
    actions["type_name"] = actions["type_name"].mask(keeper_idx, "keeper_action")

    actions["start_x"] = actions["start_x"].mask(actions.type_name == "shot_penalty", 94.5)
    actions["start_y"] = actions["start_y"].mask(actions.type_name == "shot_penalty", 34)

    return actions

def add_noise(actions):
    # Start locations
    start_list = ["cross", "shot", "dribble", "pass", "keeper_action", "clearance", "goalkick"]
    mask = actions["type_name"].isin(start_list)
    noise = np.random.normal(0, 0.5, size=actions.loc[mask, ["start_x", "start_y"]].shape)
    actions.loc[mask, ["start_x", "start_y"]] += noise

    # End locations
    end_list = ["cross", "shot", "dribble", "pass", "keeper_action", "throw_in", "corner", "freekick", "shot_penalty"]
    mask = actions["type_name"].isin(end_list)
    noise = np.random.normal(0, 0.5, size=actions.loc[mask, ["end_x", "end_y"]].shape)
    actions.loc[mask, ["end_x", "end_y"]] += noise

    return actions

def remove_outliers(actions, verbose=False):
    X = actions[["start_x","start_y","end_x","end_y"]].to_numpy(dtype=float)
    inliers = LocalOutlierFactor(contamination="auto").fit_predict(X)
    if verbose:
        print(f"Remove {(inliers==-1).sum()} out of {X.shape[0]} datapoints.")
    return actions[inliers==1]
