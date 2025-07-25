import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.special import i0e, i1e, logsumexp
from sklearn.cluster import KMeans


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
    def fit_with_mle(self, X: np.ndarray, weights: np.ndarray | None = None):
        pass

    @abstractmethod
    def fit_with_min_bregman(self, X: np.ndarray, weights: np.ndarray | None = None):
        pass

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
        θ = ( -μ/σ² , -1/(2σ²) )
    dual / mean-value form:
        η = ( μ , μ² + σ² )
    """

    def __init__(self, mean: float = 0.0, variance: float = 1.0):
        self.mean = float(mean)
        self.variance = float(variance)
        self.natural_param = None
        self.dual_param = None
        self._validate()
        self._update_params()

    def _validate(self) -> None:
        if self.variance <= 0:
            raise ValueError("variance must be > 0.")
        if self.natural_param is not None and self.natural_param[1] >= 0:
            raise ValueError("Second natural parameter must be negative.")
        if self.dual_param is not None and self.dual_param[1] <= self.dual_param[0] ** 2:
            raise ValueError("eta2 - eta1^2 must be > 0.")

    def _update_params(self) -> None:
        self.natural_param = np.array([
            -self.mean / self.variance,
            -1.0 / (2.0 * self.variance)
        ])
        self.dual_param = np.array([
            self.mean,
            self.mean ** 2 + self.variance
        ])

    # ---- setters ----
    def set_params(self, mean: float, variance: float) -> None:
        self.mean, self.variance = float(mean), float(variance)
        self._validate()
        self._update_params()

    def set_natural_params(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        self.natural_param = theta
        self.mean = -0.5 * theta[0] / theta[1]
        self.variance = -0.5 / theta[1]
        self._validate()
        self._update_params()

    def set_dual_params(self, eta: np.ndarray) -> None:
        eta = np.asarray(eta, dtype=float)
        self.dual_param = eta
        self.mean = eta[0]
        self.variance = eta[1] - eta[0] ** 2
        self._validate()
        self._update_params()

    # ---- densities ----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return -0.5 * ((X - self.mean) ** 2) / self.variance - 0.5 * np.log(2 * np.pi * self.variance)

    # pdf inherited from base

    # ---- calibration ----
    def fit_with_mle(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)
        mu = np.average(X, weights=weights)
        diff = X - mu
        variance = np.dot(weights * diff, diff)

        self.mean = mu
        self.variance = variance
        self._validate()
        self._update_params()

    def fit_with_min_bregman(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)
        # compute dual/expectation parameters using sufficient statistics.
        eta = np.array([np.average(X, weights=weights),
               np.average(X ** 2, weights=weights)])
        self.set_dual_params(eta)

    def __repr__(self) -> str:
        return f"UnivariateGaussian(mean={self.mean:.3f}, variance={self.variance:.3f})"


# -------------------- Multivariate Gaussian --------------------
class MultivariateGaussian(ExponentialFamily):
    def __init__(self, mean: np.ndarray | None = None, covariance: np.ndarray | None = None):
        super().__init__()
        self.mean = np.zeros(2) if mean is None else np.asarray(mean, dtype=float)
        self.covariance = np.eye(self.mean.size) if covariance is None else np.asarray(covariance, dtype=float)
        self._validate()
        self._cache()

    @property
    def d(self) -> int:
        return self.mean.size

    def _validate(self) -> None:
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if self.mean.shape[0] != self.covariance.shape[0]:
            raise ValueError("Mean and covariance shapes mismatch.")
        if not np.allclose(self.covariance, self.covariance.T):
            raise ValueError("Covariance must be symmetric.")
        # Positive definiteness checked in _cache via Cholesky

    def _cache(self) -> None:
        try:
            self._chol = np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError as e:
            raise ValueError("Covariance must be positive-definite.") from e
        self._log_det = 2 * np.sum(np.log(np.diag(self._chol)))

    def set_params(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self._validate()
        self._cache()

    # ---- getters ----
    def get_sufficient_stat(self, X: np.ndarray) -> np.ndarray:
        """
        Get the sufficient statistic vector e.g. case d=2: [x y x^2 xy yx y^2]
        :param X:
        :return: array (N,d+d^2)
        """
        N = X.shape[0]
        outer = np.einsum('ij,ik->ijk', X, X)  # (N,d,d)
        return np.concatenate([X, outer.reshape(N, self.d ** 2)], axis=1)


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
    def fit_with_mle(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)
        mu = np.average(X, axis=0, weights=weights)
        diff = X - mu
        # Broadcasting weights to columns; (N,1) * (N,d) -> weighted rows
        weighted_diff = weights[:, np.newaxis] * diff
        cov = weighted_diff.T @ diff
        cov += 1e-9 * np.eye(cov.shape[0]) # Numerical jitter if near-singular

        self.mean = mu
        self.covariance = cov
        self._validate()
        self._cache()

    def fit_with_min_bregman(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)  # normalizes weights
        d = self.d
        # form sufficient stats and average
        suf_stat = self.get_sufficient_stat(X)  # shape (N, d + d^2)
        dual = np.average(suf_stat, axis=0, weights=weights)  # length d + d^2
        # update params
        mu = dual[:d]  # E[x]
        second_moments = dual[d:].reshape((d, d))  # E[x x^T]
        cov = second_moments - np.outer(mu, mu)  # covariance = E[x x^T] – mu mu^T
        cov += 1e-9 * np.eye(cov.shape[0])  # Numerical jitter if near-singular
        self.mean = mu
        self.covariance = cov
        self._validate()
        self._cache()

    def __repr__(self):
        return f"MultivariateGaussian(d={self.d}, mean={self.mean}, cov={self.covariance})"


# -------------------- Von Mises --------------------
class VonMises(ExponentialFamily):
    """
    Von Mises distribution for directional data.
    X must be in sufficient statistic form: [cos x, sin x].
    """

    def __init__(self, loc=0.0, kappa=0.01):
        super().__init__()
        self.loc = float(loc)
        self.kappa = float(kappa)
        self._validate()
        self._update_params()

    @staticmethod
    def _mean_length(x):
        return i1e(x) / i0e(x)

    def _inv_mean_length(self, r, newton_steps=2):
        def initial_guess(r):
            if r < 0.53:
                return 2 * r + r ** 3 + 5 * r ** 5 / 6
            elif r < 0.85:
                return -0.4 + 1.39 * r + 0.43 / (1 - r)
            else:
                return 1 / (r ** 3 - 4 * r ** 2 + 3 * r)

        # 1) initial guess
        k = initial_guess(r)
        # 2) Newton refine
        for _ in range(newton_steps):
            Ak = self._mean_length(k)
            # derivative A'(k)
            dAk = 1 - Ak ** 2 - Ak / k
            k -= (Ak - r) / dAk
            if k <= 0:
                k = 1e-8
        return k

    def _validate(self):
        if self.kappa <= 0:
            raise ValueError("Concentration parameter kappa must be positive.")

    def _update_params(self):
        self.natural_param = np.array([self.kappa * np.cos(self.loc),
                                       self.kappa * np.sin(self.loc)])
        A = self._mean_length(self.kappa)
        self.dual_param = np.array([A * np.cos(self.loc),
                                    A * np.sin(self.loc)])

    # ---- setters ----
    def set_dual_params(self, eta: np.ndarray):
        self.dual_param = eta
        self.loc = np.arctan2(eta[1], eta[0])
        self.kappa = self._inv_mean_length(np.linalg.norm(eta))
        self._validate()
        self.natural_param = np.array([self.kappa * np.cos(self.loc),
                                       self.kappa * np.sin(self.loc)])

    # ----- densities -----
    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_partition = np.log(2 * np.pi * i0e(self.kappa)) + self.kappa
        return X @ self.natural_param - log_partition

    # pdf inherited from base

    # ----- Calibration -----
    def fit_with_mle(self, X: np.ndarray, weights: np.ndarray | None = None):
        X, weights = self._input_process(X, weights)
        const = np.log(2 * np.pi)

        def neg_ll(params):
            loc, kappa = params
            if kappa <= 0:
                return np.inf
            #i0e(kappa) = exp(-kappa)*i0(kappa)
            #-log(i0(kappa)) = -log(i0e(kappa)) - kappa
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
        self.loc, self.kappa = result.x
        self._update_params()
        return result

    def fit_with_mle_proxy(self, X: np.ndarray, weights: np.ndarray | None = None):
        X, weights = self._input_process(X, weights)

        C = np.sum(weights * X[:, 0])
        S = np.sum(weights * X[:, 1])
        loc = np.arctan2(S, C)
        R = min(np.sqrt(C * C + S * S), 0.99)
        kappa = R * (2 - R * R) / (1 - R * R)

        self.loc, self.kappa = loc, kappa
        self._update_params()
        return loc, kappa

    def fit_with_min_bregman(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        X, weights = self._input_process(X, weights)

        # compute dual/expectation parameters using sufficient statistics.
        eta = np.average(X, axis=0, weights=weights)
        self.set_dual_params(eta)

    def __repr__(self):
        return f"VonMises(loc={self.loc * 180 / np.pi:.1f}º, kappa={self.kappa:.3f})"


# -------------------- Mixture Model --------------------
class MixtureModel:
    def __init__(self, components: list[ExponentialFamily], weights: np.ndarray | None = None):
        self.components = components
        self.n_clusters = len(components)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.ndim != 1 or weights.size != self.n_clusters:
                raise ValueError("Components and weights mismatch.")
            if np.any(weights <= 0):
                raise ValueError("All weights must be > 0.")
            self.weights = weights / weights.sum()
            self.is_initialized = True
        else:
            self.weights = None
            self.is_initialized = False

    def _initialize(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        labels = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=0).fit_predict(X)
        N = X.shape[0]
        posteriors = np.zeros((N, self.n_clusters))
        posteriors[np.arange(N), labels] = 1.0
        # if there's an empty cluster, then uniform posterior probability
        counts = posteriors.sum(axis=0)
        zeros = counts == 0
        #if np.any(zeros):
        #    raise ValueError("One or more cluster at initialization have zero datamembers.")

        for j, dist in enumerate(self.components):
            if not zeros[j]:
                dist.fit_with_mle(X, weights=posteriors[:, j])
            else:
                dist.fit_with_mle(X, weights=None)

        if np.any(zeros):
            counts = np.maximum(counts, 0.001)  # may be a problem, but let's see
        self.weights = counts / counts.sum()

        self.is_initialized = True

    def log_pdf_components(self, X: np.ndarray) -> np.ndarray:
        """
        Returns log p(x_i | k) for all i,k
        Shape: (N, K)
        """
        X = np.asarray(X, dtype=float)
        return np.column_stack([c.log_pdf(X) for c in self.components])

    def log_pdf(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_pi = np.log(self.weights)
        return logsumexp(self.log_pdf_components(X) + log_pi, axis=1)

    def pdf(self, X):
        return np.exp(self.log_pdf(X))

    def _e_step(self, X):
        # E-step: Compute the posterior
        # take log to prevent underflow
        log_prior = np.log(self.weights)  # (K,)
        log_p = self.log_pdf_components(X)  # (N, K)
        log_numerator = log_prior + log_p  # (N, K)
        log_denominator = logsumexp(log_numerator, axis=1, keepdims=True)  # (N, 1)
        log_posterior = log_numerator - log_denominator  # (N, K)
        posterior = np.exp(log_posterior)  # responsibilities (N, K)
        log_likelihood = log_denominator.sum()
        return posterior, log_likelihood

    def fit_em_classic(self, X, tol=1e-6, max_iter=200, verbose=False):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        if not self.is_initialized:
            self._initialize(X)
        logger = []
        for it in range(max_iter):
            # E-step: Compute the posterior
            posterior, log_likelihood = self._e_step(X)
            logger.append(log_likelihood)

            # M-step: Maximize weighted log-likelihood
            # update priors
            self.weights = posterior.sum(axis=0) / N  # effective counts (K,)
            # update distribution parameters
            for k, comp in enumerate(self.components):
                comp.fit_with_mle(X, posterior[:, k])

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

    def fit_em_bregman(self, X, tol=1e-6, max_iter=200, verbose=False):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        if not self.is_initialized:
            self._initialize(X)
        logger = []
        for it in range(max_iter):
            # E-step: Compute the posterior
            posterior, log_likelihood = self._e_step(X)
            logger.append(log_likelihood)

            # M-step: Maximize weighted log-likelihood
            # update priors
            self.weights = posterior.sum(axis=0) / N  # effective counts (K,)
            # update distribution parameters
            for k, comp in enumerate(self.components):
                comp.fit_with_min_bregman(X, posterior[:, k])

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

    def fit_em_vonmises_approx(self, X, tol=1e-6, max_iter=200, verbose=False):
        if not all(isinstance(c, VonMises) for c in self.components):
            raise TypeError("All components must be VonMises.")

        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        if not self.is_initialized:
            self._initialize(X)
        logger = []
        for it in range(max_iter):
            # E-step: Compute the posterior
            posterior, log_likelihood = self._e_step(X)
            logger.append(log_likelihood)

            # M-step: Maximize weighted log-likelihood
            # update priors
            self.weights = posterior.sum(axis=0) / N  # effective counts (K,)
            # update distribution parameters
            for k, comp in enumerate(self.components):
                comp.fit_with_mle_proxy(X, posterior[:, k])

            # ---------- Check convergence ----------
            if it > 0 and abs(logger[-1] - logger[-2]) < tol:
                if verbose:
                    print(f"Converged at iter {it}: Delta LL={logger[-1] - logger[-2]:.3e}")
                break
        else:
            if verbose:
                print("Reached max_iter without full convergence.")
        return logger

def initial_priors(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Obtain initial guess for mixture's priors using KMeans++ algorithm.
    prior_j = (1/N) * sum_i^N I{x_i in cluster_j}
    :return:
    (n_clusters,) np.array of prior probabilities.
    """
    X = np.asarray(X, dtype=float)
    labels = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit_predict(X)
    counts = np.bincount(labels, minlength=n_clusters).astype(float)  # count how many of each distinct integer
    # protect against empty clusters
    counts = np.maximum(counts, 1e-12)
    return counts / counts.sum()
