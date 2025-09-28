# metrics.py
"""
Metrics and helpers used by distributions and mixture models.

Keep these pure functions stateless so they can be reused safely.
"""

from __future__ import annotations
import numpy as np
from scipy.special import i0e, i1e
from .distributions import (
    ExponentialFamily,
    UnivariateGaussian,
    MultivariateGaussian,
    VonMises,
    CustomBregman,
)
from .mixture import MixtureModel

_EPS = 1e-9


def kl_div_univariate_gaussian(mean: float,
                               var: float,
                               mean_0: float,
                               var_0: float) -> float:
    """
    Compute the KL divergence of a Univariate Gaussian distribution.
    KL[N(mu, var) : N(mu0, var0)] = D_phi[eta : eta_0]
    """
    return 0.5 * (np.log(var_0 / var) + (var + (mean - mean_0) ** 2) / var_0 - 1)


def kl_div_multivariate_gaussian(mean: np.ndarray,
                                 cov: np.ndarray,
                                 mean_0: np.ndarray,
                                 cov_0: np.ndarray) -> float:
    """KL( N(mean, cov) || N(mean_0, cov_0) )."""
    d = mean.shape[0]
    diff = mean - mean_0

    # Jitter if near-singular
    if np.linalg.det(cov) <= _EPS:
        cov = cov + _EPS * np.eye(cov.shape[0])
    if np.linalg.det(cov_0) <= _EPS:
        cov_0 = cov_0 + _EPS * np.eye(cov_0.shape[0])

    # quadratic term: (mu0 - mu)^T Sigma0^{-1} (mu0 - mu)
    solve_diff = np.linalg.solve(cov_0, diff)
    quadratic_term = float(np.inner(diff, solve_diff))

    # trace term: tr(Sigma0^{-1} Sigma)
    trace_term = float(np.trace(np.linalg.solve(cov_0, cov)))

    # log determinant ratio: log det Sigma0 - log det Sigma
    sign, log_det = np.linalg.slogdet(cov)
    sign0, log_det_0 = np.linalg.slogdet(cov_0)
    if sign <= 0 or sign0 <= 0:
        # fallback to plain det (shouldn't happen with jitter above)
        log_det = np.log(np.linalg.det(cov))
        log_det_0 = np.log(np.linalg.det(cov_0))

    return 0.5 * (log_det_0 - log_det + quadratic_term + trace_term - d)


def kl_div_vonmises(loc: float,
                    kappa: float,
                    loc_0: float,
                    kappa_0: float) -> float:
    """
    Compute the KL divergence of a von Mises distribution using ordinary coordinate system.
    KL[vM(loc, kappa) : vM(loc0, kappa0)] = D_phi[eta : eta_0]
    """
    log_term = np.log(i0e(kappa_0) / i0e(kappa)) + kappa_0 - kappa
    A = (i1e(kappa) / i0e(kappa))
    return kappa * A + log_term - kappa_0 * A * np.cos(loc_0 - loc)


def _num_free_params_for_component(comp: ExponentialFamily) -> int:
    """Return the number of *free* parameters for a single component.

    We count independent degrees of freedom (e.g., full symmetric covariance has d(d+1)/2,
    not d^2). This is used for AIC/BIC.
    """
    if isinstance(comp, MultivariateGaussian):
        d = comp.d
        return d + (d * (d + 1)) // 2
    if isinstance(comp, UnivariateGaussian):
        return 2
    if isinstance(comp, VonMises):
        return 2
    if isinstance(comp, CustomBregman):
        # composite: count the underlying parts
        return _num_free_params_for_component(comp.gaussian) + _num_free_params_for_component(comp.vonmises)

    # Fallback: best-effort count
    params = comp.params
    if not isinstance(params, (tuple, list)):
        params = (params,)
    count = 0
    for p in params:
        arr = np.asarray(p)
        if arr.ndim == 0:
            count += 1
        else:
            count += arr.size
    return count


def bic_score_mixture(X: np.ndarray,
                      model: MixtureModel,
                      ) -> float:
    """Bayesian Information Criterion (lower is better)."""
    X = np.asarray(X, dtype=float)
    ll = model.log_pdf(X).sum()
    dist_params = model.components[0].params
    p = model.n_components - 1
    for param in dist_params:
        if isinstance(param, float):
            p += 1 * model.n_components
        else:
            p += param.size * model.n_components
    return np.log(X.shape[0]) * p - 2 * ll


def aic_score_mixture(X: np.ndarray,
                      model: MixtureModel,
                      ) -> float:
    """Akaike Information Criterion (lower is better)."""
    X = np.asarray(X, dtype=float)
    ll = model.log_pdf(X).sum()
    dist_params = model.components[0].params
    p = model.n_components - 1
    for param in dist_params:
        if isinstance(param, float):
            p += 1
        else:
            p += param.size * model.n_components
    return 2 * p - 2 * ll


def hard_predict(X: np.ndarray,
                 model: MixtureModel,
                 ) -> np.ndarray:
    labels = np.argmax(model.get_posteriors(X), axis=1)
    return labels


def kl_ch_score(X: np.ndarray,
                model: MixtureModel,
                ) -> float:
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
    K = model.n_components
    components = model.components

    if isinstance(components[0], MultivariateGaussian):
        KL = kl_div_multivariate_gaussian
    elif isinstance(components[0], VonMises):
        KL = kl_div_vonmises
    else:
        raise TypeError("kl_ch_score currently supports MultivariateGaussian or VonMises components.")

    transform = model.components[0].from_dual_to_ordinary # dual -> ordinary
    priors = model.weights  # shape (K,)
    log_ll = model.get_data_ll(X) #float

    # cluster centroids in dual space and their ordinary params
    clus_dual = np.stack([dist.dual_param for dist in components])  # (K, ·)
    clus_ord_1, clus_ord_2 = transform(clus_dual)
    centroid_dual = np.average(clus_dual, axis=0, weights=priors)
    centroid_ord_1, centroid_ord_2 = transform(centroid_dual)

    # between-cluster separation
    bc = 0.0
    for j in range(K):
        bc += priors[j] * KL(clus_ord_1[j], clus_ord_2[j], centroid_ord_1, centroid_ord_2)

    wc = -log_ll  # non-negative within dispersion proxy

    # Guard against degenerate cases
    if K <= 1 or N <= K:
        return np.nan

    return float((bc / (K - 1)) / (wc / (N - K)))
