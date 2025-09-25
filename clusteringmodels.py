import numpy as np
import warnings
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.special import i0e, i1e, logsumexp
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotsoccer as mps

import softclustering as sc

# grab the default color cycle as a list of hex‐colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# ------ Two-layer Scheme -----
class TwoLayerScheme:
    def __init__(self,
                 loc_mixture: sc.MixtureModel,
                 dir_mixtures: list[sc.MixtureModel]):
        self.loc_mixture = loc_mixture
        self.loc_n_clusters = loc_mixture.n_components
        assert len(
            dir_mixtures) == self.loc_n_clusters, "Components in loc mixture and number of dir mixture don't match"
        self.dir_mixtures = dir_mixtures

    def fit(self, loc_data, dir_data, sample_weight=None, tol=1e-4, max_iter=1000, verbose=False, case="classic"):
        loc_data = np.asarray(loc_data, dtype=float)
        dir_data = np.asarray(dir_data, dtype=float)
        N = loc_data.shape[0]
        assert N == dir_data.shape[0], "Location and direction number of observation don't match"

        _ = self.loc_mixture.fit_em(loc_data,
                                    sample_weight=None,
                                    tol=tol,
                                    max_iter=max_iter,
                                    verbose=verbose,
                                    case=case)
        # include a jitter in the posteriors probabilities
        loc_posteriors = self.loc_mixture.get_posteriors(loc_data) + 1e-9
        for j in range(self.loc_n_clusters):
            _ = self.dir_mixtures[j].fit_em(dir_data,
                                            sample_weight=loc_posteriors[:, j],
                                            case=case)

    def log_pdf(self, loc_data: np.ndarray, dir_data: np.ndarray):
        """
        Returns log p(x). Shape: (N,)
        """
        loc_posteriors = self.loc_mixture.get_posteriors(loc_data) + 1e-9  # (N,K)
        dir_log_pdf_array = [self.dir_mixtures[k].log_pdf(dir_data)[:, None]  # (N,1)
                             for k in range(self.loc_n_clusters)]
        dir_log_pdf = np.concatenate(dir_log_pdf_array, axis=1)  # (N,K)
        return logsumexp(np.log(loc_posteriors) + dir_log_pdf, axis=1)  # (N,)

    def pdf(self, loc_data: np.ndarray, dir_data: np.ndarray):
        return np.exp(self.log_pdf(loc_data, dir_data))

    def bic_score(self, loc_data, dir_data):
        loc_n_params = self.loc_mixture.n_components - 1  # prior parameters
        dist_params = self.loc_mixture.get_components()[0].params
        for param in dist_params:
            if isinstance(param, float):
                loc_n_params += 1
            else:
                loc_n_params += param.size * self.loc_mixture.n_components
        dir_n_params = 0
        for k in range(self.loc_n_clusters):
            dir_mixture = self.dir_mixtures[k]
            dir_n_params += dir_mixture.n_components - 1  # prior parameters

            dist_params = dir_mixture.get_components()[0].params
            for param in dist_params:
                if isinstance(param, float):
                    dir_n_params += 1
                else:
                    dir_n_params += param.size * dir_mixture.n_components
        p = dir_n_params + loc_n_params
        ll = self.log_pdf(loc_data, dir_data).sum()
        return np.log(loc_data.shape[0]) * p - 2 * ll

    def plot(self, figsize: float = 6, arrow_scale: float = 12.0, title: str = None, save: bool = False):
        """
        Plot every (Gaussian + VonMises arrows) on one shared Axes,
        using a different color per cluster, and arrow lengths proportional to mean length r.
        """
        ax = mps.field(show=False, figsize=figsize)

        n = self.loc_mixture.n_components
        palette = colors * ((n // len(colors)) + 1)

        for i, (loc, dir) in enumerate(zip(self.loc_mixture.get_components(),
                                           self.dir_mixtures)):
            col = palette[i]
            mean, cov = loc.params
            add_ellips(ax, mean, cov, color=col, alpha=0.5)
            x0, y0 = mean

            for vonm in dir.get_components():
                loc, _ = vonm.params
                r = vonm.mean_length  # in [0, 1]
                length = arrow_scale * r  # scale accordingly
                dx, dy = np.cos(loc), np.sin(loc)
                add_arrow(ax, x0, y0,
                          length * dx, length * dy,
                          linewidth=0.8)

        if title is not None:
            plt.title(title)
        if save:
            plt.savefig(f"plots/model_{title}.pdf", bbox_inches='tight')
        plt.show()


# ----- Custom Bregman Distribution -----
class CustomBregman(sc.ExponentialFamily):
    """
    p(x) = Gauss(x|theta_gaus)^alpha * VonMises(x|theta_vm)^beta
    """

    def __init__(self, coef_gaus: float, coef_vm: float):
        super().__init__()
        self._vonmises = sc.VonMises()
        self._gaussian = sc.MultivariateGaussian()
        self._coef_gauss = coef_gaus
        self._coef_vm = coef_vm

    @property
    def params(self):
        return [self._vonmises.params, self._gaussian.params, self._coef_gauss, self._coef_vm]

    @property
    def dual_param(self):
        pass

    @staticmethod
    def from_dual_to_ordinary(eta: np.ndarray):
        pass

    @staticmethod
    def kl_div(p_param1, p_param2, q_param1, q_param2):
        return 0

    def log_pdf(self, X: np.ndarray):
        X_gauss = np.asarray(X, dtype=float)[:, :2]
        X_vm = np.asarray(X, dtype=float)[:, 2:]
        log_gauss = self._gaussian.log_pdf(X_gauss)
        log_vm = self._vonmises.log_pdf(X_vm)
        return self._coef_gauss * log_gauss + self._coef_vm * log_vm

    def fit(self, X: np.ndarray, sample_weight: np.ndarray | None = None, case: str | None = None):
        X_gauss = np.asarray(X, dtype=float)[:, :2]
        X_vm = np.asarray(X, dtype=float)[:, 2:]
        self._gaussian.fit(X_gauss, sample_weight=sample_weight, case=case)
        self._vonmises.fit(X_vm, sample_weight=sample_weight, case=case)

    def get_gaussian(self):
        return self._gaussian

    def get_vonmises(self):
        return self._vonmises


# ---- One-shot Scheme ----
class OneShotScheme():
    def __init__(self, n_clusters: int, alpha, beta, init: str = 'k-means++'):
        self._components = [CustomBregman(alpha, beta) for _ in range(n_clusters)]
        self.n_components = n_clusters
        self._mixture = sc.MixtureModel(self._components,
                                       weights=None,
                                       init=init)

    def fit(self, loc_data, dir_data, sample_weight=None, tol=1e-4, max_iter=1000, verbose=False):
        X = np.concatenate([loc_data, dir_data], axis=1)

        _ = self._mixture.fit_em(X,
                                    sample_weight=None,
                                    tol=tol,
                                    max_iter=max_iter,
                                    verbose=verbose,
                                    case='bregman')

    def bic_score(self, loc_data, dir_data):
        X = np.concatenate([loc_data, dir_data], axis=1)
        ll = self._mixture.log_pdf(X).sum()
        n_params = (2+4+2)*self.n_components # 2 + 4 for the Gaussian; 2 for the von Mises
        n_params += 2 # alpha and beta
        return np.log(X.shape[0]) * n_params - 2 * ll

    def plot(self, figsize: float = 6, arrow_scale: float = 12.0, title: str = None, save: bool = False):
        ax = mps.field(show=False, figsize=figsize)

        n = self.n_components
        palette = colors * ((n // len(colors)) + 1)
        for i, cluster in enumerate(self._mixture.get_components()):
            gauss = cluster.get_gaussian()
            vonmises = cluster.get_vonmises()

            col = palette[i]
            mean, cov = gauss.params
            add_ellips(ax, mean, cov, color=col, alpha=0.5)
            x0, y0 = mean

            loc, _ = vonmises.params
            r = vonmises.mean_length  # in [0, 1]
            length = arrow_scale * r  # scale accordingly
            dx, dy = np.cos(loc), np.sin(loc)
            add_arrow(ax, x0, y0,
                      length * dx, length * dy,
                      linewidth=0.8)

        if title is not None:
            plt.title(title)
        if save:
            plt.savefig(f"plots/model_{title}.pdf", bbox_inches='tight')
        plt.show()






def add_ellips(ax, mean, covar, color=None, alpha=0.7):
    eigvals, eigvecs = linalg.eigh(covar)
    lengths = 2.0 * np.sqrt(2.0) * np.sqrt(eigvals)
    direction = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    width, height = max(lengths[0], 3), max(lengths[1], 3)

    ell = mpl.patches.Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,  # or edgecolor=color
        alpha=alpha
    )
    ax.add_patch(ell)
    return ax


def add_arrow(ax, x, y, dx, dy,
              arrowsize=2.5,
              linewidth=2.0,
              threshold=1.8,
              alpha=1.0,
              fc='grey',
              ec='grey'):
    """
    Draw an arrow only if its dx or dy exceed the threshold,
    with both facecolor and edgecolor set to grey by default.
    """
    if np.sqrt(dx ** 2 + dy ** 2) > threshold:
        return ax.arrow(
            x, y, dx, dy,
            head_width=arrowsize,
            head_length=arrowsize,
            linewidth=linewidth,
            fc=fc,
            ec=ec,
            length_includes_head=True,
            alpha=alpha,
            zorder=3,
        )
