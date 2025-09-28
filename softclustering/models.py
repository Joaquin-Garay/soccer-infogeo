# models.py

"""
Models
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np

from scipy.special import logsumexp

from .distributions import ExponentialFamily, CustomBregman
from .mixture import MixtureModel
from .metrics import _num_free_params_for_component
from .utils import (
    add_ellips,
    add_arrow,
)

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotsoccer as mps

# grab the default color cycle as a list of hex‐colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

_EPS = 1e-9


# ------ Two-layer Scheme -----
class TwoLayerScheme:
    def __init__(self,
                 loc_mixture: MixtureModel,
                 dir_mixtures: Sequence[MixtureModel]):
        self.loc_mixture = loc_mixture
        self.loc_n_clusters = loc_mixture.n_components
        if len(dir_mixtures) != self.loc_n_clusters:
            raise ValueError("Components in loc mixture and number of dir mixture don't match")
        self.dir_mixtures = dir_mixtures

    def fit(self,
            loc_data: np.ndarray,
            dir_data: np.ndarray,
            sample_weight: Optional[Sequence[float]] = None,
            tol: float = 1e-4,
            max_iter: int = 1000,
            verbose: bool = False,
            case: str = "classic",
            ) -> None:
        loc_data = np.asarray(loc_data, dtype=float)
        dir_data = np.asarray(dir_data, dtype=float)
        n_obs = loc_data.shape[0]
        if n_obs != dir_data.shape[0]:
            raise ValueError("Location and direction number of observation don't match")

        _ = self.loc_mixture.fit(loc_data,
                                 sample_weight=None,
                                 tol=tol,
                                 max_iter=max_iter,
                                 verbose=verbose,
                                 case=case)

        # include a jitter in the posteriors probabilities
        loc_posteriors = self.loc_mixture.get_posteriors(loc_data) + _EPS
        for j in range(self.loc_n_clusters):
            _ = self.dir_mixtures[j].fit(dir_data,
                                         sample_weight=loc_posteriors[:, j],
                                         tol=tol,
                                         max_iter=max_iter,
                                         verbose=verbose,
                                         case=case)

    def log_pdf(self, loc_data: np.ndarray, dir_data: np.ndarray) -> np.ndarray:
        """
        Returns log p(x). Shape: (N,)
        """
        loc_posteriors = self.loc_mixture.get_posteriors(loc_data) + _EPS  # (N,K)
        dir_log_pdf_array = [self.dir_mixtures[k].log_pdf(dir_data)[:, None]  # (N,1)
                             for k in range(self.loc_n_clusters)]
        dir_log_pdf = np.concatenate(dir_log_pdf_array, axis=1)  # (N,K)
        return logsumexp(np.log(loc_posteriors) + dir_log_pdf, axis=1)  # (N,)

    def pdf(self, loc_data: np.ndarray, dir_data: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(loc_data, dir_data))

    def bic_score(self, loc_data, dir_data) -> float:
        """Bayesian Information Criterion (lower is better)."""

        loc_n_params = self.loc_mixture.n_components - 1  # prior parameters
        loc_n_params += _num_free_params_for_component(self.loc_mixture.components[0]) * self.loc_n_clusters

        dir_n_params = 0
        for k in range(self.loc_n_clusters):
            dir_mixture = self.dir_mixtures[k]
            dir_n_params += dir_mixture.n_components - 1  # prior parameters
            dir_n_params += _num_free_params_for_component(dir_mixture.components[0]) * dir_mixture.n_components

        p = dir_n_params + loc_n_params
        ll = self.log_pdf(loc_data, dir_data).sum()
        return np.log(loc_data.shape[0]) * p - 2 * ll

    def plot(self,
             figsize: float = 6,
             arrow_scale: float = 12.0,
             title: str = None,
             save: bool = False):
        """
        Plot every (Gaussian + VonMises arrows) on one shared Axes,
        using a different color per cluster, and arrow lengths proportional to mean length r.
        """
        ax = mps.field(show=False, figsize=figsize)

        n = self.loc_mixture.n_components
        palette = colors * ((n // len(colors)) + 1)

        for i, (loc, direction) in enumerate(zip(self.loc_mixture.components,
                                                 self.dir_mixtures)):
            col = palette[i]
            mean, cov = loc.params
            add_ellips(ax, mean, cov, color=col, alpha=0.5)
            x0, y0 = mean

            for vonm in direction.components:
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


# ---- One-shot Scheme ----
class OneShotScheme():
    def __init__(self,
                 n_clusters: int,
                 alpha: float,
                 beta: float,
                 init: str = 'k-means++'):
        self._components = [CustomBregman(alpha, beta) for _ in range(n_clusters)]
        self.n_components = n_clusters
        self._mixture = MixtureModel(self._components,
                                     weights=None,
                                     init=init)

    def fit(self,
            loc_data: np.ndarray,
            dir_data: np.ndarray,
            sample_weight=None,
            tol: float = 1e-4,
            max_iter: int = 1000,
            verbose: bool = False):
        X = np.concatenate([loc_data, dir_data], axis=1)

        _ = self._mixture.fit(X,
                              sample_weight=None,
                              tol=tol,
                              max_iter=max_iter,
                              verbose=verbose,
                              case='bregman')

    def bic_score(self, loc_data, dir_data):
        X = np.concatenate([loc_data, dir_data], axis=1)
        ll = self._mixture.log_pdf(X).sum()
        n_params = (2 + 3 + 2) * self.n_components  # 2 + 3 for the Gaussian; 2 for the von Mises
        n_params += 2  # alpha and beta
        return np.log(X.shape[0]) * n_params - 2 * ll

    def plot(self, figsize: float = 6, arrow_scale: float = 12.0, title: str = None, save: bool = False):
        ax = mps.field(show=False, figsize=figsize)

        n = self.n_components
        palette = colors * ((n // len(colors)) + 1)
        for i, cluster in enumerate(self._mixture.components):
            gauss = cluster.gaussian
            vonmises = cluster.vonmises

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
