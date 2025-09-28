#models.py

"""
Models
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np
import warnings
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.special import i0e, i1e, logsumexp
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor

from .distributions import ExponentialFamily
from .mixture import MixtureModel
from .metrics import _num_free_params_for_component

from scipy import linalg
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

    def plot(self, figsize: float = 6, arrow_scale: float = 12.0, title: str = None, save: bool = False):
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
