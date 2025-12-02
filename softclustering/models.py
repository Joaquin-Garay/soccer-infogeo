# models.py

"""
Models
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np

from scipy.special import logsumexp
from sklearn.metrics.cluster import entropy

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
            c_step: bool = False,
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
                                 case=case,
                                 c_step=c_step)

        # include a jitter in the posteriors probabilities
        loc_posteriors = self.loc_mixture.get_posteriors(loc_data) + _EPS

        # C-step: One-hot encoding of posterior matrix
        if c_step:
            idx = np.argmax(loc_posteriors, axis=1)  # shape (N,)
            one_hot = np.zeros_like(loc_posteriors, dtype=float)
            one_hot[np.arange(loc_posteriors.shape[0]), idx] = 1.0
            if np.any(one_hot.sum(axis=0) == 0):
                # there is an empty cluster
                raise ValueError("Empty cluster")
            else:
                loc_posteriors = one_hot

        for j in range(self.loc_n_clusters):
            _ = self.dir_mixtures[j].fit(dir_data,
                                         sample_weight=loc_posteriors[:, j],
                                         tol=tol,
                                         max_iter=max_iter,
                                         verbose=verbose,
                                         case=case,
                                         # c_step=c_step,
                                         )

    def log_pdf(self, loc_data: np.ndarray, dir_data: np.ndarray) -> np.ndarray:
        """
        Returns log p(x). Shape: (N,)
        """
        loc_pdf = self.loc_mixture.get_posteriors(loc_data) + _EPS  # (N,K)
        loc_pdf *= self.loc_mixture.pdf(loc_data)[:, None]
        dir_log_pdf_array = [self.dir_mixtures[k].log_pdf(dir_data)[:, None]  # (N,1)
                             for k in range(self.loc_n_clusters)]
        dir_log_pdf = np.concatenate(dir_log_pdf_array, axis=1)  # (N,K)
        return logsumexp(np.log(loc_pdf) + dir_log_pdf, axis=1)  # (N,)

    def pdf(self, loc_data: np.ndarray, dir_data: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf(loc_data, dir_data))

    def bic_penalty_term(self, n_obs):
        """ returns number of free parameters times log(n_obs) """
        loc_n_params = self.loc_mixture.n_components - 1  # prior parameters
        loc_n_params += _num_free_params_for_component(self.loc_mixture.components[0]) * self.loc_n_clusters

        dir_n_params = 0
        for k in range(self.loc_n_clusters):
            dir_mixture = self.dir_mixtures[k]
            dir_n_params += dir_mixture.n_components - 1  # prior parameters
            dir_n_params += _num_free_params_for_component(dir_mixture.components[0]) * dir_mixture.n_components

        p = dir_n_params + loc_n_params
        return np.log(n_obs) * p

    def bic_score(self, loc_data, dir_data) -> float:
        """Bayesian Information Criterion (lower is better)."""
        loc_data = np.asarray(loc_data, dtype=float)
        dir_data = np.asarray(dir_data, dtype=float)

        penalty = self.bic_penalty_term(loc_data.shape[0])
        ll = self.log_pdf(loc_data, dir_data).sum()
        return penalty - 2 * ll

    def completed_bic_score(self, loc_data, dir_data):
        loc_data = np.asarray(loc_data, dtype=float)
        dir_data = np.asarray(dir_data, dtype=float)

        penalty = self.bic_penalty_term(loc_data.shape[0])

        loc_posteriors = self.loc_mixture.get_posteriors(loc_data) + _EPS
        idx_loc = np.argmax(loc_posteriors, axis=1)  # shape (N,)
        log_prior_loc = np.log(self.loc_mixture.weights[idx_loc])  # shape (N,)

        log_expfam_loc = np.zeros_like(log_prior_loc)
        log_prior_dir = np.zeros_like(log_prior_loc)
        log_expfam_dir = np.zeros_like(log_prior_loc)

        for i, idx in enumerate(idx_loc):
            log_expfam_loc[i] = self.loc_mixture.components[idx].log_pdf(loc_data[i, :])
            dir_mixture = self.dir_mixtures[idx]
            dir_posteriors = dir_mixture.get_posteriors(dir_data)[i, :] + _EPS  # shape (K, )
            idx_dir = np.argmax(dir_posteriors)  # scalar
            log_prior_dir[i] = np.log(dir_mixture.weights[idx_dir])
            log_expfam_dir[i] = dir_mixture.components[idx_dir].log_pdf(dir_data[i, :])

        complete_data_likelihood = (log_prior_loc + log_expfam_loc
                          + log_prior_dir + log_expfam_dir).sum()

        return penalty - 2.0 * complete_data_likelihood

    def plot(self,
             figsize: float = 6,
             arrow_scale: float = 12.0,
             name: str = None,
             show_title: bool = False,
             save: bool = False,
             show: bool = True):
        """
        Plot every (Gaussian + VonMises arrows) on one shared Axes,
        using a different color per cluster, and arrow lengths proportional to mean length r.
        """
        ax = mps.field(show=False, figsize=figsize)

        cmap = plt.cm.Blues

        for i, (loc, direction) in enumerate(zip(self.loc_mixture.components,
                                                 self.dir_mixtures)):
            prior = self.loc_mixture.weights[i]
            # print(f"prior {i}: {prior*100:.2f}%")
            col = cmap(0.2 + 0.8 * prior)
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

        if show_title:
            plt.title(name)
        if save:
            plt.savefig(f"plots/model_{name}.pdf", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()


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
            verbose: bool = False,
            case: str = "classic",
            c_step: bool = False,
            ):

        X = np.concatenate([loc_data, dir_data], axis=1)

        _ = self._mixture.fit(X,
                              sample_weight=None,
                              tol=tol,
                              max_iter=max_iter,
                              verbose=verbose,
                              case=case,
                              c_step=c_step, )

    def bic_score(self, loc_data, dir_data):
        X = np.concatenate([loc_data, dir_data], axis=1)
        ll = self._mixture.log_pdf(X).sum()
        n_params = (2 + 3 + 2) * self.n_components  # 2 + 3 for the Gaussian; 2 for the von Mises
        n_params += 2  # alpha and beta
        return np.log(X.shape[0]) * n_params - 2 * ll

    def completed_bic_score(self, loc_data, dir_data):
        X = np.concatenate([loc_data, dir_data], axis=1)
        bic = self.bic_score(loc_data, dir_data)

        posterior = self._mixture.get_posteriors(X)
        idx = np.argmax(posterior, axis=1)  # shape (N,)
        one_hot = np.zeros_like(posterior, dtype=float)
        one_hot[np.arange(posterior.shape[0]), idx] = 1.0

        entropy = -1.0 * np.sum(one_hot * np.log(posterior + _EPS))

        return bic + 2.0 * entropy

    def plot(self,
             figsize: float = 6,
             arrow_scale: float = 12.0,
             name: str = None,
             show_title: bool = False,
             save: bool = False,
             show: bool = True):

        ax = mps.field(show=False, figsize=figsize)

        cmap = plt.cm.Blues
        for i, cluster in enumerate(self._mixture.components):
            gauss = cluster.gaussian
            vonmises = cluster.vonmises

            prior = self._mixture.weights[i]
            # print(f"prior {i}: {prior*100:.2f}%")
            col = cmap(0.2 + 0.8 * prior)

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

        if show_title:
            plt.title(name)
        if save:
            plt.savefig(f"plots/model_{name}.pdf", bbox_inches='tight')
            # plt.savefig(f"plots/model_{name}.png", bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
