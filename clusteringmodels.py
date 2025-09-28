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

# ---- One-shot Scheme ----
class OneShotScheme():
    def __init__(self, n_clusters: int, alpha, beta, init: str = 'k-means++'):
        self._components = [sc.CustomBregman(alpha, beta) for _ in range(n_clusters)]
        self.n_components = n_clusters
        self._mixture = sc.MixtureModel(self._components,
                                       weights=None,
                                       init=init)

    def fit(self, loc_data, dir_data, sample_weight=None, tol=1e-4, max_iter=1000, verbose=False):
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
        n_params = (2+4+2)*self.n_components # 2 + 4 for the Gaussian; 2 for the von Mises
        n_params += 2 # alpha and beta
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
