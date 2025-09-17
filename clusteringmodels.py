import numpy as np
import warnings
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.special import i0e, i1e, logsumexp
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import LocalOutlierFactor

import softclustering as sc

# ------ Two-layer Scheme -----
class TwoLayerScheme:
    def __init__(self,
                 loc_mixture: sc.MixtureModel,
                 dir_mixtures: list[sc.MixtureModel]):
        self.loc_mixture = loc_mixture
        self.loc_n_clusters = loc_mixture.n_components
        if len(dir_mixtures) != self.loc_n_clusters:
            ValueError("Components in loc mixture and number of dir mixture don't match")
        self.dir_mixtures = dir_mixtures

    def fit(self, loc_data, dir_data, sample_weight=None, tol=1e-4, max_iter=1000, verbose=False, case="classic"):
        loc_data = np.asarray(loc_data, dtype=float)
        dir_data = np.asarray(dir_data, dtype=float)
        N = loc_data.shape[0]
        if N != dir_data.shape[0]:
            raise ValueError("Location and direction number of observation don't match")
        _ = self.loc_mixture.fit_em(loc_data,
                                sample_weight=None,
                                tol=tol,
                                max_iter=max_iter,
                                verbose=verbose,
                                case=case)
        loc_posteriors = self.loc_mixture.get_posteriors(loc_data)
        for j in range(self.loc_n_clusters):
            _ = self.dir_mixtures[j].fit_em(dir_data,
                                   sample_weight=loc_posteriors[:, j],
                                   case=case)


    def log_pdf(self, loc_data: np.ndarray, dir_data: np.ndarray):
        """
        Returns log p(x)
        Shape: (N,)
        """
        loc_posteriors = self.loc_mixture.get_posteriors(loc_data) #(N,K)
        dir_log_pdf_array = [self.dir_mixtures[k].log_pdf(dir_data)[:, None] # (N,1)
                                for k in range(self.loc_n_clusters) ]
        dir_log_pdf = np.concatenate(dir_log_pdf_array, axis=1) # (N,K)
        return logsumexp(np.log(loc_posteriors) + dir_log_pdf, axis=1) # (N,)

    def pdf(self, loc_data: np.ndarray, dir_data: np.ndarray):
        return np.exp(self.log_pdf(loc_data, dir_data))

    def bic_score(self, loc_data, dir_data):
        loc_n_params = self.loc_mixture.n_components - 1 #prior parameters
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