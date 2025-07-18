import numpy as np
from abc import ABC, abstractmethod
from functools import cached_property

class ExponentialFamily(ABC):
    @abstractmethod
    def pdf(self, x):
        """Return the probability density function of x."""
        pass

class UnivariateGaussian(ExponentialFamily):
    def __init__(self, mean=0, variance=1):
        super().__init__()
        if variance <= 0:
            raise ValueError("Variance must be positive.")
        self.mean = mean
        self.variance = variance
        self.natural_param = np.array([
            -self.mean / self.variance,
            -1 / (2 * self.variance)
        ])
        self.dual_param = np.array([
            self.mean,
            self.mean**2 + self.variance
        ])

    def _update_from_mean_variance(self):
        self.natural_param = np.array([
            -self.mean / self.variance,
            -1 / (2 * self.variance)
        ])
        self.dual_param = np.array([
            self.mean,
            self.mean ** 2 + self.variance
        ])

    def set_params(self, params):
        self.mean, self.variance = params
        self._update_from_mean_variance()

    def set_natural_params(self, params):
        theta_1, theta_2 = params
        if theta_2 > 0 or theta_2 < -1e10:
            raise ValueError("Second natural parameter must be negative")
        self.mean = -0.5*theta_1/theta_2
        self.variance = -0.5/theta_2
        self._update_from_mean_variance()

    def set_dual_params(self, params):
        eta_1, eta_2 = params
        if eta_2 - eta_1**2 <= 0:
            raise ValueError("Variance must be positive: second moment minus square of first moment must be > 0.")
        self.mean = eta_1
        self.variance = eta_2 - eta_1**2
        self._update_from_mean_variance()

    def pdf(self, x):
        base = 1 / np.sqrt(2 * np.pi * self.variance)
        exponent = -0.5 * (x - self.mean) ** 2 / self.variance
        return base * np.exp(exponent)

    def __repr__(self):
        return f"UnivariateGaussian(mean={self.mean}, variance={self.variance})"

class MultivariateGaussian(ExponentialFamily):
    def __init__(self, mean=np.zeros(2), covariance=np.eye(2)):
        super().__init__()
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if mean.shape[0] != covariance.shape[0]:
            raise ValueError("Mean vector and covariance matrix dimensions do not match.")
        if np.linalg.det(covariance) <= 0:
            raise ValueError("Covariance matrix must be positive-definite.")

        self.mean = mean
        self.covariance = covariance
        self.d = mean.shape[0]

    def set_params(self, mean: np.ndarray, covariance: np.ndarray):
        pass

    def pdf(self, x):
        diff = x - self.mean
        inv_cov = np.linalg.inv(self.covariance)
        exponent = -0.5 * diff.T @ inv_cov @ diff
        base = 1 / (np.sqrt((2 * np.pi) ** self.d * np.linalg.det(self.covariance)))
        return base * np.exp(exponent)

    def __repr__(self):
        return f"MultivariateGaussian(d={self.d})"




