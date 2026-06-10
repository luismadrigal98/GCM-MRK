import re
import numpy as np
from math import pi
from scipy.stats import multivariate_normal

def log_likelihood_cor_calculator(cor, labels):
    """
    Calculate the likelihood of a given clustering solution.

    Args:
        cor (numpy.ndarray): 2D array representing a correlation matrix.
        labels (list or numpy.ndarray): list ot 1D array representing cluster labels.

    Returns:
        float: The total likelihood of the clustering solution.
    """
    # Check inputs
    assert isinstance(cor, np.ndarray) and cor.ndim == 2, "cor must be a 2D numpy array"
    assert isinstance(labels, (list, np.ndarray)) and np.array(labels).ndim == 1, "labels must be a 1D array or list"
    assert cor.shape[0] == cor.shape[1], "cor must be a square matrix"
    assert cor.shape[0] == len(labels), "cor and labels must have compatible shapes"

    log_likelihood = 0
    labels = np.array(labels)

    for s in np.unique(labels):
        # Create a boolean mask for the current cluster
        filter_s = labels == s

        # Calculate the number of samples in the current cluster
        ns = np.sum(filter_s)

        # Calculate the sum of the correlations between the samples in the current cluster
        cs = np.sum(abs(cor[np.outer(filter_s, filter_s)]))
        # << Potential are from improvement, to consider each kind of of correlation type.

        # Calculate the log-likelihood term for the current cluster
        # Using NumPy functions to handle numerical issues
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.log(ns / cs) + (ns - 1) * np.log((ns**2 - ns) / (ns**2 - cs))
            log_likelihood += np.nansum(log_term) ## Review this equation

    # The final log-likelihood is the sum of terms divided by 2
    log_likelihood *= 0.5

    return log_likelihood

def log_likelihood(X, labels):
    """
    Calculate the likelihood of a given clustering solution.

    Args:
        X (numpy.ndarray): 2D array representing a data matrix in the original configuration of interest (rows are the genes).
        labels (list or numpy.ndarray): list ot 1D array representing cluster labels.

    Returns:
        float: The total likelihood of the clustering solution.
    """
    # Check inputs
    assert isinstance(X, np.ndarray) and X.ndim == 2, "X must be a 2D numpy array"
    assert isinstance(cor, np.ndarray) and cor.ndim == 2, "cor must be a 2D numpy array"
    assert cor.shape[0] == cor.shape[1], "cor must be a square matrix"
    assert X.shape[0] == cor.shape[0], "X and cor must have compatible shapes"

    # Transform the labels into a numpy array
    labels = np.array(labels)

    # Extract the number of dimensions (columns)
    k = X.shape[1]

    log_likelihood = 0

    for label in np.unique(labels):
        
        # Set a filter
        filter = labels == label

        # For the current cluster, extract the number of observations
        n = np.sum(filter)

        # Calculate the covariance matrix for the current cluster
        cov = np.cov(X[filter], rowvar=False)

        # Calculate the mean vector:

        mean_vector = np.mean(X[filter], axis=1)

        # Calculate the sum of Mahalanobis distance of each observation from the origin (mean = 0)
        mahalanobis_sum = 0

        for i in range(n):
            mahalanobis_sum += np.dot(X[filter][i] - mean_vector, np.dot(np.linalg.inv(cov), X[filter][i]) - mean_vector)

        # Calculate the likelihood for that term

        log_l_term = n * k * np.log(2 * pi) + n * np.log(np.linalg.det(cov)) + mahalanobis_sum

        log_likelihood += - 0.5 * log_l_term

    return log_likelihood

def bic(log_likelihood, n, p):
    """
    Calculate the Bayesian Information Criterion (BIC) for a given clustering solution.

    Args:
        log_likelihood (float): The log-likelihood of the clustering solution.
        n (int): The number of observations.
        p (int): The number of parameters in the model.

    Returns:
        float: The BIC value for the clustering solution.
    """
    return -2 * log_likelihood + p * np.log(n)