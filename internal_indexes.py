import numpy as np

def likelihood_calculator(cor, labels):
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
        cs = np.sum(cor[np.outer(filter_s, filter_s)])
        # << Potential are from improvement, to consider each kinfdo of correlation tuype.

        # Calculate the log-likelihood term for the current cluster
        # Using NumPy functions to handle numerical issues
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.log(ns / cs) + (ns - 1) * np.log((ns**2 - ns) / (ns**2 - cs))
            log_likelihood += np.nansum(log_term)

    # The final log-likelihood is the sum of terms divided by 2
    log_likelihood *= 0.5

    return log_likelihood

def calculate_aic(log_likelihood, num_params):
    """
    Calculate the Akaike Information Criterion (AIC).

    Args:
        log_likelihood (float): The log-likelihood of the model.
        num_params (int): The number of parameters in the model (number of clusters).

    Returns:
        float: The AIC of the model.
    """
    return 2 * num_params - 2 * log_likelihood


def calculate_bic(log_likelihood, num_params, num_obs):
    """
    Calculate the Bayesian Information Criterion (BIC).

    Args:
        log_likelihood (float): The log-likelihood of the model.
        num_params (int): The number of parameters in the model (number of clusters).
        num_obs (int): The number of observations.

    Returns:
        float: The BIC of the model.
    """
    return np.log(num_obs) * num_params - 2 * log_likelihood