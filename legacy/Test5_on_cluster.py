"""
This script will simulate normalized expression data, with each gene having mean 0 and standard deviation 1. 
The data will be generated using Gaussian Mixture Models with varying complexity (number of clusters 
and covariance structure). The data will then be clustered using KMeans and the genetic algorithm. 
The clustering performance will be evaluated using Adjusted Rand Index, Normalized Mutual Information, and Silhouette Score. 
The results will be saved to a CSV file.

"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from GCM_MRK import GCM_MRK
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# In the context of simulated Z normalized expression data, the covariance matrix is simply the correlation matrix.

def generate_cor_matrix(n, structure='full', p=0.5, within_block_range=(0.8, 1), 
                        between_block_range=(-0.3, 0.3), block_sizes=[5]):
    """
    Generate a covariance (correlation) matrix with a specified structure.

    Args:
        n (int): Dimension of the covariance matrix.
        structure (str): Structure of the covariance (correlation) matrix ('diagonal', 'full', 'block').
        p (float): Probability to select the positive range for the correlation values.
        within_block_range (tuple): Range for within-block off-diagonal elements. Values are drawn uniformly from this range.
        between_block_range (tuple): Range for between-block off-diagonal elements. Values are drawn uniformly from this range.
        block_sizes (list): List of block sizes for the 'block' structure. The sum of block sizes should equal to n.

    Returns:
        np.ndarray: Covariance matrix with the specified structure. The matrix is symmetric and its diagonal elements are 1. 
                    For 'block' structure, the matrix contains blocks along the diagonal with strong correlations within each block 
                    and weaker correlations between different blocks.
    """
    if structure == 'diagonal':
        cor = np.eye(n)
    elif structure == 'full':
        L = np.tril(np.random.uniform(-1, 1, (n, n)))  # Generate a lower triangular matrix with random values between -1 and 1
        np.fill_diagonal(L, 1)  # Set diagonal elements to 1
        cor = L @ L.T  # Multiply L by its transpose to get a positive semi-definite correlation matrix
        # Normalize the correlation matrix
        d = np.sqrt(np.diag(cor))
        cor = cor / np.outer(d, d)
        cor = np.clip(cor, -1, 1)  # Clip the correlation matrix to ensure that it is positive semi-definite
    elif structure == 'block':
        cor = np.zeros((n, n))  # Initialize the correlation matrix with zeros
        np.fill_diagonal(cor, 1)  # Set diagonal elements to 1
        start_idx = 0
        for block_size in block_sizes:
            # Generate a symmetric block correlation matrix with strong correlations
            block_cor_values = np.random.uniform(within_block_range[0], within_block_range[1], (block_size, block_size))
            block_cor_values = np.where(np.random.rand(block_size, block_size) > p, block_cor_values, -block_cor_values)  # Randomly assign a sign to each element
            upper_triangular_block_cor = np.triu(block_cor_values)
            block_cor = upper_triangular_block_cor + upper_triangular_block_cor.T - np.diag(upper_triangular_block_cor.diagonal())
            np.fill_diagonal(block_cor, 1)  # Set diagonal elements to 1
            cor[start_idx:start_idx+block_size, start_idx:start_idx+block_size] = block_cor
            start_idx += block_size
        cor = np.clip(cor, -1, 1)  # Clip the correlation matrix to ensure that it is between -1 and 1
    else:
        raise ValueError("Invalid structure. Choose 'diagonal', 'full', or 'block'.")
    
    return cor

def generate_data(n_probes, n_genes, n_clusters, structure, args=None):
    # Generate synthetic data using Gaussian Mixture Models
    X = np.zeros((n_probes, n_genes))
    y_true = np.zeros(n_genes)

    start_idx = 0
    
    if structure != 'block':
        for i in range(n_clusters):
            if i < n_clusters - 1:
                cluster_size = np.random.randint(n_genes // n_clusters, n_genes // (n_clusters - 1))
            else:
                cluster_size = n_genes - start_idx  # Ensure that the last cluster fills the remaining space
            mean = np.zeros(n_genes)
            cov = generate_cor_matrix(n_genes, structure)
            cluster_data = np.random.multivariate_normal(mean, cov, size=cluster_size)
            X[:, start_idx:start_idx + cluster_size] = cluster_data.T
            y_true[start_idx:start_idx + cluster_size] = i + 1
            start_idx += cluster_size
    elif structure == 'block':
        mean = np.zeros(n_genes)
        cov = generate_cor_matrix(n_genes, structure, **args)
        X = np.random.multivariate_normal(mean, cov, size=n_probes).T
        y_true = np.repeat(range(1, n_clusters + 1), n_genes // n_clusters)

    return X, y_true.astype(int)


def run_algorithms(X, n_clusters):
    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    kmeans_labels = kmeans.predict(X) + 1

    # Run the genetic algorithm
    ga_labels = GCM_MRK(X, X.shape[0], n_clusters, True, population_size = 500, generations=400,  tourn_size=10, mutation_intensity=10, mutation_p=0.2, crossover_p=0.8, hf_size=10, normalize=False, Log_Likelihood=True, BIC=False, AIC=False)
    print(set(ga_labels))

    return kmeans_labels, ga_labels

def collect_cluster_metrics(y_true, kmeans_labels, ga_labels, X):
    # Calculate the metrics for KMeans
    ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
    nmi_kmeans = normalized_mutual_info_score(y_true, kmeans_labels)
    silhouette_kmeans = silhouette_score(X, kmeans_labels)

    # Calculate the metrics for the genetic algorithm
    ari_ga = adjusted_rand_score(y_true, ga_labels)
    nmi_ga = normalized_mutual_info_score(y_true, ga_labels)
    silhouette_ga = silhouette_score(X, ga_labels)

    # Return the metrics as a dictionary
    return {
        'KMeans': {'ARI': ari_kmeans, 'NMI': nmi_kmeans, 'Silhouette': silhouette_kmeans},
        'GA': {'ARI': ari_ga, 'NMI': nmi_ga, 'Silhouette': silhouette_ga}
    }

def print_cluster_metrics(y_true, kmeans_labels, ga_labels, X):
    # Calculate and print the metrics for KMeans
    ari_kmeans = adjusted_rand_score(y_true, kmeans_labels)
    nmi_kmeans = normalized_mutual_info_score(y_true, kmeans_labels)
    silhouette_kmeans = silhouette_score(X, kmeans_labels)
    print("KMeans metrics:")
    print(f"ARI: {ari_kmeans}, NMI: {nmi_kmeans}, Silhouette: {silhouette_kmeans}")

    # Calculate and print the metrics for the genetic algorithm
    ari_ga = adjusted_rand_score(y_true, ga_labels)
    nmi_ga = normalized_mutual_info_score(y_true, ga_labels)
    silhouette_ga = silhouette_score(X, ga_labels)
    print("Genetic Algorithm metrics:")
    print(f"ARI: {ari_ga}, NMI: {nmi_ga}, Silhouette: {silhouette_ga}")

def plot_clusters(X, y_true, kmeans_labels, ga_labels):
    # Reduce the data to 2 dimensions using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a figure with 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the true labels
    ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=50)
    ax[0].set_title('True Labels')

    # Plot the KMeans labels
    ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    ax[1].set_title('KMeans Clusters')

    # Plot the genetic algorithm labels
    ax[2].scatter(X_pca[:, 0], X_pca[:, 1], c=ga_labels, cmap='viridis', s=50)
    ax[2].set_title('Genetic Algorithm Clusters')

    # Show the plot
    plt.show()

# Generate synthetic data with varying complexity
results = []
structure = 'block'
for block_sizes in [[200, 300], [250, 25, 25, 200], [100, 100, 100, 100, 100]]:
    n_clusters = len(block_sizes)
    n_genes = sum(block_sizes)
    for within_block_range in [(0.8, 1), (0.6, 0.8), (0.4, 0.6)]:
        for between_block_range in [(-0.3, 0.3), (-0.1, 0.1), (0, 0.1)]:
            args = {
                'within_block_range': within_block_range,
                'between_block_range': between_block_range,
                'block_sizes': block_sizes
                }
            X, y_true = generate_data(500, n_genes, n_clusters, structure, args)
            kmeans_labels, ga_labels = run_algorithms(X, n_clusters)
            metrics = collect_cluster_metrics(y_true, kmeans_labels, ga_labels, X)
            metrics['structure'] = structure
            metrics['n_clusters'] = n_clusters
            metrics['block_sizes'] = block_sizes
            metrics['within_block_range'] = within_block_range
            metrics['between_block_range'] = between_block_range
            results.append(metrics)

            # Print the cluster metrics
            print_cluster_metrics(y_true, kmeans_labels, ga_labels, X)

            # Plot the clusters
            plot_clusters(X, y_true, kmeans_labels, ga_labels)

# Convert the results to a DataFrame and save to a CSV file
df = pd.DataFrame(results)
df.to_csv('../block_cluster_results.csv', index=False)