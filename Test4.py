import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from GCM_MRK import GCM_MRK
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_cov_matrix(n, structure='heterogeneous', off_diag_range=(-0.5, 0.5)):
    """
    Generate a covariance matrix with a specified structure.

    Args:
        n (int): Dimension of the covariance matrix.
        structure (str): Structure of the covariance matrix ('diagonal', 'full', 'heterogeneous').
        off_diag_range (tuple): Range for off-diagonal elements (if applicable).

    Returns:
        np.ndarray: Covariance matrix with the specified structure.
    """
    if structure == 'diagonal':
        cov = np.diag(np.random.rand(n))
    elif structure == 'full':
        cov = np.random.rand(n, n)
        cov = cov @ cov.T  # Ensure positive semi-definiteness
    elif structure == 'heterogeneous':
        cov = np.random.rand(n, n)
        cov = cov @ cov.T + np.eye(n)  # Ensure positive definiteness
        np.fill_diagonal(cov, np.random.rand(n))
        cov[np.triu_indices(n, k=1)] = np.random.uniform(off_diag_range[0], off_diag_range[1], size=n*(n-1)//2)
        cov = cov + cov.T - np.diag(np.diag(cov))  # Ensure symmetry
    else:
        raise ValueError("Invalid structure. Choose 'diagonal', 'full', or 'heterogeneous'.")

    return cov

def generate_data(n_samples, n_features, n_clusters, cluster_std, structure):
    # Generate synthetic data using Gaussian Mixture Models
    X = np.zeros((n_samples, n_features))
    y_true = np.zeros(n_samples)

    start_idx = 0
    for i in range(n_clusters):
        if i < n_clusters - 1:
            cluster_size = np.random.randint(n_samples // n_clusters, n_samples // (n_clusters - 1))
        else:
            cluster_size = n_samples - start_idx  # Ensure that the last cluster fills the remaining space
        mean = np.random.randn(n_features)
        cov = generate_cov_matrix(n_features, structure) * cluster_std  # Scale the covariance matrix by cluster_std
        cluster_data = np.random.multivariate_normal(mean, cov, size=cluster_size)
        X[start_idx:start_idx + cluster_size] = cluster_data
        y_true[start_idx:start_idx + cluster_size] = i + 1
        start_idx += cluster_size

    return X, y_true.astype(int)

def run_algorithms(X, n_clusters):
    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    kmeans_labels = kmeans.predict(X) + 1

    # Run the genetic algorithm
    ga_labels = GCM_MRK(X, X.shape[0], n_clusters, True, population_size = 500, generations=400,  tourn_size=10, mutation_intensity=10, mutation_p=0.2, crossover_p=0.8, hf_size=10, normalize=True, Log_Likelihood=True, BIC=False, AIC=False)
    print(set(ga_labels))

    return kmeans_labels, ga_labels

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
for structure in ['diagonal', 'full', 'heterogeneous']:
    for n_clusters in range(2, 6):
        for cluster_std in [0.5, 1.0, 1.5, 2.0]:
            print(f"Running tests for {n_clusters} clusters with standard deviation {cluster_std} and {structure} covariance structure")
            X, y_true = generate_data(500, 100, n_clusters, cluster_std, structure)
            kmeans_labels, ga_labels = run_algorithms(X, n_clusters)
            print_cluster_metrics(y_true, kmeans_labels, ga_labels, X)
            plot_clusters(X, y_true, kmeans_labels, ga_labels)  # Plot the clusters