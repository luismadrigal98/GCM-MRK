import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from GCM_MRK import GCM_MRK
from scipy.stats import norm, multivariate_normal
from sklearn.preprocessing import StandardScaler

def generate_correlation_matrix(genes_per_module, min_corr_within_range, max_corr_within_range, prob_within_max,
                                min_corr_between, max_corr_between, prob_negative):
    # Generate a block-diagonal correlation matrix, where each block corresponds to a module
    blocks = []
    total_genes = sum(genes_per_module)
    correlation_matrix = np.zeros((total_genes, total_genes))
    start = 0
    for genes in genes_per_module:
        block = np.zeros((genes, genes))
        sign_matrix = np.random.choice([-1, 1], size=(genes, genes), p=[prob_negative, 1-prob_negative])
        for i in range(genes):
            for j in range(i+1, genes):
                if np.random.rand() < prob_within_max:
                    block[i, j] = block[j, i] = np.random.uniform(max_corr_within_range[0], 
                                                                max_corr_within_range[1])
                else:
                    block[i, j] = block[j, i] = np.random.uniform(min_corr_within_range[0], 
                                                                min_corr_within_range[1])
        block *= sign_matrix
        np.fill_diagonal(block, 1)  # set the diagonal elements to 1
        blocks.append(block)
        end = start + genes
        correlation_matrix[start:end, start:end] = block
        start = end

    # Introduce correlation between the modules
    for i in range(total_genes):
        for j in range(i+1, total_genes):
            if correlation_matrix[i, j] == 0:
                correlation_matrix[i, j] = correlation_matrix[j, i] = np.random.uniform(min_corr_between, max_corr_between)

    # Ensure the matrix is symmetric
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

    # Ensure the matrix is positive semi-definite
    eigval, eigvec = np.linalg.eig(correlation_matrix)
    eigval[eigval < 0] = 0.0001
    correlation_matrix = eigvec @ np.diag(eigval) @ eigvec.T

    # Normalize the matrix to make it a correlation matrix
    d = np.sqrt(np.diag(correlation_matrix))
    correlation_matrix /= d[:, None]
    correlation_matrix /= d[None, :]

    return correlation_matrix

def simulate_data_with_copula(Modules, genes_per_module, latent_per_module, plants, mu = 0, sig_noise = 1, 
                            min_corr = -0.5, max_corr = 0.5):
    
    raw_data = np.zeros((sum(genes_per_module), plants))

    # Generate a single correlation matrix for all modules
    correlation_matrix = generate_correlation_matrix(latent_per_module * Modules, min_corr, max_corr)

    j = 0
    for module in range(Modules):
        # Generate correlated latent variables using a Gaussian copula
        latent_variables = multivariate_normal.rvs(mean=np.zeros(latent_per_module), 
                                                cov=correlation_matrix[module*latent_per_module:(module+1)*latent_per_module, 
                                                                        module*latent_per_module:(module+1)*latent_per_module], 
                                                size=plants)

        for _ in range(genes_per_module[module]):
            coeff = np.random.choice([-1, 1], latent_per_module)
            for z in range(plants):
                raw_data[j, z] = np.random.normal(mu, sig_noise) + np.sum(coeff * latent_variables[z, :])
            j += 1

    # Normalize the data
    scaler = StandardScaler()
    data = scaler.fit_transform(raw_data.T).T

    # Generate labels for the modules
    labels = np.repeat(np.arange(1, Modules + 1), genes_per_module)

    # Check for NaN and infinite values
    assert not np.isnan(raw_data).any(), "raw_data contains NaN values"
    assert not np.isnan(data).any(), "data contains NaN values"

    return data, labels, raw_data, scaler.mean_, scaler.scale_

# Define number of clusters and cluster sizes
cluster_sizes = [100, 50, 75, 25]

# Define the number of probes (dimensions)
num_probes = 1000

# Generate synthetic data
data, raw, msd, MEAN, SD = simulate_data_with_copula(4, cluster_sizes, 4, num_probes, mu=0, sig_noise=1)

# Calculate the correlation matrix
corr_matrix = np.corrcoef(data, rowvar=True)

# Create a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.show()

# Perform PCA on the original data
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(raw)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# Plot the original data
plt.figure(figsize=(10, 10))
sns.scatterplot(x="principal component 1", y="principal component 2", data=principalDf)
plt.title('PCA plot of original data')
plt.show()

# Perform PCA on the normalized data
principalComponents_normalized = pca.fit_transform(data)
principalDf_normalized = pd.DataFrame(data = principalComponents_normalized, columns = ['principal component 1', 'principal component 2'])

# Plot the normalized data
plt.figure(figsize=(10, 10))
sns.scatterplot(x="principal component 1", y="principal component 2", data=principalDf_normalized)
plt.title('PCA plot of normalized data')
plt.show()

def run_algorithms(X, n_clusters):
    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    kmeans_labels = kmeans.predict(X) + 1

    # Run the genetic algorithm
    ga_labels = GCM_MRK(X, X.shape[0], n_clusters, True, population_size = 4000, generations=600,  tourn_size=5, mutation_intensity=30, mutation_p=0.4, crossover_p=0.8, hf_size=10, normalize=False, Log_Likelihood=True, BIC=False, AIC=False)
    print(set(ga_labels))

    return kmeans_labels, ga_labels

def collect_cluster_metrics(y_true, kmeans_labels, ga_labels, X):
    metrics = {
        'kmeans_silhouette': silhouette_score(X, kmeans_labels),
        'ga_silhouette': silhouette_score(X, ga_labels),
        'kmeans_ari': adjusted_rand_score(y_true, kmeans_labels),
        'ga_ari': adjusted_rand_score(y_true, ga_labels),
    }
    return metrics

def print_cluster_metrics(y_true, kmeans_labels, ga_labels, X):
    metrics = collect_cluster_metrics(y_true, kmeans_labels, ga_labels, X)
    for name, value in metrics.items():
        print(f'{name}: {value}')

def plot_clusters(X, y_true, kmeans_labels, ga_labels):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the true labels
    ax[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50)
    ax[0].set_title('True Labels')

    # Plot the KMeans labels
    ax[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
    ax[1].set_title('KMeans Clusters')

    # Plot the genetic algorithm labels
    ax[2].scatter(X[:, 0], X[:, 1], c=ga_labels, cmap='viridis', s=50)
    ax[2].set_title('Genetic Algorithm Clusters')

    # Show the plot
    plt.show()

# Define the number of clusters for the algorithms
# n_clusters = 4

# Run the algorithms
#kmeans_labels, ga_labels = run_algorithms(data_normalized, n_clusters)

# Print the cluster metrics
# Create true labels based on cluster sizes
#y_true = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
#print_cluster_metrics(y_true, kmeans_labels, ga_labels, data)

# Plot the clusters
#plot_clusters(principalDf_normalized, y_true, kmeans_labels, ga_labels)


def data_simulator(number_base_distributions, number_common_distributions, 
                   Modules, genes_per_module, plants, MaxDependency = 4, mu = 0, sigma = 1,
                   noise = 1,
                   within_range = [0.5, 1.0]):
    """
    Simulates gene expression data by generating base distributions and creating genes as a combination of these distributions.

    Parameters:
    number_base_distributions (int): Number of base normal distributions to generate.
    number_common_distributions (int): Number of shared base distribution between modules.
    Modules (int): Number of modules to generate.
    genes_per_module (list): List of number of genes per module.
    plants (int): Number of plants (samples) to generate.
    MaxDependency (int, optional): Maximum number of base distributions a gene can depend on. Defaults to 4.
    mu (float, optional): Mean of the base normal distributions. Defaults to 0.
    sigma (float, optional): Standard deviation of the base normal distributions. Defaults to 1.

    Returns:
    data_normalized (numpy.ndarray): The simulated gene expression data, normalized to have mean 0 and standard deviation 1 per row.
    data (numpy.ndarray): The simulated gene expression data before normalization.
    msd (dict): Dictionary containing the mean and standard deviation of the rows in the simulated data.
    
    base_distributions = [np.random.normal(mu, sigma, plants) for _ in range(number_base_distributions)]
    data = np.zeros((np.sum(genes_per_module), plants))
    start = 0
    for mod in range(Modules):
        base_dep = np.random.choice(np.arange(number_common_distributions, number_base_distributions), 
                                    replace = False, size=MaxDependency - number_common_distributions)
        dep_array = np.stack([base_distributions[i] for i in np.concatenate((np.arange(number_common_distributions), base_dep))], 
        axis = 0)
        coefficients = np.random.uniform(-1, 1.0001, (len(base_dep) + number_common_distributions, plants))
        for gene in range(genes_per_module[mod]):
            data[start + gene, :] = np.sum(coefficients * dep_array, axis = 0) + np.random.normal(0, noise, plants)
            if np.random.rand() < 0.5:
                data[start + gene, :] *= - np.random.uniform(within_range[0], within_range[1], plants)
        start += genes_per_module[mod]

    msd = {}
    msd['mean'] = np.mean(data, axis = 1, keepdims=True)
    msd['std'] = np.std(data, axis = 1, keepdims=True)

    data_normalized = (data - msd['mean']) / msd['std']

    return data_normalized, data, msd
    """