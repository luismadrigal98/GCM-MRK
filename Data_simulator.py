import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from GCM_MRK import GCM_MRK

def data_definer(Z1_mean, Z1_std, sizes, num_probes, coef = [], 
                 noise_scaler = 2):
    # Initialize an empty array for the cluster data
    data = np.zeros((sum(sizes), num_probes))
    
    start = 0
    
    Z1 = np.random.normal(Z1_mean, Z1_std, num_probes)
    Z2 = coef[0] * (Z1 / 2)
    Z3 = coef[1] * ((Z1 + Z2) / (Z1 - Z2)) 
    Z4 = coef[2] * (np.abs(Z1 - Z2 + Z3))

    # Generate data for each element in the cluster
    for i in range(len(sizes)):
        for j in range(sizes[i]):
            # Define the expressions for each element
            if i == 0:
                obs = Z1 + np.random.normal(0, 1, num_probes) * noise_scaler
            elif i == 1:
                Z2 = Z2 + np.random.normal(0, 1, num_probes) * noise_scaler
                obs = Z2
            elif i == 2:
                Z3 = Z3 + np.random.normal(0, 1, num_probes) * noise_scaler
                obs = Z3
            elif i == 3:
                Z4 = Z4 + abs(np.random.normal(0, 1, num_probes)) * noise_scaler
                obs = Z4

            # Add the element data to the cluster data
            data[start + j, :] = obs
        start += sizes[i]

    return data

# Define number of clusters and cluster sizes
cluster_sizes = [100, 50, 75, 25]

# Define the number of probes (dimensions)
num_probes = 1000

# Generate synthetic data
data = data_definer(10, 5, cluster_sizes, num_probes, [-0.5, 1.7, 0.1])

# Ensuring that the values are equal or greater than 0 (simulation of RPM data)
data = data - np.min(data)

# Calculate the correlation matrix
corr_matrix = np.corrcoef(data, rowvar=True)

# Create a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.show()

# Normalize the data across the rows
data_normalized = normalize(data, axis=1)

# Perform PCA on the original data
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# Plot the original data
plt.figure(figsize=(10, 10))
sns.scatterplot(x="principal component 1", y="principal component 2", data=principalDf)
plt.title('PCA plot of original data')
plt.show()

# Perform PCA on the normalized data
principalComponents_normalized = pca.fit_transform(data_normalized)
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
n_clusters = 4

# Run the algorithms
kmeans_labels, ga_labels = run_algorithms(data_normalized, n_clusters)

# Print the cluster metrics
# Create true labels based on cluster sizes
y_true = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
print_cluster_metrics(y_true, kmeans_labels, ga_labels, data)

# Plot the clusters
plot_clusters(principalDf_normalized, y_true, kmeans_labels, ga_labels)