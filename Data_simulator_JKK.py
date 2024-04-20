"""
This script is used to simulate data for the GA testing. The data is generated using a similar method to the one used in the GCM_MRK.py script. 
The data is generated in a way that the first gene is generated randomly, and the rest of the genes are generated using a linear combination 
of the previous genes. The data is then normalized and returned as a numpy array. The script also returns the raw data and the mean and standard 
deviation of the data.

@ Author: John K. Kelly
@ Date: 2024-04-09

"""

from random import random
from numpy.random import normal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from GCM_MRK import GCM_MRK
import matplotlib.colors as mcolors

def simulate_data(Modules, genes_per_module, plants, MaxDependency = 4, mu = 0, sigma = 1, negative_influence_chance = 0.5, 
                  out_as_numpayArray = True):
    """
    This function simulates expression data using a linear combination of previous genes. The first gene is generated randomly,
    and the rest of the genes are generated using a linear combination of the previous genes. The data is then normalized and 
    returned as a numpy array. The function also returns the raw data and the mean and standard deviation of the data.

    Args:

    Modules (int): The number of modules to simulate.
    genes_per_module (list): A list containing the number of genes per module for each module.
    plants (int): The number of plants to simulate.
    MaxDependency (int): The maximum number of previous genes that can be used to generate a gene.
    mu (int): The mean of the data.
    sigma (int): The standard deviation of the data.
    negative_influence_chance (float): The chance of a negative influence in the linear combination of genes.
    out_as_numpayArray (bool): A boolean value indicating whether the output should be returned as a numpy array.
    
    Returns:

    data (np.ndarray): The simulated expression data.
    labels (list): The labels for the modules.
    raw_data (np.ndarray): The raw simulated data.
    mnsd (np.ndarray): The mean and standard deviation of the data.
    """
    raw_data = {}
    for j in range(sum(genes_per_module)):  # geneID
        raw_data[j] = [-99 for _ in range(plants)]

    j = 0
    for module in range(Modules):
        for k in range(genes_per_module[module]):
            if k == 0:
                for z in range(plants):
                    raw_data[j][z] = normal(mu, sigma)
            else:
                coeff = []
                for _ in range(min(k, MaxDependency)):
                    rx = random()
                    if rx < negative_influence_chance:
                        coeff.append(-1 + rx)
                    else:
                        coeff.append(rx)
                for z in range(plants):
                    raw_data[j][z] = normal(mu, sigma)
                    for x in range(len(coeff)):
                        raw_data[j][z] -= coeff[x] * raw_data[j - 1 - x][z]
            j += 1

    mnsd = {}
    for j in range(sum(genes_per_module)):
        mnsd[j] = [np.average(raw_data[j]), np.std(raw_data[j])]

    data = raw_data.copy()
    for z in range(plants):
        for j in range(sum(genes_per_module)):
            data[j][z] = (raw_data[j][z] - mnsd[j][0]) / mnsd[j][1]

    if out_as_numpayArray:
        data = np.array([data[j] for j in range(sum(genes_per_module))])
        raw_data = np.array([raw_data[j] for j in range(sum(genes_per_module))])
        mnsd = np.array([mnsd[j] for j in range(sum(genes_per_module))])

    labels = []
    module_ID = 1
    for module in range(Modules):
        module_labels = [module_ID for _ in range(genes_per_module[module])]
        labels.extend(module_labels)
        module_ID += 1

    # After generating raw_data
    assert not np.isnan(raw_data).any(), "raw_data contains NaN values"

    # After calculating mnsd
    assert not np.isnan(mnsd).any(), "mnsd contains NaN values"

    # After normalizing the data
    assert not np.isnan(data).any(), "data contains NaN values"

    return data, labels, raw_data, mnsd

def visualize_data(data_array, labels):
    """
    Visualizes the given data array by creating a correlation heatmap and a scatter plot using PCA.

    Parameters:
    data_array (numpy.ndarray): The input data array.
    labels (list): The true labels for the data points.

    Returns:
    None
    """

    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(data_array, rowvar=True)

    # Create a heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.show()

    # Perform PCA on the original data
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data_array)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    principalDf['label'] = labels  # add labels to the dataframe

    # Create a custom color palette
    unique_labels = np.unique(labels)
    # Define a list of vivid color names
    vivid_colors = ['crimson', 'limegreen', 'darkviolet', 'darkorange', 'gold', 'turquoise', 'royalblue']

    # Create a color palette using the vivid colors
    color_palette = {label: vivid_colors[i % len(vivid_colors)] for i, label in enumerate(unique_labels)}

    # Plot the original data with the vivid color palette
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", palette=color_palette, data=principalDf)
    plt.title('PCA plot of original data')
    plt.show()

def run_algorithms(X, n_clusters):
    """
    Run KMeans and the genetic algorithm for clustering.

    Parameters:
    X (array-like): The input data matrix of shape (n_samples, n_features).
    n_clusters (int): The number of clusters to generate.

    Returns:
    tuple: A tuple containing two arrays:
        - kmeans_labels (array-like): The cluster labels assigned by KMeans.
        - ga_labels (array-like): The cluster labels assigned by the genetic algorithm.
    """
    # Run KMeans
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(X)
    kmeans_labels = kmeans.predict(X) + 1

    # Run the genetic algorithm
    ga_labels = GCM_MRK(X, X.shape[0], n_clusters, True, population_size = 8000, generations=100,  
                        tourn_size=10, mutation_intensity=5, mutation_p = 0.4, crossover_p = 0.7, 
                        hf_size = 50, normalize=False, Log_Likelihood=True, BIC=False, AIC=False)
    print(set(ga_labels))

    return kmeans_labels, ga_labels

def collect_cluster_metrics(y_true, kmeans_labels, ga_labels, X):
    """
    Collects various clustering evaluation metrics for comparing the performance of K-means and Genetic Algorithm (GA) clustering algorithms.

    Parameters:
    - y_true: The true cluster labels.
    - kmeans_labels: The cluster labels assigned by the K-means algorithm.
    - ga_labels: The cluster labels assigned by the Genetic Algorithm (GA) algorithm.
    - X: The input data used for clustering.

    Returns:
    - metrics: A dictionary containing the following clustering evaluation metrics:
        - 'kmeans_ari': Adjusted Rand Index (ARI) score for K-means clustering.
        - 'kmeans_nmi': Normalized Mutual Information (NMI) score for K-means clustering.
        - 'kmeans_silhouette': Silhouette score for K-means clustering.
        - 'ga_ari': Adjusted Rand Index (ARI) score for GA clustering.
        - 'ga_nmi': Normalized Mutual Information (NMI) score for GA clustering.
        - 'ga_silhouette': Silhouette score for GA clustering.
    """
    metrics = {
        'kmeans_ari': adjusted_rand_score(y_true, kmeans_labels),
        'kmeans_nmi': normalized_mutual_info_score(y_true, kmeans_labels),
        'kmeans_silhouette': silhouette_score(X, kmeans_labels),
        'ga_ari': adjusted_rand_score(y_true, ga_labels),
        'ga_nmi': normalized_mutual_info_score(y_true, ga_labels),
        'ga_silhouette': silhouette_score(X, ga_labels),
    }
    return metrics

def print_cluster_metrics(y_true, kmeans_labels, ga_labels, X):
    """
    Prints the cluster metrics for evaluating clustering algorithms.

    Parameters:
    - y_true: The true labels of the data points.
    - kmeans_labels: The labels assigned by the K-means clustering algorithm.
    - ga_labels: The labels assigned by the Genetic Algorithm clustering algorithm.
    - X: The data points.

    Returns:
    None
    """
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

# Define parameters
Modules = 4
genes_per_module = [50, 50, 50, 100]  # assuming you have 50 genes per module
plants = 100  # number of plants
MaxDependency = 3
mu = 0
sigma = 1
negative_influence_chance = 0.5
out_as_numpayArray = True
n_clusters = 4  # number of clusters

# Simulate data
data, true_labels, raw_data, mnsd = simulate_data(Modules, genes_per_module, plants, MaxDependency, mu = 0, sigma = 1, negative_influence_chance = negative_influence_chance, 
                  out_as_numpayArray = True)

# Visualize data
visualize_data(data, true_labels)

# Run algorithms
kmeans_labels, ga_labels = run_algorithms(data, n_clusters)

# Print cluster metrics
print_cluster_metrics(true_labels, kmeans_labels, ga_labels, data)

plot_clusters(data, true_labels, kmeans_labels, ga_labels)