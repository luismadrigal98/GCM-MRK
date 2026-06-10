"""
This script is used to simulate data for the GA testing. The data is generated using a similar method to the one used in the GCM_MRK.py script. 
The data is generated in a way that the first gene is generated randomly, and the rest of the genes are generated using a linear combination 
of the previous genes. The data is then normalized and returned as a numpy array. The script also returns the raw data and the mean and standard 
deviation of the data.

@ Author: John K. Kelly
@ Date: 2024-04-11

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

def simulate_data(Modules, genes_per_module, latent_per_module, plants, mu=0, sigma=1, sig_noise=1, out_as_numpayArray=True): # type: ignore
    """
    Simulates data based on the given parameters.

    Parameters:
    - Modules (int): The number of modules.
    - genes_per_module (list): A list containing the number of genes in each module.
    - latent_per_module (int): The number of latent variables per module.
    - plants (int): The number of plants.
    - mu (float, optional): The mean of the normal distribution used to generate raw data. Default is 0.
    - sigma (float, optional): The standard deviation of the normal distribution used to generate raw data. Default is 1.
    - sig_noise (float, optional): The standard deviation of the noise added to the raw data. Default is 1.
    - out_as_numpayArray (bool, optional): If True, the output data, raw_data, and mnsd will be converted to numpy arrays. Default is True.

    Returns:
    - data (numpy.ndarray or list): The simulated data, normalized based on the mean and standard deviation of each gene.
    - labels (list): The labels indicating the module ID for each gene.
    - raw_data (numpy.ndarray or list): The raw simulated data before normalization.
    - mnsd (numpy.ndarray or list): The mean and standard deviation of each gene in the raw data.

    Raises:
    - AssertionError: If the generated raw_data, mnsd, or data contains NaN values.

    """
def simulate_data(Modules, genes_per_module, latent_per_module, plants, mu = 0, sigma = 1, sig_noise = 1, 
                out_as_numpayArray = True):
    
    raw_data={}
    for j in range(sum(genes_per_module)): # geneid
        raw_data[j]=[-99 for _ in range(plants)]

    j = 0
    for module in range(Modules):
        
        latents = {}
        
        for x in range(latent_per_module):
            latents[x] = []
            
            for _ in range(plants):
                latents[x].append(normal(0.0, sigma))
        
        for _ in range(genes_per_module[module]):
            
            coeff=[]
                
            for x in range(latent_per_module):
                if random() < 0.5:
                    coeff.append(-1)
                else:
                    coeff.append(1)

            for z in range(plants):
                raw_data[j][z] = normal(mu, sig_noise)
                for x in range(len(coeff)):
                    raw_data[j][z] += coeff[x] * latents[x][z]
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
    ga_labels = GCM_MRK(X, X.shape[0], n_clusters, True, population_size = 4000, generations=200,  
                        tourn_size=5, mutation_intensity=5, mutation_p = 0.4, crossover_p = 0.7, 
                        hf_size = 20, normalize=False, Log_Likelihood=True, BIC=False, AIC=False)
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
    
    # Perform PCA on the original data
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the true labels
    ax[0].scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=y_true, cmap='viridis', s=50)
    ax[0].set_title('True Labels')

    # Plot the KMeans labels
    ax[1].scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=kmeans_labels, cmap='viridis', s=50)
    ax[1].set_title('KMeans Clusters')

    # Plot the genetic algorithm labels
    ax[2].scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=ga_labels, cmap='viridis', s=50)
    ax[2].set_title('Genetic Algorithm Clusters')

    # Show the plot
    plt.show()

# Define parameters
Modules = 3
genes_per_module = [40, 40, 100] 
latent_per_module = 4
plants = 1000  # number of plants
mu = 0
sigma = 1
n_clusters = 3  # number of clusters

# Simulate data
data, true_labels, raw_data, mnsd = simulate_data(Modules, genes_per_module, latent_per_module, plants, 
                                                mu = 0, sig_noise= 1, 
                out_as_numpayArray = True)

# Visualize data
visualize_data(data, true_labels)

#Run algorithms
kmeans_labels, ga_labels = run_algorithms(data, n_clusters)

#Print cluster metrics
print_cluster_metrics(true_labels, kmeans_labels, ga_labels, data)

plot_clusters(data, true_labels, kmeans_labels, ga_labels)