import GCM_MRK
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from internal_indexes import likelihood_calculator
from utilities import normalize_data, Pearson_correlation
from sklearn.mixture import GaussianMixture

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pd.read_csv(url)
iris = iris.iloc[:, :-1].to_numpy(dtype='float')  # Use iloc for integer-location based indexing

print(iris)
print(iris.shape)

iris = normalize_data(iris, by_sample=True)

# Kmeans derived
labels = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,3,3,3,3,1,3,3,3,3,3,3,1,1,3,3,3,3,1,3,1,3,1,3,3,1,1,3,3,3,3,3,1,3,3,3,3,1,3,3,3,1,3,3,3,1,3,3,1]

Distance_matrix = pairwise_distances(iris)
cor = Pearson_correlation(iris)

print(likelihood_calculator(cor, labels))

# GA derived after 40 generations
labels2 = [4, 2, 2, 6, 1, 1, 2, 2, 2, 4, 2, 1, 2, 2, 2, 2, 2, 6, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 5, 1, 2, 1, 2, 5, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 3, 2, 1, 1, 2, 1, 1, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 6, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 5, 1, 2, 1]

print(davies_bouldin_index(iris, labels=labels2, unassigned_penalty=1))
print(bic(pairwise_distances(iris), labels2, m = 2, unassigned_penalty=1))
print(aic(pairwise_distances(iris), labels2, m = 2, unassigned_penalty=1))
print(likelihood_calculator(cor, labels2))

# Non-sense labels
labels3 = [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
print(davies_bouldin_index(iris, labels=labels3, unassigned_penalty=1))
print(bic(pairwise_distances(iris), labels3, m = 2, unassigned_penalty=1))
print(aic(pairwise_distances(iris), labels3, m = 2, unassigned_penalty=1))
print(likelihood_calculator(cor, labels3))

# 1 cluster
labels4 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
print(likelihood_calculator(cor, labels4))

# All singletons
labels5 = list(range(1, 150))
print(likelihood_calculator(cor, labels5))

gmm = GaussianMixture(n_components=3)
gmm.fit(Distance_matrix)
aic = gmm.aic(Distance_matrix)