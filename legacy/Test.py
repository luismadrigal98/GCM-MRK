import GCM_MRK
import pandas as pd
from internal_indexes import likelihood_calculator
from utilities import Pearson_correlation, normalize_data

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pd.read_csv(url)
iris = iris.iloc[:, :-1].to_numpy(dtype='float')  # Use iloc for integer-location based indexing

url = "https://web.stanford.edu/~hastie/CASI_files/DATA/leukemia_big.csv"
data = pd.read_csv(url)
data = data.to_numpy(dtype='float')
print(data)
print(data.shape)

# Preprocess the data
# Transpose the data so that rows are observations and columns are features
data = data.transpose()
print(data.shape)

# Run the genetic algorithm
res = GCM_MRK.GCM_MRK(data, 72, 1, True, population_size = 1000, generations=400,  tourn_size=10, mutation_intensity=10, mutation_p=0.2, crossover_p=0.8, hf_size=10, normalize=True, Log_Likelihood=True, BIC=False, AIC=False)

print(res)

from sklearn.metrics import confusion_matrix

# Assuming `res` is a list of predicted labels from your genetic algorithm
predicted_labels = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
data = normalize_data(data)
cor_matrix = Pearson_correlation(data)
likelihood_calculator(cor_matrix, predicted_labels)


# Convert the true labels to a format that matches `res`
true_labels = [1 if (i <= 27 or 39 <= i <= 49 or 55 <= i < 57 or i == 59 or 67 <= i <= 72) else 2 for i in range(1, 73)]
# Create the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

print(cm)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# Plot the result
plt.figure(figsize=(10, 10))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=true_labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Gene Expression Data')
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=predicted_labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Gene Expression Data')
plt.colorbar()
plt.show()