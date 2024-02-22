### Test framework 1

from internal_indexes import davies_bouldin_index
import numpy as np
from sklearn.metrics import davies_bouldin_score, pairwise_distances
from internal_indexes import bic
import os
from internal_indexes import aic
from utilities import handle_singletons

os.environ['LOKY_MAX_CPU_COUNT'] = '3'

Data = np.array([[1, 1], [1, 3], [5, 1], [5,3], [20 , 20], [-20, -20]])
labels = [1,1,2,2,0,0]

Distance_matrix = pairwise_distances(Data)

print(davies_bouldin_index.davies_bouldin_index(Data, labels=labels))
print(bic.bic(pairwise_distances(Data), labels))
print(aic.aic(pairwise_distances(Data), labels))