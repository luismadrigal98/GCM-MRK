import GCM_MRK
import pandas as pd
import numpy

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pd.read_csv(url)
iris = iris.to_numpy()

print(iris)
print(iris.shape)

out = GCM_MRK.GCM_MRK(iris[:, :-1], 149, 3, True, population_size=200)

with open(r"C:\Users\usuario\Downloads\results", "w") as file:
    file.write(str(out.items))