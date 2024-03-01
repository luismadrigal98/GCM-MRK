import GCM_MRK
import pandas as pd

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = pd.read_csv(url)
iris = iris.to_numpy()

print(iris)
print(iris.shape)

out, log = GCM_MRK.GCM_MRK(iris[:, :-1].astype(float), 149, 8, 4, True, population_size = 500, generations=200, num_reference_points=5, mutation_p=1, mutation_intensity=3, crossover_p=1)

def new_func(out, log):
    """
    Writes the results of the GCM_MRK function to a text file.

    Args:
        out (dict): The output of the GCM_MRK function. It's a dictionary containing 
                    'Partitions' and 'Fitness' as keys.
        log (deap.tools.Logbook): The logbook that contains the statistics of the 
                                  generations.
        file_path (str): The path to the file where the results should be written.

    Returns:
        None
    """    
    with open(r"C:\Users\usuario\Downloads\results_final_front.txt", "w") as file:
        file.write("****************************************************\n")
        file.write("Final partitions\n")
        file.write("____________________________________________________\n")
        file.write('\n')
        for lst in out["Partitions"]:
            file.write(f'{lst}\n')
        file.write('\n')
        file.write("****************************************************\n")
        file.write("Fitness values\n")
        file.write("____________________________________________________\n")
        file.write('\n')
        for lst in out["Fitness"]:
            file.write(f'{lst}\n')
        file.write('\n')
        file.write("****************************************************\n")
        file.write("Log\n")
        file.write("____________________________________________________\n")
        file.write('\n')
        for lst in log:
            file.write(f'{lst}\n')
        file.write('\n')
        file.write("****************************************************\n")
        file.write("Best ever and forever\n")
        file.write("____________________________________________________\n")
        file.write(f'{out["Best_individual"]}\n')

new_func(out, log)