import random
import time
from deap import base, creator, algorithms, tools
from InpPartition import InpPartition
from RandPartition import RandPartition
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import os
from utilities import pairwise_d

def GCM_MRK(Data, N_size, G_max, all_in_clusters, input = None, sep = ",", population_size = 400, weights = (-1.0, -1.0, -1.0,), 
            tourn_size = 4, mutation_intensity = 1, mutation_p = 0.1, crossover_p = 0.8, generations = 200, hf_size = 4,
            DBI = True, BIC = True, AIC = True, seed = int(time.time()), CPUs_number = multiprocessing.cpu_count() - 1):

    """
    Executes the main genetic algorithm.

    Parameters:
    Data: The dataset to be used in the genetic algorithm.
    N_size: The number of genes in an individual.
    G_max: The number of clusters.
    all_in_clusters: Boolean indicating whether all genes should be assigned to a cluster.
    population_size: The size of the population. Default is 200.
    weights: The weights for the fitness function. Default is (1.0, -1.0, -1.0,).
    tourn_size: The tournament size for selection. Default is 4.
    mutation_intensity: The intensity of mutation. Default is 1.
    mutation_p: The probability of mutation. Default is 0.1.
    crossover_p: The probability of crossover. Default is 0.8.
    generations: The number of generations for the genetic algorithm. Default is 100.
    hf_size: The size of the hall of fame. Default is 10.
    DBI: Boolean indicating whether to calculate the Davies-Bouldin Index. Default is True.
    BIC: Boolean indicating whether to calculate the Bayesian Information Criterion. Default is True.
    AIC: Boolean indicating whether to calculate the Akaike Information Criterion. Default is True.
    seed: The seed for the random number generator. Default is the current time.

    Returns:
    None
    """

    os.environ['LOKY_MAX_CPU_COUNT'] = f'{CPUs_number}'

    ## Defining constants
    # Genetic Algorithm constants:
    POPULATION_SIZE = population_size
    P_CROSSOVER = crossover_p  # probability for crossover
    P_MUTATION = mutation_p   # probability for mutating an individual
    MAX_GENERATIONS = generations # number of optimization rounds
    HALL_OF_FAME_SIZE = hf_size # number of best-ever-seen individuals preserved in memory

    # Calculated by the algorithm
    Distance_matrix = pairwise_distances.pairwise_d(Data)

    if input is None:
    # Initialize an instanceof the RandPartition class:
        OneIndividual = RandPartition(Data, N_size, G_max, sep = sep, all_in_clusters = all_in_clusters)
    else:
    # Initialize an instanceof the InpPartition class:
        OneIndividual = InpPartition(Data, N_size, G_max, input = input, sep = sep, all_in_clusters = all_in_clusters)

    # Set the random seed:
    random.seed(seed)

    toolbox = base.Toolbox()

    # Enabling parallelization of the algorithm:
    # pool = multiprocessing.Pool(processes=CPUs_number)
    # toolbox.register("map", pool.map) 

    # Genetic Algorithm flow:

    # Create the fitness function:
    creator.create("FitnessMulti", base.Fitness, weights=weights)
    
    if input is None:
        creator.create("Individual", RandPartition, fitness=creator.FitnessMulti)
    else:
        creator.create("Individual", InpPartition, fitness=creator.FitnessMulti) 
   
    toolbox.register("Individual_creator", creator.Individual, Data, N_size, G_max, input, sep, all_in_clusters)

    # Create initial population (generation 0):
    toolbox.register("Population_creator", tools.initRepeat, list, toolbox.Individual_creator)
    population = toolbox.Population_creator(n = POPULATION_SIZE)

    # genetic operators:

    # Tournament selection with tournament size of 3:
    toolbox.register("select", tools.selTournament, tournsize=tourn_size)

    # Single-point crossover:
    toolbox.register("mate", tools.cxTwoPoint)

    # Flip-bit mutation:
    # indpb: Independent probability for each attribute to be flipped
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutation_intensity/OneIndividual.__len__())
    
    # Registering the evaluation function
    def evaluate(individual):
        return individual.get_values(Data, Distance_matrix, DBI, BIC, AIC)
    toolbox.register("evaluate", evaluate)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("delta", lambda ind: abs(np.min(ind) - np.mean(ind)))

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)    

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, halloffame=hof, stats=stats, verbose=True)

    # print best solution found:
    best = hof.items[0]  

    print("-- Best Ever Individual = ", best[0])  
    print("-- Best Ever Fitness = ", best.fitness.values)

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.show()

    return best

if __name__ == "__main__":

    GCM_MRK() 