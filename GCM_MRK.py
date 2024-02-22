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

def GCM_MRK(Data, N_size, G_max, m, all_in_clusters, input = None, sep = ",", population_size = 400, weights = (-1.0, -1.0, -1.0, 1.0), 
            num_selected_ind = None, num_reference_points = 4, Scales = 0.25, nd = 'log', mutation_intensity = 1, mutation_p = 0.1, 
            crossover_p = 0.8, generations = 200, DBI = True, BIC = True, AIC = True, SI = True, unassigned_penalty = 1, seed = int(time.time()), 
            CPUs_number = multiprocessing.cpu_count() - 1):

    """
    Executes a multiobjective genetic algorithm for gene expression data clustering.
    
    Args:
        Data (np.ndarray): The gene expression dataset 
        N_size (int): Number of genes 
        G_max (int): Max number of clusters
        all_in_clusters (bool): Whether all genes should be assigned to a cluster
        input (list): Initial partition of data, if provided
        population_size (int): Size of the GA population 
        weights (tuple): Weights for DBI, BIC, AIC in fitness  
        num_selected_ind (int): Number of individuals selected each generation
        num_reference_points (int): Number of reference points for NSGA-III
        nd (str): Neighbor degree for NSGA-III selection 
        mutation_intensity (float): Probability of gene mutation
        mutation_p (float): Probability of individual mutation
        crossover_p (float): Probability of crossover  
        generations (int): Number of generations to run GA   
        DBI (bool): Whether to use DBI in fitness  
        BIC (bool): Whether to use BIC in fitness
        AIC (bool): Whether to use AIC in fitness
        seed (int): Random number generator seed
        CPUs_number (int): Number of CPU cores to use
        
    Returns:
        pop (list): Final population of partitions
        logbook (deap.tools.Logbook): Log of GA execution  
    """

    os.environ['LOKY_MAX_CPU_COUNT'] = f'{CPUs_number}'

    ## Defining constants
    # Genetic Algorithm constants:
    POPULATION_SIZE = population_size
    P_CROSSOVER = crossover_p  # probability for crossover
    P_MUTATION = mutation_p   # probability for mutating an individual
    MAX_GENERATIONS = generations # number of optimization rounds
    if num_selected_ind is None: 
        K = 0.25 * POPULATION_SIZE 
    else: 
        K = num_selected_ind
    N_OBJ = DBI + BIC + AIC + SI
    P = num_reference_points

    # Calculated by the algorithm
    Distance_matrix = pairwise_d(Data)

    if input is None:
    # Initialize an instanceof the RandPartition class:
        OneIndividual = RandPartition(N_size, G_max, all_in_clusters)
    else:
    # Initialize an instanceof the InpPartition class:
        OneIndividual = InpPartition(Data, N_size, G_max, input = input, sep = sep, all_in_clusters = all_in_clusters)

    # Set the random seed:
    random.seed(seed)

    toolbox = base.Toolbox()

    # Enabling parallelization of the algorithm:
    # pool = multiprocessing.Pool(processes=CPUs_number)
    # toolbox.register("map", pool.map) 

    # Create, combine and removed duplicates
    ref_points = [tools.uniform_reference_points(N_OBJ, P, Scales) for p, s in zip([P, P * Scales], [1, Scales])]
    ref_points = np.concatenate(ref_points, axis=0)
    _, uniques = np.unique(ref_points, axis=0, return_index=True)
    ref_points = ref_points[uniques]

    # Create the fitness function:
    creator.create("FitnessMulti", base.Fitness, weights = weights)
    
    if input is None:
        creator.create("Individual", RandPartition, fitness=creator.FitnessMulti)
        toolbox.register("Individual_creator", creator.Individual, N_size, G_max, all_in_clusters)
    else:
        creator.create("Individual", InpPartition, fitness=creator.FitnessMulti)
        toolbox.register("Individual_creator", creator.Individual, Data, N_size, G_max, input, sep, all_in_clusters)   

    # Create initial population (generation 0):
    toolbox.register("Population_creator", tools.initRepeat, list, toolbox.Individual_creator)
    population = toolbox.Population_creator(n = POPULATION_SIZE)

    # genetic operators:

    # Tournament selection with tournament size of 3:
    toolbox.register("select", tools.selNSGA3)

    # Two-points crossover:
    toolbox.register("mate", tools.cxTwoPoint)

    # ShuffleIndexes mutation:
    # indpb: Independent probability for each attribute to be flipped
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutation_intensity/OneIndividual.__len__())
    
    # Registering the evaluation function
    def evaluate(individual):
        return individual.get_values(Data, Distance_matrix, m, all_in_clusters, unassigned_penalty, DBI, BIC, 
                                     AIC, SI)
    toolbox.register("evaluate", evaluate)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("range", lambda ind: abs(np.max(ind) - np.min(ind)))    

    # perform the Genetic Algorithm flow with hof feature added:
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max", "range"

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(population)
    logbook.record(gen = 0, evals = len(invalid_ind), **record)
    print(logbook.stream)

    # Genetic Algorithm flow:
    # Begin the generational process
    for gen in range(1, MAX_GENERATIONS + 1):
        offspring = algorithms.varAnd(population, toolbox, P_CROSSOVER, P_MUTATION)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        population = toolbox.select(population + offspring, POPULATION_SIZE, ref_points, nd)

        # Compile statistics about the new population
        record = stats.compile(population)
        logbook.record(gen = gen, evals = len(invalid_ind), **record)
        print(logbook.stream)

    # extract statistics:
    minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "avg", "max")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color = 'red')
    plt.plot(meanFitnessValues, color = 'green')
    plt.plot(maxFitnessValues, color = 'blue')
    plt.xlabel('Generations')
    plt.ylabel('Min / Max / Average Fitness')
    plt.title('Min, Max, and Average fitness over Generations')
    plt.show()

    return population, logbook

if __name__ == "__main__":

    GCM_MRK()