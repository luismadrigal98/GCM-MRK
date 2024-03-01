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
from utilities import pairwise_d, handle_singletons, write_results_to_file, normalize_data, range_per_index

def GCM_MRK(Data, N_size, G_max, m, all_in_clusters, input = None, output = None, sep = ",", population_size = 400, weights = (-1.0, -1.0, -1.0, 1.0), 
            num_reference_points = 4, Scales = None, nd = 'log', mutation_intensity = 1, mutation_p = 0.1, 
            crossover_p = 0.8, generations = 200, DBI = True, BIC = True, AIC = True, SI = True, unassigned_penalty = 1, seed = int(time.time()), 
            CPUs_number = multiprocessing.cpu_count() - 1, plot_results = True, normalize = True):

    """
    Executes a multiobjective genetic algorithm for gene expression data clustering.
    
    Args:
        Data (np.ndarray): The gene expression dataset 
        N_size (int): Number of genes 
        G_max (int): Max number of clusters
        all_in_clusters (bool): Whether all genes should be assigned to a cluster
        input (list): Initial partition of data, if provided
        output (str): Path to the file where the results should be written
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

    # Check if Data is a valid numpy array
    if not isinstance(Data, np.ndarray):
        raise ValueError("Data must be a numpy array")

    # Check if N_size, G_max, m are integers
    if not all(isinstance(i, int) for i in [N_size, G_max, m]):
        raise ValueError("N_size, G_max, m must be integers")

    # Check if all_in_clusters is a boolean
    if not isinstance(all_in_clusters, bool):
        raise ValueError("all_in_clusters must be a boolean")

    os.environ['LOKY_MAX_CPU_COUNT'] = f'{CPUs_number}'

    ## Defining constants
    # Genetic Algorithm constants:
    POPULATION_SIZE = population_size
    P_CROSSOVER = crossover_p  # probability for crossover
    P_MUTATION = mutation_p   # probability for mutating an individual
    MAX_GENERATIONS = generations # number of optimization rounds
    N_OBJ = DBI + BIC + AIC + SI
    P = num_reference_points

    # Normalize the data
    if normalize:
        Data = normalize_data(Data)

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
    if Scales is None:
        ref_points = tools.uniform_reference_points(N_OBJ, P)

    else:
        ref_points = [tools.uniform_reference_points(N_OBJ, P, Scales) for p, s in zip([P, P * Scales], [1, Scales])]
        ref_points = np.concatenate(ref_points, axis=0)
        _, uniques = np.unique(ref_points, axis=0, return_index=True)
        ref_points = ref_points[uniques]

    # Create the fitness function:
    w = []
    for index, weight in zip([DBI, AIC, BIC, SI], weights):
        if index:
            w.append(weight)
    w = tuple(w)

    creator.create("FitnessMulti", base.Fitness, weights = w)
    
    if input is None:
        creator.create("Individual", RandPartition, fitness=creator.FitnessMulti)
        toolbox.register("Individual_creator", creator.Individual, N_size, G_max, all_in_clusters)
    else:
        creator.create("Individual", InpPartition, fitness=creator.FitnessMulti)
        toolbox.register("Individual_creator", creator.Individual, Data, N_size, G_max, input, sep, all_in_clusters)   

    # Create initial population (generation 0):
    toolbox.register("Population_creator", tools.initRepeat, list, toolbox.Individual_creator)
    population = toolbox.Population_creator(n = POPULATION_SIZE)

    # Genetic operators:

    # Selection with NSGA3:
    toolbox.register("select", tools.selNSGA3)

    # Two-points crossover:
    toolbox.register("mate", tools.cxTwoPoint)

    # ShuffleIndexes mutation:
    # indpb: Independent probability for each attribute to be flipped
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutation_intensity/OneIndividual.__len__())
    
    # Registering the evaluation function
    def evaluate(individual):
        try:
            return individual.get_values(Data, Distance_matrix, m, unassigned_penalty, DBI, BIC, AIC, SI)
        except ValueError as e:
            print("Error evaluating individual:", individual)
            print("Exception:", e)
            print("A new individual will be created at random.")
            # Replace the individual with a new one generated at random
            new_individual = toolbox.Individual_creator()
            new_individual.items = list(handle_singletons(Distance_matrix, new_individual.items, all_in_clusters))
            individual.items[:] = new_individual.items[:]
            return toolbox.evaluate(individual)
    
    toolbox.register("evaluate", evaluate)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("range", range_per_index)

    # GA flow:
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max", "range"

    # Evaluate the individuals with an invalid fitness
    
    # Handling the singletons in the first generation
    for ind in population:
        ind.items = list(handle_singletons(Distance_matrix, ind.items, all_in_clusters))

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

        for progeny in offspring:
            progeny.items = list(handle_singletons(Distance_matrix, progeny.items, all_in_clusters))

        for ind in population:
            ind.items = list(handle_singletons(Distance_matrix, ind.items, all_in_clusters))
        
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
    if plot_results:
        sns.set_style("whitegrid")

        # List of fitness indexes
        fitness_indexes = (np.array(['DBI', 'BIC', 'AIC', 'SI'])[DBI, BIC, AIC, SI]).flatten()
        print(fitness_indexes)

        _, axs = plt.subplots(len(minFitnessValues[0]), figsize=(10, 6*len(minFitnessValues[0])))
        for i in range(len(minFitnessValues[0])):
            axs[i].plot([gen[i] for gen in minFitnessValues], color = 'red', label='Min')
            axs[i].plot([gen[i] for gen in meanFitnessValues], color = 'green', label='Average')
            axs[i].plot([gen[i] for gen in maxFitnessValues], color = 'blue', label='Max')
            axs[i].set_xlabel('Generations')
            axs[i].set_ylabel('Fitness')
            axs[i].set_title(f'Min, Max, and Average {fitness_indexes[i]} over Generations')
            axs[i].legend()

        # Adjust the space between subplots
        plt.subplots_adjust(hspace=0.8)

        plt.show()
    
    partitions = []
    fitness = []

    for ind in population:
        partitions.append(ind.items)
        fitness.append(ind.fitness.values)

    # Find the utopian point
    utopian_point = [float('inf')] * N_OBJ
    for ind in population:
        for i, val in enumerate(ind.fitness.values):
            if val < utopian_point[i]:
                utopian_point[i] = val

    # Find the individual closest to the utopian point
    best_individual = None
    best_distance = float('inf')
    for ind in population:
        distance = sum((val - utopian_val) ** 2 for val, utopian_val in zip(ind.fitness.values, utopian_point))
        if distance < best_distance:
            best_individual = ind
            best_distance = distance

    out = {'Partitions': partitions, 'Fitness': fitness, "Best_individual": best_individual.items}

    if output is not None:
        write_results_to_file(out, logbook, output)
        return out, logbook
    else:
        return out, logbook