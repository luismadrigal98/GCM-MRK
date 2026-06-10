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
from utilities import normalize_data, write_results_to_file, Pearson_correlation
import operator

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
            halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:len(offspring)] = offspring
        population[len(offspring):] = halloffame.items

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def GCM_MRK(Data, N_size, G_max, all_in_clusters, input = None, sep = ",", population_size = 400, weights = (1.0,), 
            tourn_size = 4, mutation_intensity = 150, mutation_p = 0.1, crossover_p = 0.8, generations = 200, hf_size = 10,
            normalize = True, Log_Likelihood = True, BIC = False, AIC = False, seed = int(time.time()), CPUs_number = multiprocessing.cpu_count() - 1):

    """
    Executes the main genetic algorithm.

    Parameters:
    Data: The dataset to be used in the genetic algorithm.
    N_size: The number of genes in an individual.
    G_max: The number of clusters.
    all_in_clusters: Boolean indicating whether all genes should be assigned to a cluster.
    population_size: The size of the population. Default is 400.
    weights: The weights for the fitness function. Default is (1.0,).
    tourn_size: The tournament size for selection. Default is 4.
    mutation_intensity: The intensity of mutation. Default is 20.
    mutation_p: The probability of mutation. Default is 0.1.
    crossover_p: The probability of crossover. Default is 0.8.
    generations: The number of generations for the genetic algorithm. Default is 200.
    hf_size: The size of the hall of fame. Default is 10.
    normalize: Enables the normalization of the data. Default is True (it is a requirement for the log-likelihood approach).
    Log_Likelihood: Enables the calculation of the log_likelihood based on the correlation coefficient. Default is True.
    seed: The seed for the random number generator. Default is the current time.
    CPUs_number: The number of CPUs to be used. Default is the number of CPUs minus 1.

    Returns:
    None
    """

    os.environ['LOKY_MAX_CPU_COUNT'] = f'{CPUs_number}'

    # Normalize the data:
    if normalize:
        Data = normalize_data(Data, by_sample = True)

    # Correlation matrix calculation
    cor_matrix = Pearson_correlation(Data)

    if input is None:
    # Initialize an instanceof the RandPartition class:
        OneIndividual = RandPartition(N_size, G_max, all_in_clusters)
    else:
    # Initialize an instanceof the InpPartition class:
        OneIndividual = InpPartition(N_size, G_max, input, sep, all_in_clusters)

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
        toolbox.register("Individual_creator", creator.Individual, N_size, G_max, all_in_clusters)
    else:
        creator.create("Individual", InpPartition, fitness=creator.FitnessMulti)
        toolbox.register("Individual_creator", creator.Individual, Data, N_size, G_max, input, sep, all_in_clusters)   

    # Create initial population (generation 0):
    toolbox.register("Population_creator", tools.initRepeat, list, toolbox.Individual_creator)
    population = toolbox.Population_creator(n = population_size)

    # Genetic operators:

    # Selection:
    toolbox.register("select", tools.selTournament, tournsize=tourn_size)

    # Single-point crossover:
    toolbox.register("mate", tools.cxTwoPoint)

    # Flip-bit mutation:
    # indpb: Independent probability for each attribute to be flipped
    if all_in_clusters:
        toolbox.register("mutate", tools.mutUniformInt, low = 1, up = G_max, indpb = mutation_intensity/OneIndividual.__len__())
    else:
        toolbox.register("mutate", tools.mutUniformInt, low = 0, up = G_max, indpb = mutation_intensity/OneIndividual.__len__())
    
    # Registering the evaluation function
    def evaluate(individual):
        return individual.get_values(cor_matrix, Log_Likelihood = Log_Likelihood)
    
    toolbox.register("evaluate", evaluate)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("delta", lambda ind: abs(np.min(ind) - np.mean(ind)))

    # define the hall-of-fame object:
    hof = tools.HallOfFame(hf_size)    

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=crossover_p, mutpb=mutation_p,
                                        ngen=generations, halloffame=hof, stats=stats, verbose=True)

    # print best solution found:
    best = hof.items[0]  

    print("-- Best Ever Individual = ", best[0])  
    print("-- Best Ever items = ", best.items)
    print("-- Best Ever Fitness = ", best.fitness.values)

    # extract statistics:
    minFitnessValues, meanFitnessValues, maxFitnessValues = logbook.select("min", "avg", "max")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.plot(maxFitnessValues, color='blue')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.show()

    return best.items