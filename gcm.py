#!/usr/bin/env python3

import argparse
from GCM_MRK import GCM_MRK
import multiprocessing
import time

def parse_args():
    parser = argparse.ArgumentParser(description="This Python package is designed for clustering data using a genetic algorithm. The main function, GCM_MRK, executes the genetic algorithm and takes in several parameters, including the dataset to be used (Data), the number of genes in an individual (N_size), and the number of clusters (G).The function also allows for customization of the genetic algorithm through parameters such as population_size, weights, tourn_size, mutation_intensity, mutation_p, crossover_p, and generations. These parameters control the size of the population, the weights for the fitness function, the tournament size for selection, the intensity and probability of mutation, the probability of crossover, and the number of generations for the genetic algorithm, respectively.The function also includes parameters for calculating the Davies-Bouldin Index (DBI), Bayesian Information Criterion (BIC), and Akaike Information Criterion (AIC), which are commonly used metrics in clustering. The GCM_MRK function uses the DEAP (Distributed Evolutionary Algorithms in Python) library for the genetic algorithm. It creates an initial population and defines the genetic operators (selection, crossover, and mutation). It then performs the genetic algorithm flow, keeping track of the best individuals in a hall of fame. The function also calculates and plots the minimum and average fitness over generations, providing a visual representation of the algorithm’s progress.The package also includes the InpPartition and RandPartition classes for creating individuals in the population, and a pairwise_distances function from the utilities module for calculating the distance matrix of the data. The package is designed to be flexible and customizable, allowing users to adjust the parameters of the genetic algorithm to best suit their data and clustering needs. It also leverages multiprocessing for improved performance.Please note that the actual implementation and performance may vary depending on the specific dataset and the chosen parameters. Always refer to the function’s documentation and comments for the most accurate information.")
    parser.add_argument('--Data', required=True, help='The dataset to be used in the genetic algorithm. The dataset msut be formatted as follows: elements that are going to be clustered in the rows (samples) and features in the columns. it is also possible to read the data from a file, as long as the file is a csv or tab delimited file with the mentioned structure')
    parser.add_argument('--N_size', type=int, required=True, help='The number of genes in an individual.')
    parser.add_argument('--G_max', type=int, required=True, help='The number of clusters.')
    parser.add_argument('--all_in_clusters', type=bool, default=True, help='Boolean indicating whether all genes should be assigned to a cluster.')
    parser.add_argument('--input', default=None, help='String that represent the direction of the input for the InpPartition class. Input file must be formatted as follows: csv or tab delimited txt file, with a vector of assignation to clusters per line')
    parser.add_argument('--sep', default=",", help='Separator for the input data.')
    parser.add_argument('--population_size', type=int, default=400, help='The size of the population.')
    parser.add_argument('--weights', type=tuple, default=(-1.0, -1.0, -1.0,), help='The weights for the fitness function.')
    parser.add_argument('--tourn_size', type=int, default=4, help='The tournament size for selection.')
    parser.add_argument('--mutation_intensity', type=int, default=1, help='The intensity of mutation.')
    parser.add_argument('--mutation_p', type=float, default=0.1, help='The probability of mutation.')
    parser.add_argument('--crossover_p', type=float, default=0.8, help='The probability of crossover.')
    parser.add_argument('--generations', type=int, default=200, help='The number of generations for the genetic algorithm.')
    parser.add_argument('--hf_size', type=int, default=4, help='The size of the hall of fame.')
    parser.add_argument('--DBI', type=bool, default=True, help='Boolean indicating whether to calculate the Davies-Bouldin Index.')
    parser.add_argument('--BIC', type=bool, default=True, help='Boolean indicating whether to calculate the Bayesian Information Criterion.')
    parser.add_argument('--AIC', type=bool, default=True, help='Boolean indicating whether to calculate the Akaike Information Criterion.')
    parser.add_argument('--seed', type=int, default=int(time.time()), help='The seed for the random number generator.')
    parser.add_argument('--CPUs_number', type=int, default=multiprocessing.cpu_count() - 1, help='The number of CPUs to use.')
    return parser.parse_args()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    GCM_MRK(args.Data, args.N_size, args.G_max, args.all_in_clusters, args.input, args.sep, args.population_size, args.weights, 
            args.tourn_size, args.mutation_intensity, args.mutation, args.crossover_p, args.generations, args.hf_size,
            args.DBI, args.BIC, args.AIC, args.seed, args.CPUs_number)