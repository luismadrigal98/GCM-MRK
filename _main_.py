import argparse
from .GCM-MRK import GCM_MRK

def parse_args():
    parser = argparse.ArgumentParser(description="Your Package Description")
    parser.add_argument('--POPULATION_SIZE', type=int, default=50)
    parser.add_argument('--P_CROSSOVER', type=float, default=0.9)
    parser.add_argument('--P_MUTATION', type=float, default=0.1)
    parser.add_argument('--MAX_GENERATIONS', type=int, default=50)
    parser.add_argument('--HALL_OF_FAME_SIZE', type=int, default=1)
    parser.add_argument('--RANDOM_SEED', type=int, default=1998)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    GCM_MRK(args)