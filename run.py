import os, sys, argparse
from collections import OrderedDict

from CMAES import CMA
from SPSO2011 import PSO
from ACOR import ACOR
from optproblems.cec2005 import CEC2005
from boundary import Boundary

import numpy as np
import pandas as pd


class Algo:
    def __init__(self, n_points=12, dimension=2, function_id=0, algo_type='CMA', 
                 max_evaluations=1e4, verbose=False, plot=0, fig_dir=None):

        print('\n%s: Solving F%d in %dD with population size %d...\n' % 
              (algo_type, function_id+1, dimension, n_points))

        # Parameters for optproblems
        self.n_points = n_points
        self.dimension = dimension 
        self.function_id = function_id
        self.function = CEC2005(dimension)[function_id].objective_function 
        self.max_bounds = Boundary(dimension, function_id).max_bounds
        self.min_bounds = Boundary(dimension, function_id).min_bounds

        # Parameters for termination 
        self.FE = 0
        self.iteration = 0
        self.max_evaluations = max_evaluations*dimension
        self.termination_error = 1e-8
        self.should_terminate = False 

        # Parameters for logging
        self.optimal_position = CEC2005(dimension)[function_id].get_optimal_solutions()[0].phenome
        self.optimal_fitness = self.function(self.optimal_position)
        self.best_position = np.zeros_like(self.optimal_position)
        self.best_fitness = np.inf
        self.verbose = verbose
        self.plot = plot
        self.stats = OrderedDict([
                        ('iteration', []),
                        ('FEs', []),
                        ('error', []),
                        ('best_fitness', []),
                        ('best_position', [])
                     ])


        if algo_type=='PSO':
            self.algo = PSO( self.obj, self.n_points, self.dimension,
                             min_bounds = self.min_bounds, 
                             max_bounds = self.max_bounds )
        elif algo_type=='ACOR':
            self.algo = ACOR(self.obj, self.dimension, 
                             min_bounds = self.min_bounds, 
                             max_bounds = self.max_bounds,
                             ants_num = 2,
                             archive_size = self.n_points, # 50
                             q = 1e-4, #1e-4, 0.1, 0.3, 0.5, 0.9
                             xi = 0.85
                            )
        else:
            self.algo = CMA( self.obj, self.n_points, self.dimension,
                             min_bounds = self.min_bounds, 
                             max_bounds = self.max_bounds )


    def obj(self, x):
        self.FE += 1
        fitness = self.function(x)

        if self.FE >= self.max_evaluations:
            self.should_terminate = True
            raise Exception('Resources Exhausted!')

        error = fitness - self.optimal_fitness
        if error < self.termination_error:
            self.best_fitness = fitness
            self.best_position = x
            self.should_terminate = True
            raise Exception('Found Optima!')

        return fitness


    def run(self):
        try:
            best_position, best_fitness = self.algo.run() 
        except Exception as e:
            print(e)
            self.should_terminate = True
            return

        # Update best individual data
        if best_fitness < self.best_fitness:
            self.best_fitness = best_fitness
            self.best_position = best_position

        if self.algo.stop():
            self.should_terminate = True

        # Update statistics
        self.iteration += 1
        self.stats['iteration'].append(self.iteration)
        self.stats['FEs'].append(self.FE)
        self.stats['error'].append(self.best_fitness - self.optimal_fitness)
        self.stats['best_fitness'].append(self.best_fitness)
        self.stats['best_position'].append(self.best_position.tolist())
        if self.verbose: self.print_status()
        #if self.plot: self.algo.plot()
   

    def print_status(self):
        error = self.best_fitness - self.optimal_fitness
        print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' %
              (self.iteration, self.FE, error, self.best_fitness) )
        #print('position:', self.best_position) 


    def stop(self):
        return self.should_terminate



def main(args):

    algo = Algo(
                n_points        = args.population, 
                dimension       = args.dimension, 
                function_id     = args.function_id-1, 
                algo_type       = args.algorithms, 
                max_evaluations = args.max_evaluations, 
                verbose         = args.verbose,
                plot            = args.plot_after_iteration,
                fig_dir         = args.figure_directory,
                )

    while not algo.stop():
        algo.run()

    algo.print_status()
    if args.csv_file: pd.DataFrame(algo.stats).to_csv(args.csv_file, index=False)
    print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run original Algorithms on CEC2005 problems')
    parser.add_argument('--algorithms', '-a', default='CMA', type=str, 
                        help='0:CMA, 1:PSO, 2:ACOR')
    parser.add_argument('--max_evaluations', '-fe', default=10000, type=int, 
                        help='FEs = 1e4*Dimension')
    parser.add_argument('--population', '-n', default=12, type=int, 
                        help='population') 
    parser.add_argument('--dimension', '-d', default=2, type=int, 
                        help='Dimensions = 2, 10, 30, 50 for all CEC2005 problems') 
    parser.add_argument('--function_id', '-i', default=9, type=int, 
                        help='function_id = 1~25 for all CEC2005 problems') 
    parser.add_argument('--verbose', '-v', default=False, type=bool, 
                        help='print process')
    parser.add_argument('--plot_after_iteration', '-plot', default=0, type=int, 
                        help='number of iterations to show clusters and contour')
    parser.add_argument('--csv_file', '-csv', default=None, type=str, 
                        help='Save statistics into csv file')
    parser.add_argument('--figure_directory', '-fig_dir', default=None, type=str, 
                        help='Save figures into directory')

    args = parser.parse_args()

    if args.figure_directory:
        if os.path.exists(args.figure_directory):
            print('%s already exists!')
        else:
            os.makedirs(args.figure_directory)


    if len(sys.argv) < 2:
        parser.print_help()
    else:
        main(args)
