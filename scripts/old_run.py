#import matplotlib
#matplotlib.use('Agg')

import os, sys, argparse
import numpy as np
import pandas as pd
from operator import attrgetter
from collections import OrderedDict
from cma import CMAEvolutionStrategy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.transform import ProjectiveTransform
from scipy.spatial import Delaunay, ConvexHull
from scipy.stats import rankdata, chisquare

from optproblems.cec2005 import CEC2005
from optproblems import Individual, Problem
from optproblems.base import ResourcesExhausted

from cluster import Cluster
from boundary import Boundary
from combination import Combination
from plot import draw_original_contour, draw_contour
from bandit import Bandit

class ALGO:
    def __init__( self, max_evaluations, n_points, dimension, function_id, **kwargs ):

        self.function_id = function_id
        self.n_points = n_points
        self.dimension = dimension
        self.function = CEC2005(dimension)[function_id].objective_function
        self.max_evaluations = max_evaluations * dimension
        self.max_bounds = Boundary(dimension, function_id).max_bounds
        self.min_bounds = Boundary(dimension, function_id).min_bounds
        self.optimal_position = CEC2005(dimension)[function_id].get_optimal_solutions()[0].phenome
        self.optimal_fitness = self.function(self.optimal_position)


        #self.problem = Problem(self.function.objective_function, max_evaluations=max_evaluations)
        self.boundary = Boundary(dimension, function_id)
        self.verbose = kwargs.get('verbose', False)

        self.population = []#self.init_population( self.n_points, self.dimension )
        self.algo_type = kwargs.get('algo_type', 'CMA')
        self.init_algo()

        self.iteration = 0
        self.should_terminate = False
        self.optimal_solution = self.find_optimal_solution()

        self.stats = OrderedDict([
                      ('iteration',[]),
                      ('FEs',[]),
                      ('error',[]),
                      ('best_value',[]), 
                      #('best_position',[])
                     ]) 
        
        self.run()
        self.best_solution = min( self.population, key=attrgetter('objective_values') )



    def init_population( self, n_points, dim ):
        positions = np.zeros( (n_points, dim) )
        for d in range(dim):
            positions[:,d] = np.random.uniform( self.boundary.min_bounds[d], 
                                                self.boundary.max_bounds[d], 
                                                self.n_points )
    
        population = [Individual(position) for position in positions]
        self.problem.batch_evaluate(population)
        population = sorted( population, key=attrgetter('objective_values') )   
        population = population[:len(population)/2]
        ranks = range(1,len(population)+1)
        return [ Cluster(population, ranks) ]
    
    def init_algo( self ):
        init_min_bound = self.boundary.init_min_bounds[0]
        init_max_bound = self.boundary.init_max_bounds[0]
        min_bound = self.boundary.min_bounds[0]
        max_bound = self.boundary.max_bounds[0]

        if self.algo_type == 'CMA':
            init_point = [ (init_max_bound + init_min_bound)/2 ] * dimension
            sigma = (init_max_bound - init_min_bound) * 0.2
            #print 'init_point:', init_point
            #print 'sigma:', sigma 
            self.algo = CMAEvolutionStrategy( init_point, sigma,
                                             {'popsize': self.n_points,
                                               'bounds':[min_bound, max_bound]
                                             })
        #elif self.algo_type == 'PSO':





    def find_optimal_solution( self ):
        dimension = self.dimension
        function_id = self.function_id
        optimal_solutions = CEC2005(dimension)[function_id].get_optimal_solutions()
        test_prob = Problem( CEC2005(dimension)[function_id].objective_function )
        test_prob.batch_evaluate(optimal_solutions)
        return min( optimal_solutions, key=attrgetter('objective_values') )

    def run( self ):
        self.iteration = self.iteration + 1
        if self.algo_type == 'CMA':

            positions = self.algo.ask()

            solutions = [Individual(position) for position in positions]
            try:
                self.problem.batch_evaluate(solutions)
            except ResourcesExhausted:
                self.should_terminate = True
                return

            self.algo.tell( [p.phenome for p in solutions], [p.objective_values for p in solutions] )
            self.population = sorted( solutions, key=attrgetter('objective_values') )

        self.best_solution = min( self.population, key=attrgetter('objective_values') )
        self.update_statistics()




    def found_optimum( self, delta=1e-8 ):  
        if self.best_solution.objective_values - self.optimal_solution.objective_values < delta:
            return True
        return False


    def stop( self ):
        if self.algo.stop():
            if self.verbose: print('Algorithm stops!')
            self.should_terminate = True
        elif self.problem.remaining_evaluations < 1:
            if self.verbose: print('Consumed all evaluations!')
            self.should_terminate = True
        elif self.found_optimum( delta=goal_error ):
            if self.verbose: print('Found Optimum!')
            self.should_terminate = True
        return self.should_terminate 


    def print_status( self ):
        error = self.best_solution.objective_values - self.optimal_solution.objective_values
        print('')
        print('    Iteration %d: error = %e'%(self.iteration, error))
        print('     Evaluations: consumed %d, remain %d'%(self.problem.consumed_evaluations, 
                                               self.problem.remaining_evaluations))
        print('    best fitness: %f at %r'%(self.best_solution.objective_values, 
                                            self.best_solution.phenome.tolist()))
        print('optimal solution: %f at %r'%(self.optimal_solution.objective_values, 
                                            self.optimal_solution.phenome))
        print('')

    def update_statistics( self ):
        self.stats['iteration'].append(self.iteration)
        self.stats['FEs'].append(self.problem.consumed_evaluations)
        self.stats['error'].append(self.best_solution.objective_values - self.optimal_solution.objective_values)
        self.stats['best_value'].append(self.best_solution.objective_values)
        #self.stats['best_position'].append(self.best_solution.phenome.tolist())



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run original algorithms on CEC2005 problems')
    parser.add_argument('--max_evaluations', '-fe', default=10000, type=int, 
                        help='FEs = 1e4*Dimension')
    parser.add_argument('--population', '-n', default=6, type=int, 
                        help='population') 
    parser.add_argument('--dimension', '-d', default=2, type=int, 
                        help='Dimensions = 2, 10, 30, 50 for all CEC2005 problems') 
    parser.add_argument('--function_id', '-i', default=9, type=int, 
                        help='function_id = 1~25 for all CEC2005 problems') 
    parser.add_argument('--csv_file', '-csv', default=None, type=str, 
                        help='Save statistics into csv file')
    parser.add_argument('--algorithms', '-a', default=0, type=int, 
                        help='0:CMA-ES, 1:PSO, 2:ACOR')
    parser.add_argument('--draw_contour', '-draw', default=0, type=int, 
                        help='draw_contour')
    parser.add_argument('--verbose', '-v', default=False, type=bool, 
                        help='print process')


    args = parser.parse_args()

    n_points = args.population 
    dimension = args.dimension 
    max_evaluations = args.max_evaluations*args.dimension
    function_id = args.function_id - 1
    csv_file = args.csv_file 
    goal_error=1e-8


    if len(sys.argv) < 2:
        parser.print_help()
    else:
        algo = ALGO( max_evaluations, n_points, dimension, function_id, 
                     algo_type='CMA', verbose=args.verbose )

        if args.verbose: algo.print_status()
        if args.draw_contour>0: 
            it = 0
            draw_original_contour( function_id, population=algo.population)


        while not algo.stop():
            algo.run()

            if args.verbose: algo.print_status()
            if args.draw_contour > 0: 
                it += 1
                if it % args.draw_contour == 0:
                    draw_original_contour( function_id, population=algo.population )


        if csv_file is not None: pd.DataFrame(algo.stats).to_csv(csv_file, index=False)
        if args.draw_contour: draw_original_contour( function_id, population=algo.population)
        if not args.verbose: algo.print_status()
