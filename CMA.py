from operator import itemgetter

import cma 
import numpy as np

class CMA:
    def __init__( self, obj, n_points, min_bounds, max_bounds ):
        assert len(min_bounds) == len(max_bounds)
        assert np.all(max_bounds>min_bounds)
        sigma = max( max_bounds - min_bounds )/6.0
        self.es = cma.CMAEvolutionStrategy( [0]*len(min_bounds), sigma,
                                            {'popsize': n_points,
                                             'bounds':[list(min_bounds), list(max_bounds)]
                                            })
        self.obj = obj

    def run(self):
        positions = self.es.ask()
        fitnesses = [ self.obj(p) for p in positions ]
        self.es.tell( positions, fitnesses )
        index, best_fitness = max( enumerate(fitnesses), key=itemgetter(1) )
        return positions[index], best_fitness

    def stop(self):
        if self.es.stop(): 
            print('CMA-ES converges!')
        return self.es.stop()
