from operator import itemgetter
import cma
import numpy as np

class CMA:
    
    def __init__(self, obj, n_points, dimension, **kwargs):
        self.obj = obj
        self.positions = np.zeros((n_points, dimension))
        self.fitnesses = np.zeros(n_points)
        min_bounds = kwargs.get('min_bounds', np.array([0.0] * dimension))
        max_bounds = kwargs.get('max_bounds', np.array([1.0] * dimension))

        assert len(min_bounds) == dimension
        assert len(min_bounds) == len(max_bounds)
        assert all(max_bounds > min_bounds)

        # Optimal solution should be within mean +/- 3*sigma
        sigma = max(max_bounds - min_bounds) / 6.0
        mean = (max_bounds + min_bounds) / 2.0
        self.es = cma.CMAEvolutionStrategy( mean.tolist(), sigma, 
                                            { 'popsize': n_points,
                                              'bounds': [ min_bounds.tolist(), 
                                                          max_bounds.tolist()] 
                                            })


        init_positions = kwargs.get('init_positions', None)
        init_fitnesses = kwargs.get('init_fitnesses', None)
        if init_positions is not None and init_fitnesses is not None:
            
            assert len(init_fitnesses) == n_points
            assert len(init_positions) == n_points
            assert len(init_positions[0]) == dimension

            self.positions = np.array(init_positions)
            self.fitnesses = np.array(init_fitnesses)
            _ = self.es.ask()
            self.es.tell(self.positions.tolist(), self.fitnesses.tolist())
        else:
            best_position, best_fitness = self.run()            
    

    def run(self):
        try:
            self.positions = self.es.ask()
        except AssertionError:
            print(self.positions)
            input()

        #self.positions = self.es.ask()
        self.fitnesses = [ self.obj(p) for p in self.positions ]
        self.es.tell(self.positions, self.fitnesses)
        index, best_fitness = min( enumerate(self.fitnesses), key = itemgetter(1) )
        if self.es.stop():
            print('CMA-ES converges!')
        return self.positions[index], best_fitness

    
    def stop(self):
        return self.es.stop()

    
    def get_positions(self):
        return self.positions
    
    def get_fitnesses(self):
        return self.fitnesses


