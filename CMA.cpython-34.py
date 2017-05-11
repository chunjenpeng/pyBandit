# Source Generated with Decompyle++
# File: CMA.cpython-34.pyc (Python 3.4)

from operator import itemgetter
import cma
import numpy as np

class CMA:
    __qualname__ = 'CMA'
    
    def __init__(self, obj, n_points, dimension, **kwargs):
        self.obj = obj
        self.positions = np.zeros((n_points, dimension))
        self.fitnesses = np.zeros(n_points)
        min_bounds = kwargs.get('min_bounds', np.array([
            0] * dimension))
        max_bounds = kwargs.get('max_bounds', np.array([
            1] * dimension))
        init_positions = kwargs.get('init_positions', None)
        init_fitnesses = kwargs.get('init_fitnesses', None)
        if not len(min_bounds) == dimension:
            raise AssertionError
        if not None(min_bounds) == len(max_bounds):
            raise AssertionError
        if not None.all(max_bounds > min_bounds):
            raise AssertionError
        sigma = None(max_bounds - min_bounds) / 6
        mean = (max_bounds + min_bounds) / 2
        self.es = cma.CMAEvolutionStrategy(mean.tolist(), sigma, {
            'popsize': n_points,
            'bounds': [
                min_bounds.tolist(),
                max_bounds.tolist()] })
        if init_positions is not None and init_fitnesses is not None:
            if not len(init_fitnesses) == n_points:
                raise AssertionError
            if not None(init_positions) == n_points:
                raise AssertionError
            if not None(init_positions[0]) == dimension:
                raise AssertionError
            self.positions = None
            self.fitnesses = init_fitnesses
            X = self.es.ask()
            self.es.tell(self.positions.tolist(), self.fitnesses.tolist())

    
    def run(self):
        self.positions = self.es.ask()
        self.fitnesses = (lambda .0: continue[ self.obj(p) for p in .0 ])(self.positions)
        self.es.tell(self.positions, self.fitnesses)
        (index, best_fitness) = max(enumerate(self.fitnesses), key = itemgetter(1))
        return (self.positions[index], best_fitness)

    
    def stop(self):
        if self.es.stop():
            print('CMA-ES converges!')
        return self.es.stop()

    
    def get_positions(self):
        return self.positions

    
    def get_fitnesses(self):
        return self.fitnesses


