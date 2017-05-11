# Source Generated with Decompyle++
# File: SPSO2011.cpython-34.pyc (Python 3.4)

from copy import deepcopy
from operator import attrgetter
import numpy as np

class Point:
    __qualname__ = 'Point'
    
    def __init__(self, fitness, position):
        self.fitness = fitness
        self.position = position



class Particle:
    __qualname__ = 'Particle'
    
    def __init__(self, fitness, position, velocity):
        self.current = Point(fitness, position)
        self.velocity = velocity
        self.previous_best = deepcopy(self.current)
        self.previous_best_neighbor = deepcopy(self.current)



class PSO:
    __qualname__ = 'PSO'
    
    def __init__(self, obj, n_points, dimension, **kwargs):
        self.obj = obj
        self.dimension = dimension
        self.n_points = n_points
        self.min_bounds = kwargs.get('min_bounds', np.array([
            0] * dimension))
        self.max_bounds = kwargs.get('max_bounds', np.array([
            1] * dimension))
        self.max_velocity = np.abs(self.max_bounds - self.min_bounds)
        self.min_velocity = -(self.max_velocity)
        self.config = kwargs.get('config', 'SPSO2011')
        self.topology = self.random_topology()
        self.w = 1 / 2 * np.log(2)
        self.c = 0.5 + np.log(2)
        self.iteration = 0
        self.should_terminate = False
        self.best_fitness = float('Inf')
        self.swarm = []
        for i in range(n_points):
            x = (np.array,)((lambda .0: continue[ np.random.uniform(self.min_bounds[d], self.max_bounds[d]) for d in .0 ])(range(dimension)))
            if self.config == 'SPSO2006':
                v = (np.array,)((lambda .0: continue[ (np.random.uniform(min_bounds[d], max_bounds[d]) - x[d]) / 2 for d in .0 ])(range(dimension)))
            else:
                v = (None, np.array)((lambda .0: continue[ np.random.uniform(self.min_bounds[d] - x[d], self.max_bounds[d] - x[d]) for d in .0 ])(range(dimension)))
            self.swarm.append(Particle(self.obj(x), x, v))
        

    
    def ring_topology(self):
        ring = []
        l = range(self.n_points)
        for (i, _) in enumerate(l):
            if i + 1 < len(l):
                ring.append([
                    l[i - 1],
                    l[i],
                    l[i + 1]])
                continue
            ring.append([
                l[i - 1],
                l[i],
                l[0]])
        
        return np.array(ring)

    
    def random_topology(self, K = 3):
        A = np.eye(self.n_points)
        v = np.random.randint(self.n_points, size = (self.n_points, K))
        for i in range(v.shape[0]):
            A[(i, v[i])] = 1
        
        topology = (lambda .0: continue[ [] for i in .0 ])(range(self.n_points))
        for (i, j) in np.argwhere(A > 0):
            topology[j].append(i)
        
        return np.array(topology)

    
    def stop(self):
        return self.should_terminate

    
    def run(self):
        self.iteration += 1
        for i in np.random.permutation(self.n_points):
            p = self.swarm[i]
            best_neighbor = (min,)((lambda .0: continue[ self.swarm[neighbor] for neighbor in .0 ])(self.topology[i]), key = attrgetter('previous_best.fitness'))
            p.previous_best_neighbor = deepcopy(best_neighbor.previous_best)
            random_position = self.sample_from_hypersphere(p)
            p.velocity = self.w * p.velocity + random_position - p.current.position
            p.current.position += p.velocity
            self.check_confinement(i)
            p.current.fitness = self.obj(p.current.position)
            if p.current.fitness < p.previous_best.fitness:
                p.previous_best = deepcopy(p.current)
                if p.previous_best.fitness < p.previous_best_neighbor.fitness:
                    p.previous_best_neighbor = deepcopy(p.previous_best)
                
        best = min(self.swarm, key = attrgetter('current.fitness'))
        if best.current.fitness >= self.best_fitness:
            self.topology = self.random_topology()
        self.best_fitness = best.current.fitness
        return (best.current.position, best.current.fitness)

    
    def sample_from_hypersphere(self, particle):
        x = particle.current.position
        p = particle.previous_best.position
        l = particle.previous_best_neighbor.position
        if (l != p).any():
            centre = x + (self.c / 3) * (p + l - 2 * x)
        else:
            centre = x + (self.c / 2) * (p - x)
        r_max = np.linalg.norm(centre - x)
        r = np.random.uniform(0, r_max)
        v = np.random.uniform(0, 1, size = self.dimension)
        v = v * (r / np.linalg.norm(v))
        return centre + v

    
    def check_confinement(self, index):
        p = self.swarm[index]
        if self.config == 'SPSO2006':
            zeros = np.zeros(self.dimension)
        min_index = np.where(p.current.position < self.min_bounds)
        max_index = np.where(p.current.position > self.max_bounds)
        if min_index:
            p.current.position[min_index] = self.min_bounds[min_index]
            if self.config == 'SPSO2006':
                p.velocity[min_index] = zeros[min_index]
            else:
                p.velocity[min_index] = -0.5 * p.velocity[min_index]
        if max_index:
            p.current.position[max_index] = self.max_bounds[max_index]
            if self.config == 'SPSO2006':
                p.velocity[max_index] = zeros[max_index]
            else:
                p.velocity[max_index] = -0.5 * p.velocity[max_index]

    
    def print_status(self):
        for i in range(self.n_points):
            print('Particle', i, ' neighbors:', self.topology[i])
            particle = self.swarm[i]
            x = particle.current.position
            p = particle.previous_best.position
            l = particle.previous_best_neighbor.position
            print('x:', x, particle.current.fitness)
            print('p:', p, particle.previous_best.fitness)
            print('l:', l, particle.previous_best_neighbor.fitness)
            print()
        

    
    def get_positions(self):
        return (lambda .0: continue[ p.current.position for p in .0 ])(self.swarm)



def draw_quiver(pso, function, fig_name, **kwargs):
    import os
    import matplotlib.pyplot as plt
    cm = cm
    import matplotlib
    Individual = Individual
    Problem = Problem
    import optproblems
    problem = Problem(function)
    angle = kwargs.get('angle', 240)
    optimal = kwargs.get('optimal', None)
    xlim = kwargs.get('xlim', [
        -100,
        100])
    ylim = kwargs.get('ylim', [
        -100,
        100])
    fig_title = kwargs.get('fig_title', fig_name)
    fig_dir = kwargs.get('fig_dir', 'test_pso')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig = plt.figure(figsize = plt.figaspect(0.85))
    ax = fig.add_subplot(111, aspect = 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cmap = cm.coolwarm
    fig.tight_layout()
    st = fig.suptitle(fig_title, fontsize = 16)
    st.set_y(0.95)
    fig.subplots_adjust(top = 0.85)
    step = (xlim[1] - xlim[0]) / 100
    X = np.arange(xlim[0], xlim[1] + step, step)
    Y = np.arange(ylim[0], ylim[1] + step, step)
    (X, Y) = np.meshgrid(X, Y)
    positions = (lambda .0: continue[ [
x,
y] for (x, y) in .0 ])(zip(X.ravel(), Y.ravel()))
    solutions = (lambda .0: continue[ Individual(position) for position in .0 ])(positions)
    problem.batch_evaluate(solutions)
    Z = np.array((lambda .0: continue[ solution.objective_values for solution in .0 ])(solutions))
    vmin = min(Z)
    vmax = max(Z)
    vmin = vmin - (vmax - vmin) * 0.2
    vmax = vmax + (vmax - vmin) * 0.2
    Z = Z.reshape(X.shape)
    cset = ax.contourf(X, Y, Z, cmap = cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(cset, aspect = 20)
    if optimal:
        ax.scatter(optimal[0], optimal[1], color = 'w', marker = 'x', s = 100)
    X = np.array((lambda .0: continue[ p.current.position[0] for p in .0 ])(pso.swarm))
    Y = np.array((lambda .0: continue[ p.current.position[1] for p in .0 ])(pso.swarm))
    U = np.array((lambda .0: continue[ p.velocity[0] for p in .0 ])(pso.swarm))
    V = np.array((lambda .0: continue[ p.velocity[1] for p in .0 ])(pso.swarm))
    M = np.hypot(U, V)
    Q = ax.quiver(X, Y, U, V, color = 'r', units = 'x', pivot = 'tail', scale = 1)
    ax.scatter(X, Y, color = 'r', s = 10)
    plt.savefig('%s/%s' % (fig_dir, fig_name))
    plt.close(fig)


class TestSPSO2011:
    __qualname__ = 'TestSPSO2011'
    
    def __init__(self, n_points = 12, dimension = 2, function_id = 1, max_evaluations = 10000, verbose = False, plot = 0, fig_dir = None):
        print('\nSPSO2011: Solving F%d in %dD with population size %d...\n' % (function_id, dimension, n_points))
        CEC2005 = CEC2005
        import optproblems.cec2005
        Boundary = Boundary
        import boundary
        self.n_points = n_points
        self.dimension = dimension
        self.function_id = function_id - 1
        self.function = CEC2005(dimension)[self.function_id].objective_function
        self.max_bounds = Boundary(dimension, self.function_id).max_bounds
        self.min_bounds = Boundary(dimension, self.function_id).min_bounds
        self.FE = 0
        self.iteration = 0
        self.max_evaluations = max_evaluations * dimension
        self.termination_error = 1e-08
        self.should_terminate = False
        self.optimal_position = CEC2005(dimension)[self.function_id].get_optimal_solutions()[0].phenome
        self.optimal_fitness = self.function(self.optimal_position)
        self.best_position = np.zeros_like(self.optimal_position)
        self.best_fitness = np.inf
        self.verbose = verbose
        self.plot = plot
        self.algo = PSO(self.obj, self.n_points, self.dimension, min_bounds = self.min_bounds, max_bounds = self.max_bounds)

    
    def obj(self, x):
        self.FE += 1
        fitness = self.function(x)
        if self.FE >= self.max_evaluations:
            raise Exception('Resources Exhausted!')
        error = fitness - self.optimal_fitness
        if error < self.termination_error:
            self.best_fitness = fitness
            self.best_position = x
            raise Exception('Found Optima!')
        return fitness

    
    def run(self):
        while not self.algo.stop():
            self.iteration += 1
            
            try:
                (self.best_position, self.best_fitness) = self.algo.run()
            except Exception:
                e = None
                
                try:
                    print(e)
                finally:
                    e = None
                    del e


            if self.verbose:
                error = self.best_fitness - self.optimal_fitness
                print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % (self.iteration, self.FE, error, self.best_fitness))
                continue
            error = self.best_fitness - self.optimal_fitness
            print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % (self.iteration, self.FE, error, self.best_fitness))
            print('position:%s\n' % self.best_position)
            return None


if __name__ == '__main__':
    testSPSO2011 = TestSPSO2011(n_points = 40, dimension = 30, function_id = 11, max_evaluations = 10000, plot = 0, fig_dir = None)
    testSPSO2011.run()
