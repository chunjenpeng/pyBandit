# Source Generated with Decompyle++
# File: SPSO2011.cpython-34.pyc (Python 3.4)

from copy import deepcopy
from operator import attrgetter
import numpy as np

class Point:
    def __init__(self, fitness, position):
        self.fitness = fitness
        self.position = position



class Particle:
    def __init__(self, fitness, position, velocity):
        self.current = Point(fitness, position)
        self.velocity = velocity
        self.previous_best = deepcopy(self.current)
        self.previous_best_neighbor = deepcopy(self.current)



class PSO:
    def __init__(self, obj, n_points, dimension, **kwargs):

        self.obj = obj
        self.dimension = dimension
        self.n_points = n_points
        self.min_bounds = kwargs.get('min_bounds', np.array([0] * dimension))
        self.max_bounds = kwargs.get('max_bounds', np.array([1] * dimension))
        # Speed limit for SPSO2006
        #self.max_velocity = np.abs(self.max_bounds - self.min_bounds)
        #self.min_velocity = -(self.max_velocity)

        # Parameters for update
        self.config = kwargs.get('config', 'SPSO2011')
        self.topology = self.random_topology(K=3)
        self.w = 1.0 / (2.0 * np.log(2.0))
        self.c = 0.5 + np.log(2.0)


        # Parameters for termiantion
        self.iteration = 0
        self.update_count = 0
        self.update_order = np.random.permutation(self.n_points)
        self.should_terminate = False
        self.best_fitness = float('Inf')


        # Initialize swarm
        init_positions = kwargs.get('init_positions', None)
        init_fitnesses = kwargs.get('init_fitnesses', None)
        if init_positions is not None and init_fitnesses is not None:
            assert len(init_fitnesses) == n_points
            assert len(init_positions) == n_points
            assert len(init_positions[0]) == dimension
            self.swarm = []
            for f, x in zip(init_fitnesses, init_positions):
                v = [ np.random.uniform(self.min_bounds[d] - x[d], self.max_bounds[d] - x[d]) 
                      for d in range(self.dimension) ] 
                self.swarm.append( Particle( f, np.array(x), np.array(v) ) )
        else:
            self.swarm = self.init_swarm()

        # Ensure exploitation state if particles are within hypersphere
        self.exploitation = np.zeros(self.n_points)


    def init_swarm(self):

        swarm = []

        for i in range(self.n_points):
            x = [ np.random.uniform(self.min_bounds[d], self.max_bounds[d]) 
                  for d in range(self.dimension) ]

            if self.config == 'SPSO2006':
                v = [ (np.random.uniform(self.min_bounds[d], self.max_bounds[d]) - x[d])/2.0 
                      for d in range(self.dimension) ] 
            else: #SPSO 2011
                v = [ np.random.uniform(self.min_bounds[d] - x[d], self.max_bounds[d] - x[d]) 
                      for d in range(self.dimension) ] 

            swarm.append( Particle(float('Inf'), np.array(x), np.array(v)) )

        return swarm

    
    def ring_topology(self):
        ring = []
        l = range(self.n_points)
        for (i, _) in enumerate(l):
            if i + 1 < len(l):
                ring.append([ l[i - 1], l[i], l[i + 1]])
            else:
                ring.append([ l[i - 1], l[i], l[0]])

        return np.array(ring)

    
    def random_topology(self, K = 3):
        A = np.eye(self.n_points)
        v = np.random.randint(self.n_points, size = (self.n_points, K))
        for i in range(v.shape[0]):
            A[i, v[i]] = 1
        
        topology = [ [] for i in range(self.n_points) ]
        for (i, j) in np.argwhere(A > 0):
            topology[j].append(i)
        
        return np.array(topology)

    
    def stop(self):
        return self.should_terminate

    
    def run(self):
        self.iteration += 1

        for i in np.random.permutation(self.n_points):
            p = self.swarm[i]

            # Update best of previous_best in neighborhood
            best_neighbor = min( [self.swarm[neighbor] for neighbor in self.topology[i] ], 
                                 key = attrgetter('previous_best.fitness') )
            p.previous_best_neighbor = deepcopy(best_neighbor.previous_best)

            # Update velocity
            random_position, centre = self.sample_from_hypersphere(p)
            p.velocity = self.w * p.velocity + random_position - p.current.position

            # Update position
            p.current.position += p.velocity
            self.check_confinement(i)
            
            # Update previous_best
            p.current.fitness = self.obj(p.current.position)
            if p.current.fitness < p.previous_best.fitness:
                p.previous_best = deepcopy(p.current)

                # Update best of previous_best in neighborhood
                if p.previous_best.fitness < p.previous_best_neighbor.fitness:
                    p.previous_best_neighbor = deepcopy(p.previous_best)
                

        # Update topology if the best know solution has not been improved
        best = min(self.swarm, key = attrgetter('current.fitness'))
        if best.current.fitness >= self.best_fitness:
            self.topology = self.random_topology()
        self.best_fitness = best.current.fitness

        return (best.current.position, best.current.fitness)


    def update_one_particle(self):
        i = self.update_order[ self.update_count ]  
        p = self.swarm[i]

        # Update best of previous_best in neighborhood
        best_neighbor = min( [self.swarm[neighbor] for neighbor in self.topology[i] ], 
                             key = attrgetter('previous_best.fitness') )
        p.previous_best_neighbor = deepcopy(best_neighbor.previous_best)

        # Update velocity
        random_position, centre = self.sample_from_hypersphere(p)
        p.velocity = self.w * p.velocity + random_position - p.current.position

        # Check if new position is within hyperspace (exploitation state)
        new_position = p.current.position + p.velocity
        if np.linalg.norm( new_position - centre ) < np.linalg.norm( p.current.position - centre ):
            self.exploitation[i] = 1
        else:
            self.exploitation[i] = 0
        

        # Update position
        p.current.position += p.velocity
        self.check_confinement(i)
            
        # Update previous_best
        p.current.fitness = self.obj(p.current.position)
        if p.current.fitness < p.previous_best.fitness:
            p.previous_best = deepcopy(p.current)

            # Update best of previous_best in neighborhood
            if p.previous_best.fitness < p.previous_best_neighbor.fitness:
                p.previous_best_neighbor = deepcopy(p.previous_best)
                

        best = min(self.swarm, key = attrgetter('current.fitness'))

        self.update_count += 1
        if self.update_count >= self.n_points:
            self.iteration += 1
            self.update_count = 0
            self.update_order = np.random.permutation(self.n_points)

            # Update topology if the best know solution has not been improved
            if best.current.fitness >= self.best_fitness:
                self.topology = self.random_topology()
            self.best_fitness = best.current.fitness

        return best.current.position, best.current.fitness
        


    def replace(self, indices, positions, fitnesses):

        for i, x, fitness in zip(indices, positions, fitnesses):
            v = [ np.random.uniform(self.min_bounds[d] - x[d], self.max_bounds[d] - x[d]) 
                  for d in range(self.dimension) ] 
            self.swarm[i] = Particle( fitness, np.array(x), np.array(v) )

        # Update topology if the best know solution has not been improved
        best = min(self.swarm, key = attrgetter('current.fitness'))
        if best.current.fitness >= self.best_fitness:
            self.topology = self.random_topology()
        self.best_fitness = best.current.fitness

        for p in self.swarm:
            # Update previous_best
            if p.current.fitness < p.previous_best.fitness:
                p.previous_best = deepcopy(p.current)

                # Update best of previous_best in neighborhood
                if p.previous_best.fitness < p.previous_best_neighbor.fitness:
                    p.previous_best_neighbor = deepcopy(p.previous_best)


    
    def sample_from_hypersphere(self, particle):

        x = particle.current.position
        p = particle.previous_best.position
        l = particle.previous_best_neighbor.position

        if (l != p).any():
            centre = x + (self.c / 3.0) * (p + l - 2.0 * x)
        else:
            centre = x + (self.c / 2.0) * (p - x)

        r_max = np.linalg.norm(centre - x)
        r = np.random.uniform(0.0, r_max)
        v = np.random.uniform(0.0, 1.0, size = self.dimension)
        v = v * (r / np.linalg.norm(v))

        return centre + v, centre

    
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



    def transform(self, inv_matrix, new_matrix):

        def trans(X):
            original_X = inv_matrix.inverse_transform( [X] )[0]
            return new_matrix.transform([original_X])[0]


        for i, particle in enumerate(self.swarm):

            start_point = particle.current.position - particle.velocity
            start_point = trans(start_point)

            particle.current.position = trans(particle.current.position)
            particle.previous_best.positoin = trans(particle.previous_best.position)
            particle.previous_best_neighbor.positoin = trans(particle.previous_best_neighbor.position)

            particle.velocity = particle.current.position - start_point
            self.check_confinement(i) 
            

    
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
        return [ p.current.position for p in self.swarm ]
    def get_fitnesses(self):
        return [ p.current.fitness for p in self.swarm ]


    def get_previous_best_positions(self):
        return [ p.previous_best.position for p in self.swarm ]
    def get_previous_best_fitnesses(self):
        return [ p.previous_best.fitness for p in self.swarm ]


    def get_previous_best_neighbor_positions(self):
        return [ p.previous_best_neighbor.position for p in self.swarm ]
    def get_previous_best_neighbor_fitnesses(self):
        return [ p.previous_best_neighbor.fitness for p in self.swarm ]



    def draw(self, ax, color, matrix = None, draw_algo=True):
        # Draw arrow
        positions = np.array([ p.current.position for p in self.swarm ])
        velocities = np.array([ p.velocity for p in self.swarm ])


        if matrix is not None:
            start_points = positions - velocities

            positions = matrix.inverse_transform( positions )

            start_points = matrix.inverse_transform( start_points )
            velocities = positions - start_points

        X = positions[:,0]
        Y = positions[:,1]

        U = velocities[:,0]
        V = velocities[:,1]
        #M = np.hypot(U, V)
        # Draw scatter
        ax.scatter(X, Y, color=color, s=10)

        # Draw velocity arrow
        if draw_algo:
            Q = ax.quiver(X, Y, U, V, color=color, alpha=0.4, units='x', pivot='tip', scale=1)
    



def draw_quiver( pso, obj, fig_name, **kwargs ):

    import os
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from optproblems import Individual, Problem

    # Parameters
    problem = Problem( obj )
    angle = kwargs.get( 'angle', 240 )
    optimal = kwargs.get( 'optimal', None )
    xlim = kwargs.get( 'xlim', [-100,100] )
    ylim = kwargs.get( 'ylim', [-100,100] )
    fig_title = kwargs.get( 'fig_title', fig_name )
    fig_dir = kwargs.get( 'fig_dir', 'test_pso' )
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig = plt.figure(figsize=plt.figaspect(0.85))
    ax = fig.add_subplot(111, aspect=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cmap = cm.coolwarm
    fig.tight_layout()
    st = fig.suptitle( fig_title, fontsize=16 )
    st.set_y(0.95)
    fig.subplots_adjust(top=0.85)

    # Generate mesh solutions
    step = (xlim[1] - xlim[0])/100.0
    X = np.arange( xlim[0], xlim[1]+step, step )
    Y = np.arange( ylim[0], ylim[1]+step, step )

    X, Y = np.meshgrid(X, Y)
    positions = [ [x,y] for x, y in zip(X.ravel(), Y.ravel()) ]
    solutions = [ Individual(position) for position in positions ]

    # Evaluate solutions
    problem.batch_evaluate(solutions)
    Z = np.array( [solution.objective_values for solution in solutions] )
    vmin, vmax = min(Z), max(Z)
    vmin = vmin - (vmax-vmin)*0.2
    vmax = vmax + (vmax-vmin)*0.2
    Z = Z.reshape(X.shape)


    # Draw contour
    cset = ax.contourf(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cset, aspect=20)
    if optimal:
        ax.scatter( optimal[0], optimal[1], color='w', marker='x', s=100 )

    pso.draw( ax, 'r' )
    '''
    # Draw arrow
    X = np.array([ p.current.position[0] for p in pso.swarm ])
    Y = np.array([ p.current.position[1] for p in pso.swarm ])
    U = np.array([ p.velocity[0] for p in pso.swarm ])
    V = np.array([ p.velocity[1] for p in pso.swarm ])
    M = np.hypot(U, V)
    Q = ax.quiver(X, Y, U, V, color='m', units='x', pivot='tip', scale=1)
    
    # Draw scatter
    ax.scatter(X, Y, color='r', s=10)
    '''
    plt.savefig('%s/%s' % (fig_dir, fig_name) )
    plt.close(fig)


class TestSPSO2011:

    def __init__(self, n_points = 12, dimension = 2, function_id = 1, 
                 max_evaluations = 10000, verbose = False, plot = 0, fig_dir = None):

        print('\nSPSO2011: Solving F%d in %dD with population size %d...\n' % 
              (function_id, dimension, n_points))

        from optproblems.cec2005 import CEC2005
        from boundary import Boundary

        self.n_points = n_points
        self.dimension = dimension
        self.function_id = function_id - 1
        self.function = CEC2005(dimension)[self.function_id].objective_function
        self.max_bounds = Boundary(dimension, self.function_id).max_bounds
        self.min_bounds = Boundary(dimension, self.function_id).min_bounds

        # Termination parameters
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
        self.fig_config = {
            'fig_dir': fig_dir,
            'angle': 240,
            'xlim':[ self.min_bounds[0], self.max_bounds[0] ],
            'ylim':[ self.min_bounds[1], self.max_bounds[1] ],
            'optimal': self.optimal_position
        }


        self.algo = PSO(self.obj, self.n_points, self.dimension, 
                        min_bounds = self.min_bounds, 
                        max_bounds = self.max_bounds)
    
        if self.plot > 0:
            error = self.best_fitness-self.optimal_fitness
            fig_name = ('F%d_%d.png' % (self.function_id+1, self.iteration) )
            self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                            (self.function_id+1, self.FE, error) )
            draw_quiver( self.algo, self.function, fig_name, **self.fig_config )



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
                #(self.best_position, self.best_fitness) = self.algo.run()
                (self.best_position, self.best_fitness) = self.algo.update_one_particle()
            except Exception as e:
                print(e)
                break


            if self.verbose:
                error = self.best_fitness - self.optimal_fitness
                #print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % 
                #      (self.iteration, self.FE, error, self.best_fitness))
                print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f, exploitation: %d/%d' % 
                      (self.iteration, self.FE, error, self.best_fitness, 
                       sum(self.algo.exploitation), self.n_points) )
                #print('position:%s\n' % self.best_position)
                
            if self.plot > 0 and self.iteration % self.plot == 0:
                error = self.best_fitness-self.optimal_fitness
                fig_name = ('F%d_%d.png' % (self.function_id+1, self.iteration) )
                self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                                (self.function_id+1, self.FE, error) )
                draw_quiver( self.algo, self.function, fig_name, **self.fig_config )


        error = self.best_fitness - self.optimal_fitness
        print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % 
              (self.iteration, self.FE, error, self.best_fitness))
        print('position:%s\n' % self.best_position)
        if self.plot > 0:
            fig_name = ('F%d_%d.png' % (self.function_id+1, self.iteration) )
            self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                            (self.function_id+1, self.FE, error) )
            draw_quiver( self.algo, self.function, fig_name, **self.fig_config )

        return None


if __name__ == '__main__':

    function_id = 1 # F1 ~ F25
    fig_dir = 'test_pso'

    import sys
    if len(sys.argv) == 3:
        function_id = int(sys.argv[1])
        fig_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        function_id = int(sys.argv[1])

    testSPSO2011 = TestSPSO2011(n_points = 12, 
                                dimension = 2,
                                function_id = function_id, # F1 ~ F25
                                max_evaluations = 10000,
                                verbose=True,
                                plot = 0, # number of iterations to draw a figure
                                fig_dir = '%s/F%d' % (fig_dir, function_id)
                                )
    testSPSO2011.run()
