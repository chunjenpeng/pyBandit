import random, math
from copy import deepcopy
from operator import attrgetter
from scipy.stats import norm, rv_discrete
import numpy as np


class ACOR:
    def __init__(self, obj, dimension, **kwargs):

        # Parameters for objective function
        self.obj = obj
        self.dimension = dimension
        self.min_bounds = kwargs.get('min_bounds', np.array([0] * dimension))
        self.max_bounds = kwargs.get('max_bounds', np.array([1] * dimension))


        # ACOR Parameters
        # number of ants
        self.n_ants = kwargs.get('ants_num', 2)

        # size of solution archive may not be smaller than the number of dimensions 
        k = kwargs.get('archive_size', 50)
        self.k = max( self.dimension, k ) 
        assert self.k > 1 # need to divide (self.k-1) later

        # parameter q (adjusting balance between iteration-best and best-so-far)
        self.q = kwargs.get('q', 1e-4)

        # standard deviation
        self.qk = self.q * self.k
        # parameter xi (similar to pheromone evaporation rate in ACO)
        self.xi = kwargs.get('xi', 0.85)


        # Parameters for termiantion
        self.iteration = 0
        self.converge_error = 1e-6
        self.should_terminate = False


        # Initialize archive 
        init_positions = kwargs.get('init_positions', None)
        init_fitnesses = kwargs.get('init_fitnesses', None)
        if init_positions is not None and init_fitnesses is not None:
            assert len(init_fitnesses) == self.k 
            assert len(init_fitnesses) == len(init_positions)
            assert len(init_positions[0]) == self.dimension
            init_positions = np.array(init_positions)
            init_fitnesses = np.array(init_fitnesses)
        else:
            init_positions = np.random.uniform( self.min_bounds, self.max_bounds, 
                                                size=(self.k, self.dimension) )
            init_fitnesses = np.array([self.obj(x) for x in init_positions])
        
        init_fitnesses = init_fitnesses.reshape(self.k, 1)
        self.archive = np.hstack(( init_positions, init_fitnesses ))
        # Sort archive by fitness 
        self.archive = self.archive[ np.argsort(self.archive[:,-1]) ]


        # Parameters for update
        # Initialize weight array with pdf function
        self.w = np.array( [ np.exp( -np.power(i,2)/(2.0*np.power(self.qk,2)) ) / \
                             (self.qk * np.sqrt(2.0*np.pi)) for i in range(self.k) ] )
        # probability of choosing the i-th Gaussian function
        p = self.w/sum(self.w)
        # calculate G_i and standard deviation sigma
        self.custom = rv_discrete( values=( np.arange(self.k), p[:] ) )


    def run(self):
        self.iteration += 1

        # Generate new samples
        new_archive = np.zeros((self.n_ants, self.dimension+1))
        for k in range(self.n_ants):

            # Select a Gaussian function
            selected = self.custom.rvs()
            sigma = self.xi / (self.k-1) * \
                    np.sum( np.abs(self.archive[:,:-1] - self.archive[selected, :-1]), axis=0 )
            #        np.sum( np.abs(self.archive - self.archive[selected]), axis=0 )
            assert (sigma >= 0).all()
            if max(sigma) < self.converge_error:
                self.should_terminate = True

            new_position = np.zeros(self.dimension)
            for i in range(self.dimension):
                #new_position = sigma[i] * np.random.random_sample() + self.archive[selected][i]
                new_position[i] = np.random.normal( self.archive[selected][i], sigma[i] )

                # Check if new_position is within boundary
                new_position[i] = min(new_position[i], self.max_bounds[i])
                new_position[i] = max(new_position[i], self.min_bounds[i])

            new_fitness = self.obj(new_position)
            new_archive[k,:-1] = new_position
            new_archive[k,-1] = new_fitness


        # Add new solutions into archive
        self.archive = np.vstack(( new_archive, self.archive ))
        # Sort archive by fitness 
        self.archive = self.archive[ np.argsort(self.archive[:,-1]) ]
        # Keep only top k solutions
        self.archive = self.archive[:self.k][:]


        best_position, best_fitness = self.archive[0,:-1], self.archive[0,-1]
        return (best_position, best_fitness)

    
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
        

    def stop(self):
        if self.should_terminate:
            print('ACOR converges: std in all dimensions < %.2e' % self.converge_error )
        return self.should_terminate

    def get_positions(self):
        return self.archive[:, :-1]

    def get_fitnesses(self):
        return self.archive[:, -1]
    
    



def draw_quiver( aco, obj, fig_name, **kwargs ):

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

    # Draw arrow
    X = np.array([ p.current.position[0] for p in pso.swarm ])
    Y = np.array([ p.current.position[1] for p in pso.swarm ])
    U = np.array([ p.velocity[0] for p in pso.swarm ])
    V = np.array([ p.velocity[1] for p in pso.swarm ])
    M = np.hypot(U, V)
    Q = ax.quiver(X, Y, U, V, color='m', units='x', pivot='tip', scale=1)
    
    # Draw scatter
    ax.scatter(X, Y, color='r', s=10)

    plt.savefig('%s/%s' % (fig_dir, fig_name) )
    plt.close(fig)


class TestACOR:

    def __init__(self, n_points = 50, dimension = 2, function_id = 1, 
                 max_evaluations = 10000, verbose = False, plot = 0, fig_dir = None):

        print('\nACOR: Solving F%d in %dD with population size %d...\n' % 
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


        self.algo = ACOR(self.obj, self.dimension, 
                         min_bounds = self.min_bounds, 
                         max_bounds = self.max_bounds,
                         ants_num = 2,
                         archive_size = self.n_points,
                         q = 1e-4,
                         xi = 0.85
                        )

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
                (self.best_position, self.best_fitness) = self.algo.run()
            except Exception as e:
                print(e)
                break


            if self.verbose:
                error = self.best_fitness - self.optimal_fitness
                print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % 
                      (self.iteration, self.FE, error, self.best_fitness))
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
    testACOR = TestACOR(n_points = 50, # archive size
                        dimension = 30,
                        function_id = 9, # F1 ~ F25
                        max_evaluations = 10000,
                        #verbose=True,
                        plot = 0, # number of iterations to draw a figure
                        fig_dir = 'test_acor'
                        )
    testACOR.run()
