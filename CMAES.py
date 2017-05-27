from operator import itemgetter
from matplotlib.patches import Ellipse
import cma
import numpy as np

class CMA:
    
    def __init__(self, obj, n_points, dimension, **kwargs):
        self.obj = obj
        self.positions = np.zeros((n_points, dimension))
        self.fitnesses = np.zeros(n_points)
        min_bounds = kwargs.get('min_bounds', np.array([0.0] * dimension))
        max_bounds = kwargs.get('max_bounds', np.array([1.0] * dimension))

        self.dimension = dimension
        self.n_points = n_points
        self.update_count = 0 

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
        except AssertionError as e:
            print(e)
            print(self.positions)
            input()

        #self.positions = self.es.ask()
        self.fitnesses = [ self.obj(p) for p in self.positions ]
        self.es.tell(self.positions, self.fitnesses)

        index, best_fitness = min( enumerate(self.fitnesses), key = itemgetter(1) )
        if self.es.stop():
            print('CMA-ES converges!')
        return self.positions[index], best_fitness



    def update_one_particle(self):
        position = self.es.ask(1)[0]
        self.positions[self.update_count] = position 
        self.fitnesses[self.update_count] = self.obj(position)

        self.update_count += 1 
        if self.update_count >= self.n_points:
            self.es.tell(self.positions, self.fitnesses)
            self.update_count = 0

        index, best_fitness = min( enumerate(self.fitnesses), key = itemgetter(1) )
        if self.es.stop():
            print('CMA-ES converges!')
        return self.positions[index], best_fitness
    


    def stop(self):
        return self.es.stop()

    
    def get_positions(self):
        return np.array(self.positions)
    
    def get_fitnesses(self):
        return self.fitnesses


    def draw(self, ax, color, matrix = None):

        # Draw scatter
        X = self.get_positions()
        if matrix is not None:
            X = matrix.inverse_transform( X )

        ax.scatter(X[:,0], X[:,1], color=color, s=10)

        # Draw covariance ellipse
        pos = self.es.gp.pheno( self.es.mean, into_bounds=self.es.boundary_handler.repair)
        cov = self.es.C * (self.es.sigma**2)

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:,order]

        if matrix is not None:
            pos = matrix.inverse_transform( [pos] )[0]
            #cov = matrix.inverse_transform( self.es.C ) * \
            #      (matrix.inverse_transform( [self.es.sigma] )[0] ** 2 )
            vals = abs(matrix.inverse_transform( [vals] )[0])
            vecs = matrix.inverse_transform( vecs )

        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        for nstd in range(1,4):
            width, height = 2 * nstd * np.sqrt(abs(vals))
            ellip = Ellipse( xy = pos,
                             width = width, 
                             height = height,
                             angle = theta )
            ellip.set_alpha(0.2)
            ellip.set_facecolor(color)
            ax.add_artist(ellip)



def draw_CMAES( cmaes, obj, fig_name, **kwargs ):

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

    # Draw scatter & covariance ellipse
    cmaes.draw(ax, 'red') 

    plt.savefig('%s/%s' % (fig_dir, fig_name) )
    plt.close(fig)




class TestCMAES:

    def __init__(self, n_points = 6, dimension = 2, function_id = 1, 
                 max_evaluations = 10000, verbose = False, plot = 0, fig_dir = None):

        print('\nCMA-ES: Solving F%d in %dD with population size %d...\n' % 
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


        self.algo = CMA(self.obj, self.n_points, self.dimension, 
                        min_bounds = self.min_bounds, 
                        max_bounds = self.max_bounds)
    
        if self.plot > 0:
            error = self.best_fitness-self.optimal_fitness
            fig_name = ('F%d_%d.png' % (self.function_id+1, self.iteration) )
            self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                            (self.function_id+1, self.FE, error) )
            draw_CMAES( self.algo, self.function, fig_name, **self.fig_config )



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
                print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % 
                      (self.iteration, self.FE, error, self.best_fitness))
                #print('position:%s\n' % self.best_position)
                
            if self.plot > 0 and self.iteration % self.plot == 0:
                error = self.best_fitness-self.optimal_fitness
                fig_name = ('F%d_%d.png' % (self.function_id+1, self.iteration) )
                self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                                (self.function_id+1, self.FE, error) )
                draw_CMAES( self.algo, self.function, fig_name, **self.fig_config )


        error = self.best_fitness - self.optimal_fitness
        print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % 
              (self.iteration, self.FE, error, self.best_fitness))
        print('position:%s\n' % self.best_position)
        if self.plot > 0:
            fig_name = ('F%d_%d.png' % (self.function_id+1, self.iteration) )
            self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                            (self.function_id+1, self.FE, error) )
            draw_CMAES( self.algo, self.function, fig_name, **self.fig_config )

        return None


if __name__ == '__main__':

    function_id = 1 # F1 ~ F25
    fig_dir = 'test_cmaes'

    import sys
    if len(sys.argv) == 3:
        function_id = int(sys.argv[1])
        fig_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        function_id = int(sys.argv[1])

    testCMAES= TestCMAES(n_points = 10, 
                         dimension = 2,
                         function_id = function_id, # F1 ~ F25
                         max_evaluations = 10000,
                         verbose=True,
                         plot = 1, # number of iterations to draw a figure
                         fig_dir = '%s/F%d' % (fig_dir, function_id)
                        )
    testCMAES.run()
    
