from copy import deepcopy
from operator import attrgetter
import numpy as np 

class Point:
    def __init__( self, fitness, position ):
        self.fitness = fitness
        self.position = position


class Particle:
    def __init__( self, fitness, position, velocity ):
        self.current = Point(fitness, position)
        self.velocity = velocity
        self.previous_best = deepcopy(self.current)
        self.previous_best_neighbor = deepcopy(self.current)


class PSO:
    def __init__(self, n_points, dimension, **kwargs):
        self.dimension = dimension 
        self.n_points = n_points
        self.min_bounds = kwargs.get('min_bounds', np.array([0.]*dimension) )
        self.max_bounds = kwargs.get('max_bounds', np.array([1.]*dimension) )
        self.config = kwargs.get('config', 'SPSO2011')

        #self.topology = self.ring_topology() 
        self.topology = self.random_topology() 
        self.w = 1.0/(2.0*np.log(2))
        self.c = 0.5 + np.log(2) 

        self.iteration = 0
        self.should_terminate = False
        self.best_fitness = float('Inf')

        self.swarm = []
        for i in range(n_points):
            
            x = [ np.random.uniform(self.min_bounds[d], self.max_bounds[d]) 
                  for d in range(dimension) ]

            if self.config == 'SPSO2006':
                v = [ (np.random.uniform(min_bounds[d], max_bounds[d]) - x[d])/2.0 
                      for d in range(dimension) ] 
            else: #SPSO 2011
                v = [ np.random.uniform(self.min_bounds[d] - x[d], self.max_bounds[d] - x[d]) 
                      for d in range(dimension) ] 

            self.swarm.append( Particle(float('Inf'), np.array(x), np.array(v)) )


    def ring_topology(self):
        ring = []
        l = range(self.n_points)
        for i, _ in enumerate(l):
            if i+1 < len(l):
                ring.append([ l[i-1], l[i], l[i+1] ] )
            else:
                ring.append([ l[i-1], l[i], l[0] ])
        return np.array(ring)

    def random_topology(self, K=3):
        A = np.eye(self.n_points)
        v = np.random.randint(self.n_points, size=(self.n_points,K) )
        for i in range(v.shape[0]):
            A[i,v[i]] = 1

        topology = [ [] for i in range(self.n_points) ] 
        for i, j in np.argwhere(A>0):
            topology[j].append(i)
        return np.array(topology)

    def stop(self):
        return self.should_terminate

    def tell(self, positions, fitnesses):
        for p, position, fitness in zip(self.swarm, positions, fitnesses):
            assert(position == p.current.position).all()

            p.current.fitness = fitness

            # Update previous_best
            if p.current.fitness < p.previous_best.fitness:
                p.previous_best = deepcopy(p.current)

        # Update previous_best_neighbor
        for i, p in enumerate(self.swarm):
            best = i
            for index in self.topology[i]: 
                neighbor = self.swarm[index] 
                if neighbor.previous_best.fitness <  \
                   self.swarm[best].previous_best.fitness:
                   best = index

            p.previous_best_neighbor = deepcopy(self.swarm[best].previous_best)

        #self.print_status()

        # Update global best_fitness
        current_best_fitness = min( self.swarm, key=attrgetter('current.fitness') ).current.fitness

        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
        else:
            self.topology = self.random_topology()
    
        self.update()
        

    def print_status(self):
        for i in range(self.n_points):
            print('Particle',i,' neighbors:', self.topology[i])
            particle = self.swarm[i]
            x = particle.current.position
            p = particle.previous_best.position
            l = particle.previous_best_neighbor.position
            print('x:', x, particle.current.fitness) 
            print('p:', p, particle.previous_best.fitness) 
            print('l:', l, particle.previous_best_neighbor.fitness) 
            print()



    def update(self):
        self.iteration += 1
        for i in np.random.permutation(self.n_points):
            p = self.swarm[i]
            # Update velocity
            random_position = self.sample_from_hypersphere(p)
            p.velocity = self.w*p.velocity + random_position - p.current.position


    def sample_from_hypersphere(self, particle):
        x = particle.current.position
        p = particle.previous_best.position
        l = particle.previous_best_neighbor.position

        if (l != p).any():
            centre = x + self.c/3.0*(p+l-2.0*x) 
        else:
            centre = x + self.c/2.0*(p-x) 

        r_max = np.linalg.norm(centre-x) 
        #r = np.random.uniform(0, r_max)
        r = np.random.uniform(0, r_max)**(1.0/self.dimension)
        v = np.random.uniform(0.0, 1.0, size=self.dimension )
        v = v * (r/np.linalg.norm(v))
        return centre+v


    def ask(self):
        # Update position 
        for p in self.swarm:
            p.current.position += p.velocity
        self.check_confinement()

        return [ p.current.position for p in self.swarm ]


    def check_confinement(self):
        if self.config == 'SPSO2006':
            zeros = np.zeros(self.dimension)

        for p in self.swarm:
            min_index = np.where(p.current.position < self.min_bounds) 
            max_index = np.where(p.current.position > self.max_bounds) 
            
            if min_index:
                p.current.position[min_index] = self.min_bounds[min_index]
                if self.config == 'SPSO2006':
                    p.velocity[min_index] = zeros[min_index]
                else: #SPSO 2011
                    p.velocity[min_index] = -0.5*p.velocity[min_index]

            if max_index:
                p.current.position[max_index] = self.max_bounds[max_index]
                if self.config == 'SPSO2006':
                    p.velocity[max_index] = zeros[max_index]
                else: #SPSO 2011
                    p.velocity[max_index] = -0.5*p.velocity[max_index]
            



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

    # Draw arrow
    X = np.array([ p.current.position[0] for p in pso.swarm ])
    Y = np.array([ p.current.position[1] for p in pso.swarm ])
    U = np.array([ p.velocity[0] for p in pso.swarm ])
    V = np.array([ p.velocity[1] for p in pso.swarm ])
    M = np.hypot(U, V)
    Q = ax.quiver(X, Y, U, V, color='r', units='x', pivot='tail', scale=1)
    
    # Draw scatter
    ax.scatter(X, Y, color='r', s=10)

    plt.savefig('%s/%s' % (fig_dir, fig_name) )
    plt.close(fig)




def main():

    from optproblems import Individual, Problem
    from optproblems.cec2005 import CEC2005
    from optproblems.base import ResourcesExhausted
    from boundary import Boundary

    DEBUG=True
    n_points = 12 
    dimension = 2
    function_id = 1 
    max_evaluations = 1e4

    func = CEC2005(dimension)[function_id]
    optimal_solution = func.get_optimal_solutions()[0]
    Problem( func.objective_function ).evaluate(optimal_solution)
    best_fitness = optimal_solution.objective_values 
    problem = Problem( func.objective_function,  max_evaluations=max_evaluations*dimension )
    boundary = Boundary(dimension, function_id)
    fig_config = {
        'fig_dir': 'test_pso',
        'angle': 240,
        'xlim':[ boundary.min_bounds[0], boundary.max_bounds[0] ],
        'ylim':[ boundary.min_bounds[1], boundary.max_bounds[1] ],
        'optimal': func.get_optimal_solutions()[0].phenome
    }


    pso = PSO( n_points, dimension, 
               min_bounds=boundary.min_bounds,
               max_bounds=boundary.max_bounds,
             )
    

    if DEBUG:
        FE = problem.consumed_evaluations
        fig_name = ('F%d_%d' % (function_id+1, pso.iteration) )
        fig_config['fig_title'] = 'F%d, FE=%d' % (function_id+1, FE) 
        draw_quiver( pso, func.objective_function, fig_name, **fig_config )

    while not pso.stop():
        positions = pso.ask()
        solutions = [Individual(position) for position in positions]
        try:
            problem.batch_evaluate(solutions)
        except ResourcesExhausted: break

        pso.tell([p.phenome for p in solutions], [p.objective_values for p in solutions])


        if DEBUG:
            FE = problem.consumed_evaluations
            error = pso.best_fitness-best_fitness
            fig_name = ('F%d_%d' % (function_id+1, pso.iteration) )
            fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % (function_id+1, FE, error) )
            draw_quiver( pso, func.objective_function, fig_name, **fig_config )
            '''
            for p in pso.swarm:
                print(p.fitness, 
                      p.position, 
                      p.velocity, 
                      #p.previous_best.fitness, 
                      #p.previous_best.position,
                      #p.previous_best_neighbor.fitness, 
                      #p.previous_best_neighbor.position,
                     )
            '''
        print('Iteration %d, FE:%d, best fitness: %f, error:%.2e' 
              % (pso.iteration, FE, pso.best_fitness, error) )


if __name__ == "__main__":
    main()
