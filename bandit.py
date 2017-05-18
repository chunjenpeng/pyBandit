import sys
from operator import attrgetter
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import rankdata
import numpy as np

from optproblems.cec2005 import CEC2005

from arm import Arm, draw_arms
from matrix import Matrix
from combination import Combination

class Bandit:

    def __init__( self, obj, n_points, dimension, **kwargs ):

        # Parameters for optproblems
        self.objective = obj
        self.n_points = n_points
        self.dimension = dimension
        self.min_bounds = kwargs.get('min_bounds', np.array([0.]*dimension) )
        self.max_bounds = kwargs.get('max_bounds', np.array([1.]*dimension) )

        self.init_min_bounds = kwargs.get('init_min_bounds', np.array([0.]*dimension) )
        self.init_max_bounds = kwargs.get('init_max_bounds', np.array([1.]*dimension) )
        self.algo_type = kwargs.get('algo_type', 'CMA')
        self.verbose = kwargs.get('verbose', False)
        self.fig_config = kwargs.get('fig_config', None)

        # Parameters for Bandit arms
        self.arms = []
        self.max_arms_num = min(self.n_points, kwargs.get('max_arms_num', 10) )
        self.max_evaluations = kwargs.get('max_evaluations', 1e4*dimension )
        self.f_left = self.max_evaluations

        # Initialize Bandit
        self.n_samples = 50 * dimension
        init_n_points = max(50*dimension, self.n_points*self.max_arms_num) 
        #init_n_points = self.n_points*self.max_arms_num
        population_step = 20 

        self.init_bandit(init_n_points, population_step)

        # Calculate f allocation
        self.remain_f_allocation = Combination(self.f_left, len(self.arms), 
                                               self.n_points, self.get_ranks()).combination



    def obj(self, X):
        self.f_left -= 1
        return self.objective(X) 



    def run(self):

        best_arm = self.find_best_arm() 

        if best_arm is not None:
            best_position, best_fitness = self.arms[best_arm].pull()
            self.remain_f_allocation -= self.n_points

            if best_fitness < self.best_fitness:
                self.best_fitness = best_fitness
                self.best_position = best_position

            # Check if need to recluster
            if self.arms[best_arm].reached_border():
                self.arms[best_arm].translate_to(best_position)
                self.recluster()

        return self.best_position, self.best_fitness



    def stop(self):
        for arm in self.arms:
            if not arm.algo.stop():
                return False
        print('All algorithms stop!')
        return True



    def find_best_arm(self):
        # Make sure at least one arm is still working
        if self.stop():
            return None

        c = Combination(self.f_left, len(arms), self.n_points, self.get_ranks())
        self.remain_f_allocation += c.combination
        best_arm = argmax(self.remain_f_allocation)

        # If best_arm stops, set allocation=-inf and choose again
        while self.arms[best_arm].stop():
            self.remain_f_allocation[best_arm] = -np.inf
            best_arm = argmax(self.remain_f_allocation)

        return best_arm



    def get_ranks(self, values=None):
        ranks = []
        if values is None:
            values = [ arm.get_fitnesses() for arm in self.arms ]
        all_ranks = rankdata( np.hstack(values), method = 'min' ) # TODO method='ordinal'
        start = 0

        for value in values:
            end = start + len(value)
            ranks.append(all_ranks[start:end].tolist())
            start = end

        return ranks





    ##############################
    #    Bandit Initialization   #
    ##############################

    def init_bandit(self, init_n_points, population_step):

        # Esitmate number of clusters with KMeans and silhouette_score
        cluster_positions, cluster_fitnesses = self.init_clusters( init_n_points, population_step )

        # Initialize arms and optimize projection matrix in each arm
        self.init_arms( cluster_positions, cluster_fitnesses )



    def init_clusters(self, init_n_points, population_step):
        # Initialize population with init_n_points
        positions, fitnesses = self.init_population( init_n_points )

        # Selection
        selected_positions, selected_fitnesses = self.selection( positions, fitnesses )

        # Esitmate number of clusters with silhouette_score
        k = self.estimate_k_clusters(selected_positions, self.max_arms_num)
        print('k:', k, 'pop:', len(selected_positions))


        # Increase population until k == last_k
        last_k = np.inf
        while k != last_k:
            last_k = k

            # Add "population_step" points into population
            new_positions, new_fitnesses = self.init_population( population_step )
            positions = np.concatenate((positions, new_positions), axis=0)
            fitnesses = np.concatenate((fitnesses, new_fitnesses), axis=0)
            
            selected_positions, selected_fitnesses = self.selection( positions, fitnesses )
            k = self.estimate_k_clusters(selected_positions, self.max_arms_num)
            print('k:', k, 'pop:', len(selected_positions))


        # Update best_fitness and best_position
        best_index = np.argmin(selected_fitnesses)
        self.best_fitness = selected_fitnesses[ best_index ]
        self.best_position = selected_positions[ best_index ]


        # KMeans clustering
        cluster_positions, cluster_fitnesses = \
            self.k_means(k, selected_positions, selected_fitnesses)  
        assert len(cluster_positions) == k
        assert len(cluster_positions) == len(cluster_fitnesses)

        return cluster_positions, cluster_fitnesses



    def init_arms(self, cluster_positions, cluster_fitnesses):

        k = len(cluster_positions)
        # Default matrix that translate and shrink search space to [0,1]^D 
        matrices = [ Matrix(positions) for positions in cluster_positions ]

        if DEBUG: 
            draw_arms( function_id, cluster_positions, matrices, 
                       fig_name='initialize.png', **self.fig_config )
        

        # Random sample n_samples to represent search space for non-overlapped optimization
        trans_samples = np.random.uniform( 0, 1, size=(self.n_samples, self.dimension) )

        for i in range(k):
            exclude = []
            for j in range(k):
                if i != j:
                    samples = matrices[j].inverse_transform( trans_samples )
                    exclude.extend(samples)

            exclude = np.array(exclude)
            arm = Arm( self.obj, 
                       self.n_points,
                       cluster_positions[i], 
                       cluster_fitnesses[i],
                       algo_type = self.algo_type,
                       exclude = exclude,
                       min_bounds = self.min_bounds,
                       max_bounds = self.max_bounds
                      )
            self.arms.append(arm) 
            matrices[i] = self.arms[i].matrix


            if DEBUG: 
                opt_points = []
                for j in range(k):
                    if i == j:
                        opt_points.append( cluster_positions[i] )
                    else:
                        samples = matrices[j].inverse_transform( trans_samples )
                        opt_points.append(samples)
                draw_arms( function_id, opt_points, matrices, 
                           fig_name='optimize_%d.png'%i,  **self.fig_config )


        if DEBUG: 
            opt_points = [ arm.get_positions() for arm in self.arms ]
            opt_matrices = [ arm.matrix for arm in self.arms ]
            draw_arms( function_id, opt_points, opt_matrices,
                       fig_name='optimized.png',  **self.fig_config )


    def init_population(self, n_points):
        positions = np.zeros( (n_points, self.dimension) )
        for d in range(self.dimension):
            positions[:,d] = np.random.uniform( self.init_min_bounds[d], 
                                                self.init_max_bounds[d], 
                                                n_points )
        fitnesses = np.array([self.obj(position) for position in positions])

        return positions, fitnesses



    def selection( self, positions, fitnesses, selection_pressure=2 ):

        index = fitnesses.argsort()
        selected = index[:int(len(positions)/selection_pressure)]
        selected_positions, selected_fitnesses = positions[selected], fitnesses[selected]

        return selected_positions, selected_fitnesses



    def k_means( self, k, positions, fitnesses ):

        labels = KMeans(n_clusters = k).fit_predict(positions) 

        cluster_positions = []
        cluster_fitnesses = []
        for i in range(k):
            indices = np.where(labels==i)[0]
            cluster_positions.append( positions[indices] )
            cluster_fitnesses.append( fitnesses[indices] )
        
        return np.array(cluster_positions), np.array(cluster_fitnesses)


    def estimate_k_clusters( self, X, max_n_clusters ):

        # silhouette_score is between [-1,1], 
        # so set score of 0 cluster as -1.0
        # so set score of 1 cluster as 0.0
        score = np.zeros(max_n_clusters+1)
        score[0] = -1.0

        for k in range(2, max_n_clusters+1):
            labels = KMeans(n_clusters = k).fit_predict(X)
            score[k] = silhouette_score(X, labels)

        return np.argmax(score) 





    #########################
    #    Bandit Recluster   #
    #########################

    def recluster( self, max_n_clusters ):

        positions, fitnesses = [], []
        for arm in self.arms:
            positions.extend( arm.get_positions() )
            fitnesses.extend( arm.get_fitnesses() )

        # Estimate number of clusters with cluster number < current cluster number 
        k = self.estimate_n_clusters( positions, len(self.arms) )
        # Recluster using KMeans
        cluster_positions, cluster_fitnessses = self.k_means(k, positions, fitnesses)


        # Compare ranks to find unchanged arms
        old_ranks = self.get_ranks()
        new_ranks = self.get_ranks(cluster_fitnesses)

        unchanged_arms = [i for i, ranks in enumerate(new_ranks) if ranks in old_ranks] 
        if self.verbose: print('unchanged_arms:%r' % unchanged_arms)

        '''
        new_arms = []
        for index, arm in enumerate(self.arms):

        

        new_algos = []
        for index, cluster in enumerate(new_clusters):
            if index in unchanged_clusters:

                for old_index, cluster in enumerate(self.clusters):
                    if all( x in new_ranks[index] for x in cluster.ranks):
                        new_clusters[index] = copy.deepcopy(self.clusters[old_index])
                        new_algos.append(copy.deepcopy(self.algos[old_index]))
                        break

            else:
                if self.algo_type == 'CMA':
                    new_algos.append( CMAEvolutionStrategy( [0.5]*self.dimension, 0.2, 
                                                            {'popsize': self.n_points, 
                                                             'bounds': [ 0, 1 ] } ) )
                    init_positions = new_algos[index].ask()

            
        self.clusters = new_clusters
        self.algos = new_algos
        #for i in range( len(self.clusters) ):
        #    if i not in unchanged_clusters:
        #        self.update_matrix(i)

        #self.update_borders()
        '''



class TestBandit:

    def __init__(self, n_points = 12, dimension = 2, function_id = 1, algo_type = 'CMA', 
                 max_evaluations = 10000, verbose = False, plot = 0, fig_dir = None):

        print('\nBandit + %s: Solving F%d in %dD with population size %d...\n' % 
              (algo_type, function_id, dimension, n_points))

        from optproblems.cec2005 import CEC2005
        from boundary import Boundary

        # Bandit parameters
        self.n_points = n_points
        self.dimension = dimension
        self.function_id = function_id - 1
        self.function = CEC2005(dimension)[self.function_id].objective_function
        max_bounds = Boundary(dimension, self.function_id).max_bounds
        min_bounds = Boundary(dimension, self.function_id).min_bounds
        init_max_bounds = Boundary(dimension, self.function_id).init_max_bounds
        init_min_bounds = Boundary(dimension, self.function_id).init_min_bounds

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
            'xlim':[ min_bounds[0], max_bounds[0] ],
            'ylim':[ min_bounds[1], max_bounds[1] ],
            'optimal': self.optimal_position
        }


        self.algo = Bandit( self.obj, self.n_points, self.dimension, 
                            algo_type = algo_type,
                            min_bounds = min_bounds,
                            max_bounds = max_bounds,
                            init_max_bounds = init_max_bounds,
                            init_min_bounds = init_min_bounds,
                            verbose = verbose,
                            max_evaluations =self.max_evaluations,
                            max_arms_num = 10,
                            fig_config = self.fig_config
                           )

    
        if self.plot > 0:
            error = self.best_fitness-self.optimal_fitness
            fig_name = ('F%d_%d.png' % (self.function_id+1, self.iteration) )
            self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                            (self.function_id+1, self.FE, error) )
            #draw_arms( self.function_id, self.function, fig_name, **self.fig_config )



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
                #draw_quiver( self.algo, self.function, fig_name, **self.fig_config )


        error = self.best_fitness - self.optimal_fitness
        print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % 
              (self.iteration, self.FE, error, self.best_fitness))
        print('position:%s\n' % self.best_position)

        if self.plot > 0:
            fig_name = ('F%d_%d.png' % (self.function_id+1, self.iteration) )
            self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                            (self.function_id+1, self.FE, error) )
            #draw_quiver( self.algo, self.function, fig_name, **self.fig_config )

        return None



if __name__ == '__main__':
    DEBUG = False 
    function_id = int(sys.argv[1])
    testBandit = TestBandit( n_points = 6,
                             dimension = 2,
                             function_id = function_id+1, # F1 ~ F25
                             max_evaluations = 1e4, 
                             algo_type = 'CMA', # 'CMA', 'PSO', 'ACOR'
                             #verbose = True,
                             plot = 0,
                             fig_dir = 'test_initialization/F%d'%(function_id+1)
                            )
    #testBandit.run()

