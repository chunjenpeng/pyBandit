import sys, random
from operator import attrgetter
from collections import OrderedDict
from copy import deepcopy
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import rankdata
import numpy as np
import pandas as pd

from optproblems.cec2005 import CEC2005

from arm import Arm, draw_arms, draw_optimization
from matrix import Matrix
from combination import Combination
from cluster import hierarchical_clustering, trim_by_MDL

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
        self.plot = kwargs.get('plot', 0)
        self.fig_config = kwargs.get('fig_config', None)

        # Parameters for Bandit arms
        self.arms = []
        #self.max_arms_num = min(self.n_points, kwargs.get('max_arms_num', 10) )
        self.max_evaluations = kwargs.get('max_evaluations', 1e4*dimension )
        self.f_left = self.max_evaluations
        self.iteration = 0

        # Initialize Bandit
        self.n_samples = 200 * dimension
        #init_n_points = max(100*dimension, self.n_points*self.max_arms_num) 
        init_n_points = 100*dimension
        population_step = int(init_n_points/10)

        self.init_bandit(init_n_points, population_step)

        # Calculate f allocation
        self.remain_f_allocation = np.zeros(len(self.arms))
        self.remain_f_ratio = self.calculate_remain_f_ratio()



    def obj(self, X):
        self.f_left -= 1
        return self.objective(X) 



    def run(self):
        self.iteration += 1

        best_arm = self.find_best_arm() 

        if best_arm is None:
            return self.best_position, self.best_fitness
        
        #best_position, best_fitness = self.arms[best_arm].pull()
        ################################################
        try:
            best_position, best_fitness = self.arms[best_arm].pull()
        except AssertionError as e:
            draw_arms( function_id-1, self.arms,
                       fig_name='it%d_error.png'% (self.iteration-1), 
                       **self.fig_config )
            print(self.arms[best_arm].algo.es.D)
            raise e
        ################################################

        self.remain_f_allocation[best_arm] -= 1.0

        # Update statistics
        if best_fitness < self.best_fitness:
            self.best_fitness = best_fitness
            self.best_position = best_position
        
        # Check if need to update border
        if self.arms[best_arm].mean_shifted():
            if self.plot > 0: 
                draw_arms( function_id-1, self.arms,
                           fig_name='it%d.png'% (self.iteration-1), 
                           **self.fig_config )

            if self.should_delete_arm(best_arm):
                del self.arms[best_arm]
                self.remain_f_allocation = np.zeros(len(self.arms))
                self.remain_f_ratio = self.calculate_remain_f_ratio()
                #self.remain_f_allocation = np.delete(self.remain_f_allocation, best_arm)
                if self.verbose: print('Delete arm %d\n' % best_arm)
                if self.plot > 0: 
                    draw_arms( function_id-1, self.arms,
                               fig_name='it%d_0_delete.png'% (self.iteration-1), 
                               **self.fig_config )
                return self.best_position, self.best_fitness


            include = self.arms[best_arm].get_positions()
            include_fitnesses = self.arms[best_arm].get_fitnesses()
            trans_samples = np.random.uniform( 0, 1, size=(self.n_samples, self.dimension) )
            exclude = []
            for i, arm in enumerate(self.arms):
                if i != best_arm:
                    samples = arm.matrix.inverse_transform( trans_samples )
                    exclude.extend(samples)
            exclude = np.array(exclude)

            matrix = Matrix(include, include_fitnesses, self.min_bounds, self.max_bounds)
            matrix.optimize( best_position,
                             include,
                             exclude )

            self.arms[best_arm].transform(matrix)
            self.arms[best_arm].matrix = matrix
            self.arms[best_arm].update_model()

            if self.verbose: print('Update matrix of arm %d\n' % best_arm)
            if self.plot > 0: 
                draw_arms( function_id-1, self.arms,
                           fig_name='it%d_1_update.png'% (self.iteration-1), 
                           **self.fig_config )

        #############################################################################
            #if True:
            positions = self.arms[best_arm].get_positions()
            fitnesses = self.arms[best_arm].get_fitnesses()
    
            # Check if need to recluster 
            avg_search_space = np.mean([arm.search_space for arm in self.arms])
            splitting_threshold = self.arms[best_arm].search_space / avg_search_space
            if random.random() < splitting_threshold:
                if self.verbose: 
                    print('Pass splitting_threshold: %f = %f/%f' 
                          % (splitting_threshold, 
                             self.arms[best_arm].search_space, 
                             avg_search_space) )
    
                labels = hierarchical_clustering( positions, fitnesses )
                labels = trim_by_MDL( positions, fitnesses, labels )
            else:
                labels = [0]*len(fitnesses)
    
            max_label = max(labels)
            if max_label > 0:
                if self.verbose: 
                    print('\nReclustering... spliting arm %d into %d' 
                          % (best_arm, max_label+1) )
    
            # Check if merging is possible
            if max_label > 0 or len(self.arms) > 1:
                label = 0
                old_labels = [label] * len(fitnesses)
                for i, arm in enumerate(self.arms):
                    if i != best_arm:
                        positions = np.concatenate(( positions, arm.get_positions() ))
                        fitnesses = np.concatenate(( fitnesses, arm.get_fitnesses() ))
                        max_label += 1
                        label += 1
                        labels = np.concatenate((labels,[max_label]*len(arm.get_fitnesses())))
                        old_labels.extend([label]*len(arm.get_fitnesses()))
        
                    
                labels = trim_by_MDL( positions, fitnesses, labels )
                old_labels = np.array(old_labels)
                if not (labels == old_labels).all():
                    if self.verbose: 
                        print('Reclustering... estimate %d clusters with MDL' 
                               % (max(labels)+1) )
                        print('old_labels:', old_labels)
                        print('new_labels:', labels)
                    
                    positions, fitnesses = np.array(positions), np.array(fitnesses)
                    cluster_positions, cluster_fitnesses = [], []
                    for i in range( max(labels)+1 ):
                        indices = np.where(labels == i)[0]
                        cluster_positions.append(positions[indices])
                        cluster_fitnesses.append(fitnesses[indices])
                    
                    # Recluster
                    self.recluster(cluster_positions, cluster_fitnesses, labels)
        
                    self.remain_f_allocation = np.zeros( len(self.arms) )
                    self.remain_f_ratio = self.calculate_remain_f_ratio()
        
                    if self.plot > 0: 
                        draw_arms( function_id-1, self.arms,
                                   fig_name='it%d_2_recluster.png'% (self.iteration-1), 
                                   **self.fig_config )
        #############################################################################


        return self.best_position, self.best_fitness




    def stop(self):
        for arm in self.arms:
            if not arm.algo.stop():
                return False
        print('All algorithms stop!')
        return True

    
    def calculate_remain_f_ratio(self, threshold = 5000):
    
        self.check_point = max( threshold, int(self.f_left/2) )
        c = Combination(self.f_left, len(self.arms), self.n_points*len(self.arms), self.get_ranks())
        ratio = c.combination / sum(c.combination)

        if self.verbose:
            print('\nRecalculate ratio:', c.combination, '[%s]'%','.join('%6.3f'%i for i in ratio))
            
        return np.array(ratio)



    def find_best_arm(self):
        # Make sure at least one arm is still working
        if self.stop():
            return None

        if self.f_left < self.check_point:
            self.remain_f_ratio = self.calculate_remain_f_ratio()

        self.remain_f_allocation = self.remain_f_allocation + self.remain_f_ratio
        best_arm = np.argmax(self.remain_f_allocation)
        if self.verbose:
            print('Choose arm %d: [%s] ' % 
                  (best_arm, ','.join('%5.2f' % i for i in self.remain_f_allocation)), end='')

        # If best_arm stops, set allocation=-inf and choose again
        while self.arms[best_arm].stop():
            self.remain_f_allocation[best_arm] = -np.inf
            best_arm = np.argmax(self.remain_f_allocation)

        return best_arm



    def get_ranks(self, values=None):
        ranks = []
        if values is None:
            values = [ arm.get_fitnesses() for arm in self.arms ]
        #all_ranks = rankdata( np.hstack(values), method = 'min' ) 
        all_ranks = rankdata( np.hstack(values), method = 'ordinal' ) 
        start = 0

        for value in values:
            end = start + len(value)
            ranks.append(all_ranks[start:end].tolist())
            start = end

        return ranks



    def should_delete_arm(self, index):
        if len(self.arms) > 1:
            all_ranks = self.get_ranks()
            best_rank_of_worst_cluster = ((len(self.arms) - 1) * self.n_points) 
            if min(all_ranks[index]) >= best_rank_of_worst_cluster: 
                print('Delete worst arm %d with ranks:' % index)
                for ranks in all_ranks:
                    print(ranks)
                return True
        return False





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
        selected_positions, selected_fitnesses = self.selection( positions, fitnesses, selection_pressure=2 )

        # Update best_fitness and best_position
        best_index = np.argmin(selected_fitnesses)
        self.best_fitness = selected_fitnesses[ best_index ]
        self.best_position = selected_positions[ best_index ]

        labels = hierarchical_clustering(selected_positions, selected_fitnesses) 
        if self.verbose: print( 'Estimate %d clusters with hierarchical clustering' % (max(labels)+1) )

        labels = trim_by_MDL(selected_positions, selected_fitnesses, labels) 
        if self.verbose: print( 'Estimate %d clusters with MDL' % (max(labels)+1) )

        cluster_positions, cluster_fitnesses = [], []
        for i in range(max(labels) + 1):
            indices = np.where(labels==i)[0]
            cluster_positions.append(selected_positions[indices])
            cluster_fitnesses.append(selected_fitnesses[indices])

        return cluster_positions, cluster_fitnesses



    def init_arms(self, cluster_positions, cluster_fitnesses):

        # Default matrix that translate and shrink search space to [0,1]^D 
        matrices = [ Matrix(positions, fitnesses, self.min_bounds, self.max_bounds)
                     for positions, fitnesses in zip(cluster_positions, cluster_fitnesses) ]

        if self.plot > 0: 
            draw_optimization( function_id-1, cluster_positions, matrices, 
                               fig_name='initial.png', **self.fig_config )
        

        # Random sample n_samples to represent search space for non-overlapped optimization
        trans_samples = np.random.uniform( 0, 1, size=(self.n_samples, self.dimension) )

        k = len(cluster_positions)
        for i in range(k):
            exclude = []
            for j in range(k):
                if i != j:
                    samples = matrices[j].inverse_transform( trans_samples )
                    exclude.extend(samples)

            exclude = np.array(exclude)

            if self.verbose: print('optimizing matrix for arm %d' % i)

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


            if self.plot > 0: 
                opt_points = []
                for j in range(k):
                    if i == j:
                        opt_points.append( cluster_positions[i] )
                    else:
                        samples = matrices[j].inverse_transform( trans_samples )
                        opt_points.append(samples)
                draw_optimization( function_id-1, opt_points, matrices, 
                                   fig_name='initial_optimize_%d.png'%i,  **self.fig_config )


        if self.plot > 0: 
            opt_points = [ arm.get_positions() for arm in self.arms ]
            opt_matrices = [ arm.matrix for arm in self.arms ]
            draw_optimization( function_id-1, opt_points, opt_matrices,
                               fig_name='initial_optimized.png',  **self.fig_config )


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


    def estimate_k_clusters( self, X, max_num_clusters ):
        
        from gap import gap
        num_clusters = gap(X, max_num_clusters = max_num_clusters)
        
        # silhouette_score is between [-1,1], 
        # so set score of 0 cluster as -1.0
        # so set score of 1 cluster as 0.0
        score = np.zeros(max_num_clusters+1)
        score[0] = -1.0

        for k in range(2, max_num_clusters+1):
            labels = KMeans(n_clusters = k).fit_predict(X)
            score[k] = silhouette_score(X, labels)

        print('silhouette   :', np.argmax(score))
        return num_clusters





    #########################
    #    Bandit Recluster   #
    #########################
    def recluster( self, cluster_positions, cluster_fitnesses, labels):

        k = max(labels) + 1

        # Compare ranks to find unchanged arms
        old_arms_ranks = self.get_ranks()
        new_arms_ranks = self.get_ranks(cluster_fitnesses)
        if self.verbose:
            print('\nold ranks:')
            for ranks in old_arms_ranks:
                print(ranks)
            print('new ranks:')
            for ranks in new_arms_ranks:
                print(ranks)


        # Generate new arms
        new_arms = []
        matrices = [ Matrix(positions, fitnesses, self.min_bounds, self.max_bounds)
                     for positions, fitnesses in zip(cluster_positions, cluster_fitnesses) ]
        trans_samples = np.random.uniform( 0, 1, size=(self.n_samples, self.dimension) )

        for new_index, new_ranks in enumerate(new_arms_ranks):

            new_set = set(new_ranks) 
            arm = None

            # Get best positions for matrix optimization
            argmax = np.argmax( cluster_fitnesses[new_index] )
            best = cluster_positions[ new_index ][ argmax ]

            # Get exclude positions for matrix optimization
            exclude = []
            for j in range(k):
                if new_index != j :
                    samples = matrices[j].inverse_transform( trans_samples )
                    exclude.extend(samples)
            exclude = np.array(exclude)


            # Case 1: Copy old arm with more than 50% similarities
            for old_index, old_ranks in enumerate(old_arms_ranks):

                intersection = new_set.intersection(old_ranks)
                duplicate_ratio = len(intersection) / max( len(old_ranks), len(new_ranks) )

                #if len(intersection) > max( len(old_ranks)/2, len(new_ranks)/2 ):
                if duplicate_ratio > 0.5:

                    # Optimize matrix
                    if self.verbose: print('\nOptimizing matrix for new arm %d' % new_index )
                    matrices[new_index].optimize( best, 
                                                  cluster_positions[new_index], 
                                                  exclude )

                    # Copy unchanged arm to new arms
                    arm = self.arms[old_index]

                    # Transform Algorithm to new subspace
                    arm.transform( matrices[new_index] )

                    # Find update particles
                    insert_ranks = new_set.difference(intersection) 
                    remove = set(old_ranks).difference(new_set) 

                    resize_ranks = sorted( deepcopy( new_ranks ) )
                    resize_ranks = resize_ranks[:self.n_points]
                    


                    if self.verbose:
                        print('Copy arm %d -> %d: %.1f%% duplicate' % 
                               (old_index, new_index, 100*duplicate_ratio) )
                        #print('intersection:', list(intersection))
                        #print('insert_ranks:', list(insert_ranks))
                        #print('remove_ranks:', list(remove))
                        #print('resize_ranks:', resize_ranks)

                        #print('\noriginal:')
                        #print(self.arms[old_index].get_positions())
                        #print(self.arms[old_index].get_fitnesses())
                        #print(old_ranks)

                        #print('\nnew:')
                        #print(cluster_positions[new_index])
                        #print(cluster_fitnesses[new_index]) 
                        #print(new_ranks)


                    replace_indices, replace_positions, replace_fitnesses = [], [], []
                    for i, rank in enumerate(old_ranks):
                        if rank not in resize_ranks:

                            # insert set is not empty
                            if insert_ranks:
                                insert_rank = insert_ranks.pop()
                                index = new_ranks.index( insert_rank )
                                replace_position = cluster_positions[new_index][index]
                                subspace_position = arm.matrix.transform([replace_position])[0]
                                replace_fitness  = cluster_fitnesses[new_index][index] 
                                #if self.verbose:
                                #if self.verbose:
                                #    print('old_rank[%d] = %d is replaced by rank %d, f = %f, pos=' 
                                #          % (i, rank, insert_rank, replace_fitness), replace_position)

                            # Replace particle with a random sample 
                            else:
                                subspace_position = np.random.uniform( 0, 1, size=(self.dimension))
                                replace_position = arm.matrix.inverse_transform([subspace_position])[0]
                                replace_fitness  = self.obj(replace_position)
                                #if self.verbose:
                                #    print('old_rank[%d] = %d is replaced by sample point f = %f, pos=' 
                                #          % (i, rank, replace_fitness), replace_position)

                            replace_indices.append(i)
                            replace_positions.append(subspace_position)
                            replace_fitnesses.append(replace_fitness)

                
                    try:
                        arm.algo.replace( replace_indices, replace_positions, replace_fitnesses ) 
                    except AssertionError as e:
                        arm = None
                        
                        for i, p, f in zip(replace_indices, replace_positions, replace_fitnesses):
                            print(i, p, f)
                        input()
                        break
                    arm.update_model()
                    break


            # Case 2: Create new arm
            if arm is None:
                arm = Arm( self.obj, 
                           self.n_points,
                           cluster_positions[new_index],
                           cluster_fitnesses[new_index],
                           algo_type = self.algo_type,
                           exclude = exclude,
                           min_bounds = self.min_bounds,
                           max_bounds = self.max_bounds
                          )

            # Append new_arm and update matrix
            new_arms.append(arm)
            matrices[new_index] = new_arms[new_index].matrix

        self.arms = new_arms





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
                            plot = plot,
                            fig_config = self.fig_config
                           )

    
        if self.plot > 0:
            error = self.best_fitness-self.optimal_fitness
            self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                            (self.function_id+1, self.FE, error) )
            draw_arms( self.function_id, self.algo.arms,
                       fig_name='it%d.png'%self.iteration, **self.fig_config )



    def obj(self, x):

        self.FE += 1
        fitness = self.function(x)

        if self.FE >= self.max_evaluations:
            raise Exception('Resources Exhausted!')

        error = fitness - self.optimal_fitness
        if error < self.termination_error:
            self.best_fitness = fitness
            self.best_position = x
            print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % 
                  (self.iteration, self.FE, error, self.best_fitness))
            print('position:%s\n' % self.best_position)

            if self.plot > 0:
                self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                                (self.function_id+1, self.FE, error) )
                draw_arms( self.function_id, self.algo.arms,
                           fig_name='it%d.png'%self.iteration, **self.fig_config )
            raise Exception('Found Optima!')

        return fitness

    

    def run(self):
        while not self.algo.stop():
            self.iteration += 1
            
            (self.best_position, self.best_fitness) = self.algo.run()
            #try:
            #    (self.best_position, self.best_fitness) = self.algo.run()
            #except Exception as e:
            #    print(e)
            #    break

            error = self.best_fitness - self.optimal_fitness
            self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                                (self.function_id+1, self.FE, error) )

            if self.verbose:
                print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % 
                      (self.iteration, self.FE, error, self.best_fitness))
                #print('position:%s\n' % self.best_position)
                
            if self.plot > 0 and self.iteration % self.plot == 0:
                draw_arms( self.function_id, self.algo.arms,
                           fig_name='it%d.png'%self.iteration, **self.fig_config )


        error = self.best_fitness - self.optimal_fitness
        print('Iter:%d, FE:%d, error:%.2e, fitness:%.2f' % 
              (self.iteration, self.FE, error, self.best_fitness))
        print('position:%s\n' % self.best_position)

        if self.plot > 0:
            self.fig_config['fig_title'] = ('F%d, FE=%d, error=%.2e' % 
                                            (self.function_id+1, self.FE, error) )
            draw_arms( self.function_id, self.algo.arms,
                       fig_name='it%d.png'%self.iteration, **self.fig_config )
        return None



if __name__ == '__main__':

    function_id = 9 # F1 ~ F25
    fig_dir = 'test_bandit_2' 

    if len(sys.argv) == 3:
        function_id = int(sys.argv[1])
        fig_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        function_id = int(sys.argv[1])

    testBandit = TestBandit( n_points = 30,
                             dimension = 2,
                             function_id = function_id, # F1 ~ F25
                             max_evaluations = 1e4, 
                             algo_type = 'CMA', # 'CMA', 'PSO', 'ACOR'
                             verbose = True,
                             plot = 1000, 
                             fig_dir = '%s/F%d' % (fig_dir, function_id)
                            )
    testBandit.run() 
