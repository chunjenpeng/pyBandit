import matplotlib
matplotlib.use('Agg')

import os, sys, copy, argparse
import numpy as np
import pandas as pd
from operator import attrgetter
from collections import OrderedDict
from cma import CMAEvolutionStrategy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.transform import ProjectiveTransform
from scipy.spatial import Delaunay, ConvexHull
from scipy.stats import rankdata, chisquare, entropy
from scipy.optimize import minimize, fmin_tnc

from optproblems.cec2005 import CEC2005
from optproblems import Individual, Problem
from optproblems.base import ResourcesExhausted

from arm import Arm 
from combination import Combination
from plot import draw_surface3d, draw_cluster, draw_all, draw_contour

class Bandit:
    def __init__( self, obj, n_points, dimension, **kwargs ):

        # Parameters for optproblems
        self.obj = obj
        self.n_points = n_points
        self.dimension = dimension
        self.min_bounds = kwargs.get('min_bounds', np.array([0.]*dimension) )
        self.max_bounds = kwargs.get('max_bounds', np.array([1.]*dimension) )
        self.init_min_bounds = kwargs.get('init_min_bounds', np.array([0.]*dimension) )
        self.init_max_bounds = kwargs.get('init_max_bounds', np.array([1.]*dimension) )
        self.algo_type = kwargs.get('algo_type', 'CMA')
        self.verbose = kwargs.get('verbose', False)

        # Parameters for Bandit arms
        self.arms = []
        self.remain_f_allocation = np.zeros(len(self.clusters))
        self.max_arms_num = min(self.n_points, kwargs.get('max_arms_num', 10) )
        self.n_rand_points = 50*dimension #TODO

        # Initialize Bandit
        self.init_bandit()



    def run(self):

        best_arm = self.find_best_arm() 
        best_position, best_fitness = self.arms[best_arm].pull()

        if best_fitness < self.best_fitness:
            self.best_fitness = best_fitness
            self.best_position = best_position

        return self.best_position, self.best_fitness


    def stop(self):
        for arm in self.arms:
            if not arm.algo.stop():
                return False
        print('All algorithms stop!')
        return True


    def find_best_arm(self):
        #TODO
        best_arm = argmax(self.remain_f_allocation)
        while self.arms[best_arm].stop():
            best_arm = argmax(self.remain_f_allocation)
        return best_arm




    ##########################
    #    Initialize Bandit   #
    ##########################

    def init_bandit(self, init_n_points):

        # Initialize population
        positions, fitnesses = self.init_selected_population( init_n_points )

        # Esitmate number of clusters with silhouette_score
        min_k = self.estimate_k_clusters(positions, self.max_n_clusters)

        # Increase population until k <= min_k
        population_step = int(init_n_points/5)
    


        # KMeans clustering
        clusters_positions, clusters_fitnesses = self.k_means(positions, fitnesses)  
        assert len(clusters_positions) == k
        assert len(clusters_positions) == len(clusters_fitnesses)

        # Initialize Arms
        for positions, fitnesses in zip(clusters_positions, clusters_fitnesses):
            self.arms.append( Arm(self.obj, positions, fitnesses) )

        # Update projection matrix
        for i in range(k):
            positions_in = clusters_positions[i] 
            positions_out = [] 
            for j in range(k):
                if j != i: 
                    positions_out.extend( clusters_positions[j] )

            self.arms[i].update_matrix( positions_in, clusters_fitnesses[i], positions_out )

        # Update population == n_points in each arm
        self.resize_each_arm()        



    def init_selected_popualtion(self, n_points):
        positions = np.zeros( (n_points, self.dimension) )
        for d in range(self.dimension):
            positions[:,d] = np.random.uniform( self.init_min_bounds[d], 
                                                self.init_max_bounds[d], 
                                                n_points )
        fitnesses = np.array([self.obj(position) for position in positions])

        # Selection
        index = fitnesses.argsort()
        selected = index[:n_points//2]
        positions, fitnesses = positions[selected], fitnesses[selected]
        return positions, fitnesses


    def estimate_k_clusters( self, X, max_n_clusters ):
        score = np.zeros(max_n_clusters+1)
        score[0] = -1.0
        for k in range(2, max_n_clusters+1):
            km = KMeans(n_clusters=k)
            labels = km.fit_predict(X)
            score[k] = silhouette_score(X, labels)
        return np.argmax(score) 




        population = [Individual(position) for position in positions]
        self.problem.batch_evaluate(population)
        population = sorted( population, key=attrgetter('objective_values') )   
        selected_population = population[:len(population)//2]
        X = np.array([p.phenome for p in selected_population])


        if self.verbose: print('popsize:%d, k:%d'%(len(selected_population), k)) 
        ranks = range(1,len(selected_population)+1)
        self.clusters = [ Cluster(selected_population, ranks) ]


        min_k = k

        while True:
            positions = np.zeros( (population_step, self.dimension) )
            for d in range(self.dimension):
                positions[:,d] = np.random.uniform( self.boundary.min_bounds[d], 
                                                    self.boundary.max_bounds[d], 
                                                    population_step )
            new_population = [Individual(position) for position in positions]
            self.problem.batch_evaluate(new_population)
            population.extend(new_population)
            population = sorted( population, key=attrgetter('objective_values') )   
            selected_population = population[:len(population)//2]
            X = np.array([p.phenome for p in selected_population]) 

            k = self.estimate_n_clusters( X, self.max_n_clusters )
            if self.verbose: print('popsize:%d, k:%d'%(len(selected_population), k)) 
            if k <= min_k: break
            min_k = k 
    
            ### Should delete
            #self.clusters = self.k_means(k)
            #draw_contour( self.function_id, clusters=self.clusters )
            ###########

        self.clusters = self.k_means(k)
        #draw_contour( self.function_id, clusters=self.clusters )

        #self.update_borders()
        #for i in range(k):
        #    self.update_matrix(i)

        if args.draw_contour > 0 and args.figure_directory:
            draw_contour( self.function_id, clusters=self.clusters,
                          fig_name="%sF%d_init"%(args.figure_directory,self.function_id+1), 
                          fig_title="F%d_init"%(self.function_id+1) )
        #draw_contour( self.function_id, clusters=self.clusters )

        self.resize_each_cluster()


    def resize_each_arm(self):
        for arm in self.arms:
            

            delta_population = len(cluster.population) - self.n_points
            if delta_population < 0:
                trans_positions = np.random.uniform( 0, 1, (abs(delta_population), self.dimension) ) 
                positions = cluster.transform_inverse(trans_positions)
                new_population = [Individual(position) for position in positions]
                self.problem.batch_evaluate(new_population)
                cluster.population.extend(new_population)

            cluster.population = sorted( cluster.population, key=attrgetter('objective_values') )   
            cluster.population = cluster.population[:self.n_points]

        ranks = self.get_ranks(self.clusters)
        for i, cluster in enumerate(self.clusters):
            cluster.ranks = ranks[i]
            #print [p.objective_values for p in self.clusters[i].population]
            if self.verbose: print('cluster%d: %r'%(i, self.clusters[i].ranks) )


        '''
        values = []
        for cluster in self.clusters:
            values.append([p.objective_values for p in cluster.population])

        values = np.array(values)
        order = values.ravel().argsort()
        ranks = order.argsort().reshape( values.shape )

        for i in range( len(self.clusters) ):
            self.clusters[i].ranks = ranks[i]
            print [p.objective_values for p in self.clusters[i].population]
            print self.clusters[i].ranks 
        '''




    def recluster( self, max_n_clusters ):

        positions = []
        for cluster in self.clusters:
            positions.extend([ p.phenome for p in cluster.population ])
        X = np.array(positions)
        k = self.estimate_n_clusters( X, len(self.clusters) )

        should_recluster = True
        while should_recluster:

            self.remain_f_allocation = np.zeros(k)
            if self.verbose: print('recluster to %d clusters'%(k) )

            old_clusters = copy.deepcopy(self.clusters)
            old_algos = copy.deepcopy(self.algos)
            old_ranks = [cluster.ranks for cluster in self.clusters]
            if self.verbose: print('old_ranks:')
            for i in range( len(self.clusters) ):
                if self.verbose: print('cluster%d:%r'%(i,old_ranks[i]))

            new_clusters = self.k_means(k)
            new_ranks = self.get_ranks(new_clusters)#[cluster.ranks for cluster in new_clusters]
            if self.verbose: print('new_ranks:')
            for i in range(k):
                if self.verbose: print('cluster%d:%r'%(i,new_ranks[i]))


            should_recluster = False
            for cluster in new_clusters:
                if len(cluster.population) < 2:
                    k = k-1
                    should_recluster = True
                    break




        unchanged_clusters = [i for i, ranks in enumerate(new_ranks) if ranks in old_ranks] 
        if self.verbose: print('unchanged_clusters:%r' % unchanged_clusters)


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

        self.resize_each_cluster()


    def init_algos( self ):
        del self.algos[:]
        if self.algo_type == 'CMA':
            for cluster in self.clusters:
                self.algos.append( CMAEvolutionStrategy( [0.5]*self.dimension, 0.2, 
                                                    {'popsize': self.n_points, 
                                                     'bounds': [ 0, 1 ] } ) )
            for algo in self.algos:
                new_trans_positions = algo.ask()
    

    def k_means( self, n_clusters ):
        population = []
        for cluster in self.clusters:
            population.extend( cluster.population )
            
        X = np.array([ p.phenome for p in population ])
        labels = KMeans( n_clusters=n_clusters ).fit_predict(X)
        clusters = [ [] for i in range(n_clusters) ]
        ranks = [ [] for i in range(n_clusters) ]

        for i in range( len(population) ):
            clusters[ labels[i] ].append( population[i] )
            ranks[ labels[i] ].append(i+1)
        return [ Cluster(cluster, rank) for cluster, rank in zip(clusters, ranks) ]

    def update_matrix( self, i ):
        cluster = self.clusters[i]
        self.current_index = i
        solution = cluster.matrix.ravel()[0:-1]

        #res = minimize( self.evaluate_matrix, solution, method='BFGS', 
        #                options={'maxiter':10000, 'disp':True} ) 
        #x_best = res.x

        #res = minimize(self.evaluate_matrix, solution, method='nelder-mead', options={'disp': True})
        #x_best = res.x
        
        #res = fmin_tnc(self.evaluate_matrix, solution, approx_grad=True, disp=0, maxfun=100)
        res = fmin_tnc(self.evaluate_matrix, solution, approx_grad=True, maxfun=1000)
        x_best = res[0]

        cluster.matrix = np.append(x_best,1).reshape( cluster.matrix.shape )

        '''
        #std = np.std(solution)/10.0
        std = 0.0005
        #print std
        #self.evaluate_matrix(solution, i, debug=True)
        es = CMAEvolutionStrategy( solution.tolist(), std, {'maxiter':1000} )
        #es = CMAEvolutionStrategy( cluster.matrix[:,:-1].ravel().tolist(), 0.2 )
        best_score = self.evaluate_matrix(solution)
        while not es.stop():
            solutions = es.ask()
            es.tell( solutions, [self.evaluate_matrix(solution) for solution in solutions] ) 
            x_best = es.result()[0]
            current_best_score = self.evaluate_matrix(x_best)
            if x_best is not None and current_best_score > best_score :
                cluster.matrix = np.append(x_best,1).reshape( cluster.matrix.shape )
                best_score = current_best_score
                #cluster.border = cluster.transform_inverse( np.array([[0,0],[1,0],[1,1],[0,1]]) )
                #cluster.matrix = solution.reshape( cluster.matrix.shape )
                #x_best = np.append(x_best, np.zeros(self.dimension+1))
                #x_best[-2] = -1
                #cluster.matrix = x_best.reshape( cluster.matrix.shape )

                #self.evaluate_matrix(x_best, i, debug=True)
        #draw_contour( self.function_id, clusters=self.clusters )
        '''


    #def evaluate_matrix( self, solution, cluster_index, debug=False ):
    def evaluate_matrix( self, solution ):
        debug=False
        cluster_index = self.current_index
        cluster = self.clusters[ cluster_index ]
        original_matrix = cluster.matrix 
        cluster.matrix = np.append(solution,1).reshape( cluster.matrix.shape )
        #cluster.matrix = solution.reshape( cluster.matrix.shape )
        #solution = np.append(solution, np.zeros(self.dimension+1))
        #solution[-2] = -1
        #print solution
        #cluster.matrix = solution.reshape( cluster.matrix.shape )

        # Transform data points 
        original_best_point = min( cluster.population, key=attrgetter('objective_values') ).phenome 
        original_points_in = [ p.phenome for p in cluster.population ]
        original_points_out = []
        for i in range(len(self.clusters)):
            if i == cluster_index: continue
            #num_rand_points = 100
            trans_rand_points = np.random.rand(self.n_rand_points,self.dimension)
            rand_points = self.clusters[i].transform_inverse( trans_rand_points ).tolist()
            rand_points.extend( [p.phenome for p in self.clusters[i].population] )
            original_points_out.extend(rand_points)

            #original_points_out.extend( [p.phenome for p in self.clusters[i].population] )

        original_trans_points_in = cluster.transform( original_points_in, get_original=True )
        original_trans_points_out = cluster.transform( original_points_out, get_original=True )

        # Feature: transform error
        transform_error = sum((original_trans_points_in[:,-1]-1)**2)
        transform_error = transform_error + sum((original_trans_points_out[:,-1]-1)**2)

        trans_best_point = cluster.transform( [original_best_point] )[0]
        trans_points_in = cluster.transform( original_points_in )
        trans_points_out = cluster.transform( original_points_out )


        # Feature: distance of the transformed best point to center [0.5] * dimension
        best_point_to_center = np.linalg.norm( trans_best_point - 0.5 )

        # Feature: distance of the transformed mean point to center [0.5] * dimension
        mean_position = np.mean(trans_points_in, axis=0)
        mean_position_to_center = np.linalg.norm( mean_position - 0.5 )

        # Feature: find distance from trans_points_in that are out to border
        dist_points_in = sum(trans_points_in[np.where(trans_points_in>1)]-1) - \
                         sum(trans_points_in[np.where(trans_points_in<0)])
        #dist_points_in = dist_points_in/len(trans_points_in)

        # Feature: find distance from trans_points_out that are in to border
        trans_points_out = trans_points_out[np.all(trans_points_out >= 0, axis=1)]
        trans_points_out = trans_points_out[np.all(trans_points_out <= 1, axis=1)]
        dist_points_out = sum(trans_points_out[ np.where( \
                                  np.logical_and(trans_points_out>=0, trans_points_out<0.5) \
                              )]) + \
                          sum(trans_points_out[ np.where( \
                                  np.logical_and(trans_points_out>=0.5, trans_points_out<=1) \
                              )]-0.5)
        #dist_points_out = dist_points_out/len(trans_points_out)
       

        # Feature: find distance of vertices that are out to border
        #num_rand_points = 100 
        dist_vertice_out = 0
        trans_rand_points = np.random.rand(self.n_rand_points,self.dimension)
        rand_points = cluster.transform_inverse( trans_rand_points )

        diff = self.boundary.min_bounds - rand_points
        dist_vertice_out += sum( diff[diff > 0] )

        diff = rand_points - self.boundary.max_bounds
        dist_vertice_out += sum( diff[diff > 0] )

        #dist_vertice_out /= (self.boundary.max_bounds[0] - self.boundary.min_bounds[0])
        dist_vertice_out /= self.n_rand_points

        
                
        means = np.mean(original_points_in, axis=0)
        stds = np.std(original_points_in, axis=0)


        limit = 10
        mean, std = 0.5, 0.2
        norm_distribution = np.random.normal( mean, std, trans_points_in.shape )
        KL_divergence = entropy( norm_distribution, trans_points_in )
        #uniform_distribution = np.random.normal( 0, 1, trans_points_in.shape )
        #KL_divergence = entropy( uniform_distribution, trans_points_in ).clip( -limit, limit )



        norm_distribution = np.random.normal( mean, std, np.array(original_points_in).shape )
        original_points = cluster.transform_inverse( norm_distribution )
        #KL_divergence = entropy( original_points, original_points_in )
        
        ori_max = np.amax( original_points, axis=0 )
        ori_max_in = np.amax( original_points_in, axis=0 )
        ori_min = np.amin( original_points, axis=0 )
        ori_min_in = np.amin( original_points_in, axis=0 )

        mean_position_to_center = np.linalg.norm( mean_position - 0.5 )
        dist_mean = np.linalg.norm( np.mean(original_points, axis=0) - np.mean(original_points_in, axis=0) )

        dist_max = np.linalg.norm( ori_max - ori_max_in )
        dist_min = np.linalg.norm( ori_min - ori_min_in )
        
        hull = ConvexHull(rand_points) 
        #trans_hull = ConvexHull(trans_points_in) 



        score = 0
        score += 100*dist_points_in
        score += 50*best_point_to_center
        score += dist_points_out 
        score += dist_vertice_out 
        #score += transform_error  
        #score += dist_mean
        #score += dist_max
        #score += dist_min
        score += sum(KL_divergence.clip(0,1)/self.dimension)
        #score *= hull.volume
        #score += (hull.volume / ((self.boundary.max_bounds[0] - self.boundary.min_bounds[0])**2) )
        score = score * (1+ (hull.volume / ((self.boundary.max_bounds[0] - self.boundary.min_bounds[0])**2) ))


        if debug:
            print(cluster.matrix)
            #print(cluster.transform_inverse( np.array([[0,0],[1,0],[1,1],[0,1]]) ))
            #print(cluster.transform_inverse( np.array([[0,0],[1,0],[1,1],[0,1]]), get_original=True ))
            print(trans_points_out)
            print('           dist_mean: ', dist_mean )
            print('            dist_min: ', dist_min )
            print('            dist_max: ', dist_max)
            print('             entropy: ', KL_divergence)
            print('best_point_to_center: ', best_point_to_center)
            print('      dist_points_in: ', dist_points_in)
            print('     dist_points_out: ', dist_points_out)
            print('    dist_vertice_out: ', dist_vertice_out)
            #print('     transform_error: ', transform_error)
            #print('                hull: ', (hull.volume / (self.boundary.max_bounds[0] - self.boundary.min_bounds[0]) ))
            print('                hull: ', (hull.volume / ((self.boundary.max_bounds[0] - self.boundary.min_bounds[0])**2) ))
            #print('          trans_hull: ', trans_hull.volume )
            print('               score: ', score)
            print('')

        else:
            cluster.matrix = original_matrix

        #score = (1+dist_points_in+mean_position_to_center) \
        #      + (1+dist_points_out+best_point_to_center) \
        #      + 10.*transform_error
        '''
        score = 10*(1+dist_points_in)  \
              + 1*(1+dist_points_out) \
              + 1*(1+dist_vertice_out) \
              + 1*(1+transform_error) 
              #+ sum(KL_divergence) 
              #+ (hull.volume / ((self.boundary.max_bounds[0] - self.boundary.min_bounds[0])**2) )
              #+ 10*abs(1.-trans_hull.volume)
        '''
        return score


    def update_borders( self ):
        for i in range(len(self.clusters)):
            cluster = self.clusters[i]
            len_bounds = np.linalg.norm(cluster.border[1]-cluster.border[1])
            es = CMAEvolutionStrategy( cluster.border.ravel().tolist(), len_bounds*0.1, 
                                       {'bounds':[ self.boundary.min_bounds[0], 
                                                   self.boundary.max_bounds[0] ] } )
            while not es.stop():
                solutions = es.ask()
                #TODO
                es.tell( solutions, [self.evaluate_border(border, i) for border in solutions] ) 
                #es.tell( solutions, [cluster.evaluate_border(border) for border in solutions] ) 

                x_best = es.result()[0]
                #if x_best is not None and cluster.in_global_border( x_best ):
                if x_best is not None:
                    cluster.border = x_best.reshape( cluster.border.shape )

            '''
            vertices = np.concatenate(( cluster.border, cluster.border[0:1] ))
            min_edge = min( [np.linalg.norm(vertices[i]-vertices[i+1]) for i in range(len(vertices)-1)] )
            original_points_in = [ p.phenome for p in cluster.population ]
            trans_points_in = cluster.transform( original_points_in )
            mean_position = np.mean(trans_points_in, axis=0)
            mean_position_to_center = np.linalg.norm( mean_position - 0.5 )
            print'mean: ', mean_position_to_center 
            '''

    def in_global_boundary( self, points ): 
        for point in points:
            if not self.boundary.in_boundary( point ):
                return False
        return True


    def evaluate_border( self, border, cluster_index ):

        cluster = self.clusters[ cluster_index ]
        border = border.reshape( cluster.border.shape )
        border_hull = ConvexHull(border) 

        if len(border_hull.vertices) != cluster.border.shape[0]: #number of vertices in border
            return np.inf 

        # Transform data points 
        original_best_point = min( cluster.population, key=attrgetter('objective_values') ).phenome 
        original_points_in = [ p.phenome for p in cluster.population ]
        original_points_out = []
        for i in range(len(self.clusters)):
            if i == cluster_index: continue
            original_points_out.extend( [p.phenome for p in self.clusters[i].population] )


        #transform = ProjectiveTransform() 
        #dst_cord = np.array([[0,0],[1,0],[1,1],[0,1]])
        #if not transform.estimate(border, dst_cord):
            #return np.inf
            #raise Exception("estimation failed")

        trans_best_point = cluster.transform( [original_best_point] )[0]
        trans_points_in = cluster.transform( original_points_in )
        trans_points_out = cluster.transform( original_points_out )

        original_trans_points_in = cluster.transform( original_points_in, True )
        original_trans_points_out = cluster.transform( original_points_out, True )

        transform_error = sum(original_trans_points_in[:,-1] - 1)
        transform_error = transform_error + sum(original_trans_points_out[:,-1] - 1)

        # Feature 1: distance of the transformed best point to center [0.5] * dimension
        best_point_to_center = np.linalg.norm( trans_best_point - 0.5 )

        # Feature 2: find distance from trans_points_in that are out to border
        dist_points_in = sum(trans_points_in[np.where(trans_points_in>1)]-1) - \
                         sum(trans_points_in[np.where(trans_points_in<0)])

        # Feature 3: find distance from trans_points_out that are in to border
        dist_points_out = sum(trans_points_out[ np.where( \
                                  np.logical_and(trans_points_out>=0, trans_points_out<0.5) \
                              )]) + \
                          sum(trans_points_out[ np.where( \
                                  np.logical_and(trans_points_out>=0.5, trans_points_out<=1) \
                              )]-0.5)
        
        #hull = ConvexHull(trans_points_in) 

        #total_area = sum(np.square((self.boundary.max_bounds - self.boundary.min_bounds)))

        points_in_out_of_hull = len(cluster.population) - sum( cluster.in_hull( original_points_in, border) ) 
        #points_out_in_hull = sum( cluster.in_hull( original_points_out, border) ) 

        original_hull = ConvexHull(original_points_in)

        area_diff = border_hull.volume - original_hull.volume
        if area_diff < 0:
            area_diff = -2*area_diff

        # V1
        #score = (border_hull.volume) * (1+points_in_out_of_hull)

        # V2
        #score = (border_hull.volume) * (1+points_in_out_of_hull) + area_diff*(1+dist_points_out+best_point_to_center)

       
        mean_position = np.mean(trans_points_in, axis=0)
        mean_position_to_center = np.linalg.norm( mean_position - 0.5 )

        # V3
        #score = (border_hull.volume) * (1+points_in_out_of_hull+mean_position_to_center) \
        #      + area_diff*(1+dist_points_out+best_point_to_center)

        # V4
        score = (1+points_in_out_of_hull+mean_position_to_center) \
              + (1+dist_points_out+best_point_to_center) \
              + 10 * transform_error

        return score
        #score = area_diff * (1+points_in_out_of_hull)#+dist_points_out)
        #score = area_diff * (1+points_in_out_of_hull)#+points_out_in_hull)
        #score += (border_hull.volume/total_area) * (1+dist_points_in)
        #score += -hull.volume 
        #score += best_point_to_center 
        #score += dist_points_in   
        #score += dist_points_out

        print('chisquare: ', chisquare( trans_points_in ))
        print('border.volume: ', border_hull.volume)
        print('dist_points_in: ', dist_points_in)
        print('dist_points_out: ', dist_points_out)
        print('border.volume: ', border_hull.volume / total_area)
        print('hull.volume: ', hull.volume )
        print('best_point_to_center: ', best_point_to_center )
        print('chisquare: ', chisquare( trans_points_in ))
        #print('point_to_center: ', best_point_to_center)
        #print('border_hull.volume: ', border_hull.volume )
        '''
        score = 0.0
        for i in range(len(self.clusters)):
            cluster = self.clusters[i]
            border = borders[i]

            hull = ConvexHull(border)
            if len(hull.vertices) != border.shape[0]: #number of vertices in border
                return np.inf 


            points_in = [p.phenome for p in cluster.population]
            points_out = []
            for j in range( len(self.clusters) ):
                if j == i: continue
                #points_out.extend([p.phenome for p in self.clusters[i].population])
                points_out.extend(borders[j])
    
            delaunay = Delaunay(border)
            points_in_excluded = sum(delaunay.find_simplex(points_in) < 0)
            points_out_included = sum(delaunay.find_simplex(points_out) >= 0)

            score += hull.volume * ( 1 + points_in_excluded ) + points_out_included

        return score
        ''' 
    def get_ranks( self, clusters ):
        ranks = []
        values = [ [p.objective_values for p in cluster.population] for cluster in clusters]
        #all_ranks = rankdata( np.hstack(values), method='dense') 
        all_ranks = rankdata( np.hstack(values), method='min') 
        start = 0
        for value in values:
            end = start+len(value)
            ranks.append( all_ranks[start:end].tolist() )
            start = end
        return ranks

    def run( self ):
        self.iteration = self.iteration + 1 

        # Choose best cluster to update
        ranks = self.get_ranks(self.clusters)
        remain_f_allocation = Combination( self.problem.remaining_evaluations, len(self.clusters), self.n_points, ranks, model='linear', debug=False).combination

        self.remain_f_allocation += np.array(remain_f_allocation)
        best_arm = np.argmax(self.remain_f_allocation)
        remain_f = np.amax(remain_f_allocation)

        if self.algo_type == 'CMA':
            if self.algos[best_arm].stop():
                self.remain_f_allocation[best_arm] = - self.remain_f_allocation[best_arm]
                if self.verbose: print('CMA-ES at cluster %d stops!!'%(best_arm))
                #draw_contour( function_id, clusters=bandit.clusters )
                return

        print('Update cluster %d'%best_arm)

        # TODO
        #self.remain_f_allocation[best_arm] = 0
        self.remain_f_allocation[best_arm] -= len(self.clusters[best_arm].population)

        # Transform data points to [0-1] space and resume algorithm
        original_points = [ p.phenome for p in self.clusters[best_arm].population ]
        trans_points = self.clusters[best_arm].transform( original_points ).clip(0,1)
        fitness_values = [ p.objective_values for p in self.clusters[best_arm].population ]

        new_trans_positions = self.update_positions( best_arm, trans_points, fitness_values )

        # Update cluster.population
        new_positions = self.clusters[best_arm].transform_inverse(new_trans_positions)
        solutions = [Individual(position) for position in new_positions]

        try:
            self.problem.batch_evaluate(solutions)
        except ResourcesExhausted:
            self.should_terminate = True
            return


        self.clusters[best_arm].population = sorted( solutions, key=attrgetter('objective_values') )   
        ranks = self.get_ranks(self.clusters)
        for i, cluster in enumerate(self.clusters):
            cluster.ranks = ranks[i]

        self.best_solution = self.update_best_solution()


        # Check if need to recluster
        trans_mean_position = np.mean(new_trans_positions, axis=0)
        best_point = self.clusters[best_arm].population[0].phenome
        trans_best_point = self.clusters[best_arm].transform( [best_point] ).clip(0,1)[0]
        margin = 0.05
        if ((trans_best_point < margin).any() or (trans_best_point > 1-margin).any()) and \
           ((trans_mean_position < 2*margin).any() or (trans_mean_position > 1-2*margin).any()):

                if self.verbose: 
                    print('Reclustering...')  
                    print('due to best_point of cluster %d at: %r'%(best_arm,trans_best_point))
                    print('                      and mean at: %r'%(trans_mean_position) )
                if args.draw_contour > 0: draw_contour( self.function_id, clusters=self.clusters )
                self.shift_matrix(best_arm, best_point)
                if args.draw_contour > 0: draw_contour( self.function_id, clusters=self.clusters )
                self.recluster( len(self.clusters) )
                if args.draw_contour > 0: draw_contour( self.function_id, clusters=self.clusters )

        self.update_statistics(best_arm)

    def shift_matrix( self, i, best_point):
        center = self.clusters[i].transform_inverse([[0.5]*self.dimension])[0]
        translate = best_point - center
        translate_matrix = np.eye(self.dimension+1)
        translate_matrix[0:-1,-1] = -translate.T 
        self.clusters[i].matrix = np.dot(self.clusters[i].matrix, translate_matrix)
        #self.update_matrix(i)
        if self.algo_type == 'CMA':
            self.algos[i] = CMAEvolutionStrategy( [0.5]*self.dimension, 0.2, 
                                                  {'popsize': self.n_points, 
                                                   'bounds': [ 0, 1 ] } ) 
            init_positions = self.algos[i].ask()


    def update_positions( self, k, X, f ):
        if self.algo_type == 'CMA':
            #algo_es.feedForResume( trans_points, fitness_values )
            self.algos[k].tell( X, f )
            new_trans_positions = self.algos[k].ask()
            return new_trans_positions

    def update_best_solution( self ):
        best_solutions = [] 
        for cluster in self.clusters:
            best_solutions.append(min( cluster.population, key=attrgetter('objective_values') ) )
        return min( best_solutions, key=attrgetter('objective_values') )

    def found_optimum( self, delta=1e-8 ):  
        if self.best_solution.objective_values - self.optimal_solution.objective_values < delta:
            return True
        return False


    def stop( self ):
        if self.problem.remaining_evaluations < 1:
            if self.verbose: print('Consumed all evaluations!')
            self.should_terminate = True
        elif self.found_optimum( delta=goal_error ):
            if self.verbose: print('Found Optimum!')
            self.should_terminate = True
        elif self.algo_type == 'CMA':
            all_algo_stop = True 
            for algo in self.algos:
                if not algo.stop():
                    all_algo_stop = False
                    break
            if all_algo_stop:
                if self.verbose: print('All algorithms stop!')
                self.should_terminate = True
        return self.should_terminate 


    def print_status( self ):
        error = self.best_solution.objective_values - self.optimal_solution.objective_values
        print('')
        print('    Iteration %d: error = %e'%(self.iteration, error))
        print('     Evaluations: consumed %d, remain %d'%(self.problem.consumed_evaluations, 
                                               self.problem.remaining_evaluations))
        print('      allocation: %r'%self.remain_f_allocation.tolist())
        print('    best fitness: %f at %r'%(self.best_solution.objective_values, 
                                            self.best_solution.phenome.tolist()))
        print('optimal solution: %f at %r'%(self.optimal_solution.objective_values, 
                                            self.optimal_solution.phenome))
        print('')

    def update_statistics( self, best_arm ):
        self.stats['iteration'].append(self.iteration)
        self.stats['FEs'].append(self.problem.consumed_evaluations)
        self.stats['error'].append(self.best_solution.objective_values - self.optimal_solution.objective_values)
        self.stats['best_value'].append(self.best_solution.objective_values)
        self.stats['pulled_arm'].append(best_arm)
        self.stats['num_arms'].append(len(self.clusters))
        #self.stats['best_position'].append(self.best_solution.phenome.tolist())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Bandit on CEC2005 problems')
    parser.add_argument('--algorithms', '-a', default=0, type=int, 
                        help='0:CMA-ES, 1:PSO, 2:ACOR')
    parser.add_argument('--max_evaluations', '-fe', default=10000, type=int, 
                        help='FEs = 1e4*Dimension')
    parser.add_argument('--population', '-n', default=6, type=int, 
                        help='population') 
    parser.add_argument('--dimension', '-d', default=2, type=int, 
                        help='Dimensions = 2, 10, 30, 50 for all CEC2005 problems') 
    parser.add_argument('--function_id', '-i', default=9, type=int, 
                        help='function_id = 1~25 for all CEC2005 problems') 

    parser.add_argument('--verbose', '-v', default=True, type=bool, 
                        help='print process')
    parser.add_argument('--draw_contour', '-draw', default=0, type=int, 
                        help='number of iterations to show clusters and contour')
    parser.add_argument('--csv_file', '-csv', default=None, type=str, 
                        help='Save statistics into csv file')
    parser.add_argument('--figure_directory', '-fig_dir', default=None, type=str, 
                        help='Save figures into directory')


    args = parser.parse_args()

    n_points = args.population 
    dimension = args.dimension 
    max_evaluations = args.max_evaluations*args.dimension
    function_id = args.function_id - 1
    csv_file = args.csv_file 
    figure_directory = args.figure_directory
    goal_error=1e-8
    
    if figure_directory:
        if os.path.exists(figure_directory):
            print('%s already exists!' % figure_directory )
        else:
            os.makedirs(figure_directory)

    if len(sys.argv) < 2:
        parser.print_help()
    else:
        algo = Bandit( max_evaluations, n_points, dimension, function_id,
                       algo_type='CMA', verbose=args.verbose )

        if args.verbose: algo.print_status()
        if args.draw_contour > 0: 
            it = 0
            if figure_directory:
                FE = algo.problem.consumed_evaluations
                error = algo.best_solution.objective_values - \
                        algo.optimal_solution.objective_values
                draw_contour( function_id, clusters=algo.clusters,
                              fig_name="%sF%d_%d"%(figure_directory,function_id+1,it), 
                              fig_title="F%d, FE=%d, error=%e"%(function_id+1, FE, error) )
            else:
                draw_contour( function_id, clusters=algo.clusters )
            #draw_all( function_id, clusters=bandit.clusters )
            #draw_surface3d( function_id, clusters=bandit.clusters )


        while not algo.stop():
            algo.run()

            if args.verbose: algo.print_status()
            if args.draw_contour > 0: 
                it += 1

                if it % args.draw_contour == 0:

                    if figure_directory:
                        FE = algo.problem.consumed_evaluations
                        error = algo.best_solution.objective_values - \
                                algo.optimal_solution.objective_values
                        draw_contour( function_id, clusters=algo.clusters,
                                      fig_name="%sF%d_%d"%(figure_directory,function_id+1,it), 
                                      fig_title="F%d, FE=%d, error=%e"%(function_id+1, FE, error) )
                    else:
                        draw_contour( function_id, clusters=algo.clusters )



        if args.draw_contour > 0:
            if figure_directory:
                FE = algo.problem.consumed_evaluations
                error = algo.best_solution.objective_values - \
                        algo.optimal_solution.objective_values
                draw_contour( function_id, clusters=algo.clusters,
                              fig_name="%sF%d_%d"%(fure_directory,function_id+1,it), 
                              fig_title="F%d, FE=%d, error=%e"%(function_id+1, FE, error) )
            else:
                draw_contour( function_id, clusters=algo.clusters )

        if not args.verbose: algo.print_status()
        if csv_file is not None: pd.DataFrame(algo.stats).to_csv(csv_file, index=False)
