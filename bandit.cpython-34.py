# Source Generated with Decompyle++
# File: bandit.cpython-34.pyc (Python 3.4)

import os
import sys
import copy
import argparse
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
from optproblems.cec2005 import CEC2005
from optproblems import Individual, Problem
from optproblems.base import ResourcesExhausted
from cluster import Cluster
from boundary import Boundary
from combination import Combination
from plot import draw_surface3d, draw_cluster, draw_all, draw_contour

class Bandit:
    __qualname__ = 'Bandit'
    
    def __init__(self, max_evaluations, n_points, dimension, function_id, **kwargs):
        self.function_id = function_id
        self.n_points = n_points
        self.dimension = dimension
        self.function = CEC2005(dimension)[function_id]
        self.problem = Problem(self.function.objective_function, max_evaluations = max_evaluations)
        self.boundary = Boundary(dimension, function_id)
        self.verbose = kwargs.get('verbose', False)
        self.max_n_clusters = min(self.n_points, kwargs.get('max_n_clusters', 10))
        self.clusters = []
        self.n_rand_points = 50 * dimension
        self.init_clusters(self.n_rand_points)
        self.remain_f_allocation = np.zeros(len(self.clusters))
        self.algo_type = kwargs.get('algo_type', 'CMA')
        self.algos = []
        self.init_algos()
        self.iteration = 0
        self.should_terminate = False
        self.optimal_solution = self.find_optimal_solution()
        self.best_solution = self.update_best_solution()
        self.stats = OrderedDict([
            ('iteration', []),
            ('FEs', []),
            ('error', []),
            ('best_value', []),
            ('pulled_arm', []),
            ('num_arms', [])])

    
    def find_optimal_solution(self):
        dimension = self.dimension
        function_id = self.function_id
        optimal_solutions = CEC2005(dimension)[function_id].get_optimal_solutions()
        test_prob = Problem(CEC2005(dimension)[function_id].objective_function)
        test_prob.batch_evaluate(optimal_solutions)
        return min(optimal_solutions, key = attrgetter('objective_values'))

    
    def init_clusters(self, n_points):
        positions = np.zeros((n_points, self.dimension))
        for d in range(self.dimension):
            positions[(:, d)] = np.random.uniform(self.boundary.init_min_bounds[d], self.boundary.init_max_bounds[d], n_points)
        
        population = (lambda .0: continue[ Individual(position) for position in .0 ])(positions)
        self.problem.batch_evaluate(population)
        population = sorted(population, key = attrgetter('objective_values'))
        selected_population = population[:len(population) / 2]
        X = np.array((lambda .0: continue[ p.phenome for p in .0 ])(selected_population))
        k = self.estimate_n_clusters(X, self.max_n_clusters)
        if self.verbose:
            print('popsize:%d, k:%d' % (len(selected_population), k))
        ranks = range(1, len(selected_population) + 1)
        self.clusters = [
            Cluster(selected_population, ranks)]
        population_step = 20
        min_k = k
        while None:
            positions = np.zeros((population_step, self.dimension))
            for d in range(self.dimension):
                positions[(:, d)] = np.random.uniform(self.boundary.min_bounds[d], self.boundary.max_bounds[d], population_step)
            
            new_population = (lambda .0: continue[ Individual(position) for position in .0 ])(positions)
            population.extend(new_population)
            population = sorted(population, key = attrgetter('objective_values'))
            selected_population = population[:len(population) / 2]
            X = np.array((lambda .0: continue[ p.phenome for p in .0 ])(selected_population))
            k = self.estimate_n_clusters(X, self.max_n_clusters)
            if self.verbose:
                print('popsize:%d, k:%d' % (len(selected_population), k))
            if k <= min_k:
                break
            min_k = k
        self.clusters = self.k_means(k)
        for i in range(k):
            self.update_matrix(i)
        
        if args.draw_contour > 0:
            draw_contour(self.function_id, clusters = self.clusters)
        self.resize_each_cluster()

    
    def resize_each_cluster(self):
        for cluster in self.clusters:
            delta_population = len(cluster.population) - self.n_points
            if delta_population < 0:
                trans_positions = np.random.uniform(0, 1, (abs(delta_population), self.dimension))
                positions = cluster.transform_inverse(trans_positions)
                new_population = (lambda .0: continue[ Individual(position) for position in .0 ])(positions)
                self.problem.batch_evaluate(new_population)
                cluster.population.extend(new_population)
            cluster.population = sorted(cluster.population, key = attrgetter('objective_values'))
            cluster.population = cluster.population[:self.n_points]
        
        ranks = self.get_ranks(self.clusters)
        for (i, cluster) in enumerate(self.clusters):
            cluster.ranks = ranks[i]
            if self.verbose:
                print('cluster%d: %r' % (i, self.clusters[i].ranks))
                continue

    
    def estimate_n_clusters(self, X, max_n_clusters):
        score = np.zeros(max_n_clusters + 1)
        score[0] = -1
        for k in range(2, max_n_clusters + 1):
            km = KMeans(n_clusters = k)
            labels = km.fit_predict(X)
            score[k] = silhouette_score(X, labels)
        
        return np.argmax(score)

    
    def recluster(self, max_n_clusters):
        positions = []
        for cluster in self.clusters:
            positions.extend((lambda .0: continue[ p.phenome for p in .0 ])(cluster.population))
        
        X = np.array(positions)
        k = self.estimate_n_clusters(X, len(self.clusters))
        should_recluster = True
        while should_recluster:
            self.remain_f_allocation = np.zeros(k)
            if self.verbose:
                print('recluster to %d clusters' % k)
            old_clusters = copy.deepcopy(self.clusters)
            old_algos = copy.deepcopy(self.algos)
            old_ranks = (lambda .0: continue[ cluster.ranks for cluster in .0 ])(self.clusters)
            if self.verbose:
                print('old_ranks:')
            for i in range(len(self.clusters)):
                if self.verbose:
                    print('cluster%d:%r' % (i, old_ranks[i]))
                    continue
            new_clusters = self.k_means(k)
            new_ranks = self.get_ranks(new_clusters)
            if self.verbose:
                print('new_ranks:')
            for i in range(k):
                if self.verbose:
                    print('cluster%d:%r' % (i, new_ranks[i]))
                    continue
            should_recluster = False
            for cluster in new_clusters:
                if len(cluster.population) < 2:
                    k = k - 1
                    should_recluster = True
                    break
                    continue
        unchanged_clusters = (lambda .0: for (i, ranks) in .0:
if ranks in old_ranks:
continue[][i])(enumerate(new_ranks))
        if self.verbose:
            print('unchanged_clusters:%r' % unchanged_clusters)
        new_algos = []
        for (index, cluster) in enumerate(new_clusters):
            if index in unchanged_clusters:
                for (old_index, cluster) in enumerate(self.clusters):
                    if ((None,), all)((lambda .0: continue)(cluster.ranks)):
                        new_clusters[index] = copy.deepcopy(self.clusters[old_index])
                        new_algos.append(copy.deepcopy(self.algos[old_index]))
                        break
                        continue
                    continue
                    if self.algo_type == 'CMA':
                        new_algos.append(CMAEvolutionStrategy([
                            0.5] * self.dimension, 0.2, {
                            'popsize': self.n_points,
                            'bounds': [
                                0,
                                1] }))
                        init_positions = new_algos[index].ask()
                        continue
                    self.clusters = new_clusters
                    self.algos = new_algos
                    for i in range(len(self.clusters)):
                        if i not in unchanged_clusters:
                            self.update_matrix(i)
                            continue
                    self.resize_each_cluster()
                    return None

    
    def init_algos(self):
        del self.algos[:]
        if self.algo_type == 'CMA':
            for cluster in self.clusters:
                self.algos.append(CMAEvolutionStrategy([
                    0.5] * self.dimension, 0.2, {
                    'popsize': self.n_points,
                    'bounds': [
                        0,
                        1] }))
            
            for algo in self.algos:
                new_trans_positions = algo.ask()
            

    
    def k_means(self, n_clusters):
        population = []
        for cluster in self.clusters:
            population.extend(cluster.population)
        
        X = np.array((lambda .0: continue[ p.phenome for p in .0 ])(population))
        labels = KMeans(n_clusters = n_clusters).fit_predict(X)
        clusters = (lambda .0: continue[ [] for i in .0 ])(range(n_clusters))
        ranks = (lambda .0: continue[ [] for i in .0 ])(range(n_clusters))
        for i in xrange(len(population)):
            clusters[labels[i]].append(population[i])
            ranks[labels[i]].append(i + 1)
        
        return (lambda .0: continue[ Cluster(cluster, rank) for (cluster, rank) in .0 ])(zip(clusters, ranks))

    
    def update_matrix(self, i):
        cluster = self.clusters[i]
        solution = cluster.matrix.ravel()[0:-1]
        std = 0.0005
        es = CMAEvolutionStrategy(solution.tolist(), std, {
            'maxiter': 1000 })
        best_score = self.evaluate_matrix(solution, i)
        while not es.stop():
            solutions = es.ask()
            None((es.tell, solutions), (lambda .0: continue[ self.evaluate_matrix(solution, i) for solution in .0 ])(solutions))
            x_best = es.result()[0]
            current_best_score = self.evaluate_matrix(x_best, i)
            if x_best is not None and current_best_score > best_score:
                cluster.matrix = np.append(x_best, 1).reshape(cluster.matrix.shape)
                best_score = current_best_score
                continue
            return None

    
    def evaluate_matrix(self, solution, cluster_index, debug = False):
        cluster = self.clusters[cluster_index]
        original_matrix = cluster.matrix
        cluster.matrix = np.append(solution, 1).reshape(cluster.matrix.shape)
        original_best_point = min(cluster.population, key = attrgetter('objective_values')).phenome
        original_points_in = (lambda .0: continue[ p.phenome for p in .0 ])(cluster.population)
        original_points_out = []
        for i in range(len(self.clusters)):
            if i == cluster_index:
                continue
            trans_rand_points = np.random.rand(self.n_rand_points, self.dimension)
            rand_points = self.clusters[i].transform_inverse(trans_rand_points).tolist()
            rand_points.extend((lambda .0: continue[ p.phenome for p in .0 ])(self.clusters[i].population))
            original_points_out.extend(rand_points)
        
        original_trans_points_in = cluster.transform(original_points_in, get_original = True)
        original_trans_points_out = cluster.transform(original_points_out, get_original = True)
        transform_error = sum((original_trans_points_in[(:, -1)] - 1) ** 2)
        transform_error = transform_error + sum((original_trans_points_out[(:, -1)] - 1) ** 2)
        trans_best_point = cluster.transform([
            original_best_point])[0]
        trans_points_in = cluster.transform(original_points_in)
        trans_points_out = cluster.transform(original_points_out)
        best_point_to_center = np.linalg.norm(trans_best_point - 0.5)
        mean_position = np.mean(trans_points_in, axis = 0)
        mean_position_to_center = np.linalg.norm(mean_position - 0.5)
        dist_points_in = sum(trans_points_in[np.where(trans_points_in > 1)] - 1) - sum(trans_points_in[np.where(trans_points_in < 0)])
        trans_points_out = trans_points_out[np.all(trans_points_out >= 0, axis = 1)]
        trans_points_out = trans_points_out[np.all(trans_points_out <= 1, axis = 1)]
        dist_points_out = sum(trans_points_out[np.where(np.logical_and(trans_points_out >= 0, trans_points_out < 0.5))]) + sum(trans_points_out[np.where(np.logical_and(trans_points_out >= 0.5, trans_points_out <= 1))] - 0.5)
        dist_vertice_out = 0
        trans_rand_points = np.random.rand(self.n_rand_points, self.dimension)
        rand_points = cluster.transform_inverse(trans_rand_points)
        diff = self.boundary.min_bounds - rand_points
        dist_vertice_out += sum(diff[diff > 0])
        diff = rand_points - self.boundary.max_bounds
        dist_vertice_out += sum(diff[diff > 0])
        dist_vertice_out /= self.n_rand_points
        means = np.mean(original_points_in, axis = 0)
        stds = np.std(original_points_in, axis = 0)
        limit = 10
        (mean, std) = (0.5, 0.2)
        norm_distribution = np.random.normal(mean, std, trans_points_in.shape)
        KL_divergence = entropy(norm_distribution, trans_points_in)
        norm_distribution = np.random.normal(mean, std, np.array(original_points_in).shape)
        original_points = cluster.transform_inverse(norm_distribution)
        ori_max = np.amax(original_points, axis = 0)
        ori_max_in = np.amax(original_points_in, axis = 0)
        ori_min = np.amin(original_points, axis = 0)
        ori_min_in = np.amin(original_points_in, axis = 0)
        mean_position_to_center = np.linalg.norm(mean_position - 0.5)
        dist_mean = np.linalg.norm(np.mean(original_points, axis = 0) - np.mean(original_points_in, axis = 0))
        dist_max = np.linalg.norm(ori_max - ori_max_in)
        dist_min = np.linalg.norm(ori_min - ori_min_in)
        hull = ConvexHull(rand_points)
        score = 0
        score += 100 * dist_points_in
        score += 50 * best_point_to_center
        score += dist_points_out
        score += dist_vertice_out
        score += sum(KL_divergence.clip(0, 1) / self.dimension)
        score = score * (1 + hull.volume / (self.boundary.max_bounds[0] - self.boundary.min_bounds[0]) ** 2)
        if debug:
            print(cluster.matrix)
            print(trans_points_out)
            print('           dist_mean: ', dist_mean)
            print('            dist_min: ', dist_min)
            print('            dist_max: ', dist_max)
            print('             entropy: ', KL_divergence)
            print('best_point_to_center: ', best_point_to_center)
            print('      dist_points_in: ', dist_points_in)
            print('     dist_points_out: ', dist_points_out)
            print('    dist_vertice_out: ', dist_vertice_out)
            print('                hull: ', hull.volume / (self.boundary.max_bounds[0] - self.boundary.min_bounds[0]) ** 2)
            print('               score: ', score)
            print('')
        else:
            cluster.matrix = original_matrix
        return score

    
    def update_borders(self):
        for cluster in range(len(self.clusters)):
            i = None
            len_bounds = np.linalg.norm(cluster.border[1] - cluster.border[1])
            es = CMAEvolutionStrategy(cluster.border.ravel().tolist(), len_bounds * 0.1, {
                'bounds': [
                    self.boundary.min_bounds[0],
                    self.boundary.max_bounds[0]] })
            while not es.stop():
                solutions = es.ask()
                None((es.tell, solutions), (lambda .0: continue[ self.evaluate_border(border, i) for border in .0 ])(solutions))
                x_best = es.result()[0]
                if x_best is not None:
                    cluster.border = x_best.reshape(cluster.border.shape)
                    continue

    
    def in_global_boundary(self, points):
        for point in points:
            if not self.boundary.in_boundary(point):
                return False
        
        return True

    
    def evaluate_border(self, border, cluster_index):
        cluster = self.clusters[cluster_index]
        border = border.reshape(cluster.border.shape)
        border_hull = ConvexHull(border)
        if len(border_hull.vertices) != cluster.border.shape[0]:
            return np.inf
        original_best_point = min(cluster.population, key = attrgetter('objective_values')).phenome
        original_points_in = (lambda .0: continue[ p.phenome for p in .0 ])(cluster.population)
        original_points_out = []
        for i in range(len(self.clusters)):
            if i == cluster_index:
                continue
            original_points_out.extend((lambda .0: continue[ p.phenome for p in .0 ])(self.clusters[i].population))
        
        trans_best_point = cluster.transform([
            original_best_point])[0]
        trans_points_in = cluster.transform(original_points_in)
        trans_points_out = cluster.transform(original_points_out)
        original_trans_points_in = cluster.transform(original_points_in, True)
        original_trans_points_out = cluster.transform(original_points_out, True)
        transform_error = sum(original_trans_points_in[(:, -1)] - 1)
        transform_error = transform_error + sum(original_trans_points_out[(:, -1)] - 1)
        best_point_to_center = np.linalg.norm(trans_best_point - 0.5)
        dist_points_in = sum(trans_points_in[np.where(trans_points_in > 1)] - 1) - sum(trans_points_in[np.where(trans_points_in < 0)])
        dist_points_out = sum(trans_points_out[np.where(np.logical_and(trans_points_out >= 0, trans_points_out < 0.5))]) + sum(trans_points_out[np.where(np.logical_and(trans_points_out >= 0.5, trans_points_out <= 1))] - 0.5)
        points_in_out_of_hull = len(cluster.population) - sum(cluster.in_hull(original_points_in, border))
        original_hull = ConvexHull(original_points_in)
        area_diff = border_hull.volume - original_hull.volume
        if area_diff < 0:
            area_diff = -2 * area_diff
        mean_position = np.mean(trans_points_in, axis = 0)
        mean_position_to_center = np.linalg.norm(mean_position - 0.5)
        score = 1 + points_in_out_of_hull + mean_position_to_center + 1 + dist_points_out + best_point_to_center + 10 * transform_error
        return score
        print('chisquare: ', chisquare(trans_points_in))
        print('border.volume: ', border_hull.volume)
        print('dist_points_in: ', dist_points_in)
        print('dist_points_out: ', dist_points_out)
        print('border.volume: ', border_hull.volume / total_area)
        print('hull.volume: ', hull.volume)
        print('best_point_to_center: ', best_point_to_center)
        print('chisquare: ', chisquare(trans_points_in))

    
    def get_ranks(self, clusters):
        ranks = []
        values = (lambda .0: continue[ (lambda .0: continue[ p.objective_values for p in .0 ])(cluster.population) for cluster in .0 ]
)(clusters)
        all_ranks = rankdata(np.hstack(values), method = 'dense')
        start = 0
        for value in values:
            end = start + len(value)
            ranks.append(all_ranks[start:end].tolist())
            start = end
        
        return ranks

    
    def run(self):
        self.iteration = self.iteration + 1
        ranks = self.get_ranks(self.clusters)
        remain_f_allocation = Combination(self.problem.remaining_evaluations, len(self.clusters), self.n_points, ranks, model = 'linear', debug = False).combination
        self.remain_f_allocation += np.array(remain_f_allocation)
        best_arm = np.argmax(self.remain_f_allocation)
        remain_f = np.amax(remain_f_allocation)
        if self.algo_type == 'CMA' and self.algos[best_arm].stop():
            self.remain_f_allocation[best_arm] = -self.remain_f_allocation[best_arm]
            if self.verbose:
                print('CMA-ES at cluster %d stops!!' % best_arm)
            return None
        print('Update cluster %d' % best_arm)
        self.remain_f_allocation[best_arm] -= len(self.clusters[best_arm].population)
        original_points = (lambda .0: continue[ p.phenome for p in .0 ])(self.clusters[best_arm].population)
        trans_points = self.clusters[best_arm].transform(original_points).clip(0, 1)
        fitness_values = (lambda .0: continue[ p.objective_values for p in .0 ])(self.clusters[best_arm].population)
        new_trans_positions = self.update_positions(best_arm, trans_points, fitness_values)
        new_positions = self.clusters[best_arm].transform_inverse(new_trans_positions)
        solutions = (lambda .0: continue[ Individual(position) for position in .0 ])(new_positions)
        
        try:
            self.problem.batch_evaluate(solutions)
        except ResourcesExhausted:
            self.should_terminate = True
            return None

        self.clusters[best_arm].population = sorted(solutions, key = attrgetter('objective_values'))
        ranks = self.get_ranks(self.clusters)
        for (i, cluster) in enumerate(self.clusters):
            cluster.ranks = ranks[i]
        
        self.best_solution = self.update_best_solution()
        trans_mean_position = np.mean(new_trans_positions, axis = 0)
        best_point = self.clusters[best_arm].population[0].phenome
        trans_best_point = self.clusters[best_arm].transform([
            best_point]).clip(0, 1)[0]
        margin = 0.05
        if (trans_best_point < margin).any() or (trans_best_point > 1 - margin).any():
            if (trans_mean_position < 2 * margin).any() or (trans_mean_position > 1 - 2 * margin).any():
                if self.verbose:
                    print('Reclustering...')
                    print('due to best_point of cluster %d at: %r' % (best_arm, trans_best_point))
                    print('                      and mean at: %r' % trans_mean_position)
                if args.draw_contour > 0:
                    draw_contour(self.function_id, clusters = self.clusters)
                self.shift_matrix(best_arm, best_point)
                if args.draw_contour > 0:
                    draw_contour(self.function_id, clusters = self.clusters)
                self.recluster(len(self.clusters))
                if args.draw_contour > 0:
                    draw_contour(self.function_id, clusters = self.clusters)
                
        self.update_statistics(best_arm)

    
    def shift_matrix(self, i, best_point):
        center = self.clusters[i].transform_inverse([
            [
                0.5] * self.dimension])[0]
        translate = best_point - center
        translate_matrix = np.eye(self.dimension + 1)
        translate_matrix[(0:-1, -1)] = -(translate.T)
        self.clusters[i].matrix = np.dot(self.clusters[i].matrix, translate_matrix)
        if self.algo_type == 'CMA':
            self.algos[i] = CMAEvolutionStrategy([
                0.5] * self.dimension, 0.2, {
                'popsize': self.n_points,
                'bounds': [
                    0,
                    1] })
            init_positions = self.algos[i].ask()

    
    def update_positions(self, k, X, f):
        if self.algo_type == 'CMA':
            self.algos[k].tell(X, f)
            new_trans_positions = self.algos[k].ask()
            return new_trans_positions

    
    def update_best_solution(self):
        best_solutions = []
        for cluster in self.clusters:
            best_solutions.append(min(cluster.population, key = attrgetter('objective_values')))
        
        return min(best_solutions, key = attrgetter('objective_values'))

    
    def found_optimum(self, delta = 1e-08):
        if self.best_solution.objective_values - self.optimal_solution.objective_values < delta:
            return True

    
    def stop(self):
        if self.problem.remaining_evaluations < 1:
            if self.verbose:
                print('Consumed all evaluations!')
            self.should_terminate = True
        elif self.found_optimum(delta = goal_error):
            if self.verbose:
                print('Found Optimum!')
            self.should_terminate = True
        elif self.algo_type == 'CMA':
            all_algo_stop = True
            for algo in self.algos:
                if not algo.stop():
                    all_algo_stop = False
                    break
                    continue
            if all_algo_stop:
                if self.verbose:
                    print('All algorithms stop!')
                self.should_terminate = True
            
        return self.should_terminate

    
    def print_status(self):
        error = self.best_solution.objective_values - self.optimal_solution.objective_values
        print('')
        print('    Iteration %d: error = %e' % (self.iteration, error))
        print('     Evaluations: consumed %d, remain %d' % (self.problem.consumed_evaluations, self.problem.remaining_evaluations))
        print('      allocation: %r' % self.remain_f_allocation.tolist())
        print('    best fitness: %f at %r' % (self.best_solution.objective_values, self.best_solution.phenome.tolist()))
        print('optimal solution: %f at %r' % (self.optimal_solution.objective_values, self.optimal_solution.phenome))
        print('')

    
    def update_statistics(self, best_arm):
        self.stats['iteration'].append(self.iteration)
        self.stats['FEs'].append(self.problem.consumed_evaluations)
        self.stats['error'].append(self.best_solution.objective_values - self.optimal_solution.objective_values)
        self.stats['best_value'].append(self.best_solution.objective_values)
        self.stats['pulled_arm'].append(best_arm)
        self.stats['num_arms'].append(len(self.clusters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run Bandit on CEC2005 problems')
    parser.add_argument('--algorithms', '-a', default = 0, type = int, help = '0:CMA-ES, 1:PSO, 2:ACOR')
    parser.add_argument('--max_evaluations', '-fe', default = 10000, type = int, help = 'FEs = 1e4*Dimension')
    parser.add_argument('--population', '-n', default = 6, type = int, help = 'population')
    parser.add_argument('--dimension', '-d', default = 2, type = int, help = 'Dimensions = 2, 10, 30, 50 for all CEC2005 problems')
    parser.add_argument('--function_id', '-i', default = 9, type = int, help = 'function_id = 1~25 for all CEC2005 problems')
    parser.add_argument('--verbose', '-v', default = True, type = bool, help = 'print process')
    parser.add_argument('--draw_contour', '-draw', default = 0, type = int, help = 'number of iterations to show clusters and contour')
    parser.add_argument('--csv_file', '-csv', default = None, type = str, help = 'Save statistics into csv file')
    parser.add_argument('--figure_directory', '-fig_dir', default = None, type = str, help = 'Save figures into directory')
    args = parser.parse_args()
    n_points = args.population
    dimension = args.dimension
    max_evaluations = args.max_evaluations * args.dimension
    function_id = args.function_id - 1
    csv_file = args.csv_file
    figure_directory = args.figure_directory
    goal_error = 1e-08
    if os.path.exists(figure_directory):
        print('%s already exists!')
    else:
        os.makedirs(figure_directory)
    if len(sys.argv) < 2:
        parser.print_help()
    else:
        algo = Bandit(max_evaluations, n_points, dimension, function_id, algo_type = 'CMA', verbose = args.verbose)
        if args.verbose:
            algo.print_status()
        if args.draw_contour > 0:
            it = 0
            if figure_directory:
                FE = algo.problem.consumed_evaluations
                error = algo.best_solution.objective_values - algo.optimal_solution.objective_values
                draw_contour(function_id, clusters = algo.clusters, fig_name = '%sF%d_%d' % (figure_directory, function_id + 1, it), fig_title = 'F%d, FE=%d, error=%e' % (function_id + 1, FE, error))
            else:
                draw_contour(function_id, clusters = algo.clusters)
        while not algo.stop():
            algo.run()
            if args.verbose:
                algo.print_status()
            if args.draw_contour > 0:
                it += 1
                if it % args.draw_contour == 0:
                    if figure_directory:
                        FE = algo.problem.consumed_evaluations
                        error = algo.best_solution.objective_values - algo.optimal_solution.objective_values
                        draw_contour(function_id, clusters = algo.clusters, fig_name = '%sF%d_%d' % (figure_directory, function_id + 1, it), fig_title = 'F%d, FE=%d, error=%e' % (function_id + 1, FE, error))
                    else:
                        draw_contour(function_id, clusters = algo.clusters)
        if args.draw_contour > 0:
            if figure_directory:
                FE = algo.problem.consumed_evaluations
                error = algo.best_solution.objective_values - algo.optimal_solution.objective_values
                draw_contour(function_id, clusters = algo.clusters, fig_name = '%sF%d_%d' % (fure_directory, function_id + 1, it), fig_title = 'F%d, FE=%d, error=%e' % (function_id + 1, FE, error))
            else:
                draw_contour(function_id, clusters = algo.clusters)
        if not (args.verbose):
            algo.print_status()
        if csv_file is not None:
            pd.DataFrame(algo.stats).to_csv(csv_file, index = False)
