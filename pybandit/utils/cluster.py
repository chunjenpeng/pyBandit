from __future__ import print_function
import sys, os
from collections import Counter
from itertools import combinations
from operator import itemgetter
from scipy.stats import rankdata, multivariate_normal
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from optproblems.cec2005 import CEC2005
import numpy as np


def generate_distance_matrix(positions):
    n_points, dimension = positions.shape 
    matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i+1, n_points):
            matrix[i, j] = np.linalg.norm( positions[i] - positions[j] )
            matrix[j, i] = matrix[i, j]
    return matrix



def find_directed_edges( distance_matrix, fitnesses, num_neighbors ):

    directed_edges = {}

    for i, fitness in enumerate(fitnesses):
        closest_indices = []
        for d in range(num_neighbors):
            closest = np.argmin(distance_matrix[i])
            closest_indices.append(closest)
            distance_matrix[i,closest] = np.inf

        min_index = np.argmin( [fitnesses[c] for c in closest_indices] )
        directed_edges[i] = closest_indices[min_index]

    return directed_edges



def hierarchical_clustering(positions, fitnesses):
        
    n_points, dimension = positions.shape 

    distance_matrix = generate_distance_matrix( positions )  

    num_neighbors = 4 + int(3*np.log(dimension))
    directed_edges = find_directed_edges( distance_matrix, fitnesses, num_neighbors )

    labels = [-1] * n_points
    label = 0
    for v in directed_edges:
        if labels[v] != -1: 
            continue

        merged = False 
        cluster = [v]
        while directed_edges[v] != v:

            if labels[ directed_edges[v] ] != -1:
                _label = labels[ directed_edges[v] ]
                while cluster:
                    i = cluster.pop()
                    labels[i] = _label
                merged = True 
                break

            cluster.append( directed_edges[v] )
            v = directed_edges[v]

        while cluster:
            i = cluster.pop()
            labels[i] = label

        if not merged:
            label += 1

    return np.array(labels)


# Merge all clusters.size <= 2 with the label nearest to the best_fitness position
def force_small_sample_clustering( labels, positions, fitnesses ): 

    num_labels = max(labels) + 1
    best_label_index = [ labels.index(l) for l in range(num_labels) ]
    for i, label in enumerate(labels):
        if fitnesses[i] < fitnesses[ best_label_index[label] ]:
            best_label_index[label] = i


    counter = Counter(labels)  
    should_merge = {}
    for label, count in counter.items():
        if count > 2: continue

        best_position = positions[ best_label_index[label] ]
        nearest_distance, nearest_label = np.inf, label

        for i, position in enumerate(positions):
            if labels[i] != label:
                distance = np.linalg.norm( best_position - position )
                if distance < nearest_distance and counter[ labels[i] ] > 2:
                    nearest_label = labels[i]

        should_merge[ label ] = nearest_label


    for from_label, to_label in should_merge.items():
        # Change all from_label to to_label
        for i, l in enumerate(labels):
            if l == from_label:
                labels[i] = to_label

    counter = Counter(labels)  
    label, max_label = 0, max(labels)
    while label < max_label:
        if counter[label] == 0:
            for i, l in enumerate(labels):
                if l == max_label:
                    labels[i] = label 
        label += 1
        max_label = max(labels)

    return np.array(labels)
    


def calculate_weights(fitnesses):

    # Inverse ranks so that min fitness has max value of rank
    ranks = rankdata(fitnesses, method='ordinal')
    weights = np.log(len(fitnesses) + 0.5) - np.log(ranks)
    weights = weights / sum(weights)

    return weights



def weighted_gaussian(positions, fitnesses):

    weights = calculate_weights(fitnesses)

    mean = np.average( positions, axis = 0, weights = weights )

    xm = positions - mean
    covariance = (weights[:,None] * xm).T.dot(xm)

    return mean, covariance 



def MDL(clusters_positions, clusters_fitnesses):

    # J: number of clusters
    J = len(clusters_positions)
    # D: number of dimension
    D = len(clusters_positions[0][0])
    # I: total number of points
    I = sum(len(c) for c in clusters_positions)

    # Model description score
    score = ( (J * (D**2 + 3*D + 2)/2) - 1) * np.log(I)/2


    # Data Liklihood score
    for positions, fitnesses in zip(clusters_positions, clusters_fitnesses):

        # alpha: number of points in each cluster / total number of points
        n = len(positions)
        alpha = float(n)/I

        weights = calculate_weights(fitnesses)
        
        mean, cov = weighted_gaussian(positions, fitnesses)

        residuals = positions - mean

        def calc_loglikelihood(residuals):
            # More stable version for calculating log(det)
            sign, logdet = np.linalg.slogdet(cov)
            #det = sign * np.exp(logdet)

            # Quick hack to avoid singular matrix
            inv_cov = np.linalg.inv( cov + np.eye(cov.shape[1])*1e-8 )

            return -0.5 * (logdet + residuals.T.dot(inv_cov).dot(residuals) + 2 * np.log(2 * np.pi))

        loglikelihood = np.apply_along_axis(calc_loglikelihood, 1, residuals)

        # Original version
        #score -= np.log(alpha)*len(loglikelihood) + sum(loglikelihood)
        # V1
        #score -= np.log(alpha)*len(loglikelihood) + np.dot(np.log(weights).T, loglikelihood)
        # V2
        #score -= np.log(alpha)*len(loglikelihood) + sum(np.log(weights)) + sum(loglikelihood)
        # V3 
        #for w, l, p in zip(weights, loglikelihood, positions):
        #    score -= (n * w * (np.log(alpha) + l))
        # V4:
        #score -= n * ( np.log(alpha) * sum(weights) + np.dot( weights.T, loglikelihood ) )
        # V5: (sum(weights) == 1)
        score -= n * ( np.log(alpha) + np.dot( weights.T, loglikelihood ) )

    return score 



def calculate_MDL_scores( positions, fitnesses, labels ): 

    positions, fitnesses = np.array(positions), np.array(fitnesses)
    k = max(labels) + 1
    scores = np.empty((k+1, k+1)) 
    scores[:] = np.inf

    for combination in combinations( range(k) , 2 ):
        merged_positions, merged_fitnesses = [], []
        clusters_positions, clusters_fitnesses = [], []

        for i in range(k):
            indices = np.where(labels==i)[0]

            if i in combination:
                merged_positions.extend( positions[indices] ) 
                merged_fitnesses.extend( fitnesses[indices] ) 
            else:
                clusters_positions.append( positions[indices] ) 
                clusters_fitnesses.append( fitnesses[indices] ) 

        clusters_positions.append( np.array(merged_positions) )
        clusters_fitnesses.append( np.array(merged_fitnesses) )

        scores[combination] = MDL(clusters_positions, clusters_fitnesses)

    return scores



def clustering(positions, fitnesses):

    labels = hierarchical_clustering( positions, fitnesses )
    labels = trim_by_MDL( positions, fitnesses, labels ) 

    return labels


def trim_by_MDL( positions, fitnesses, labels ):

    assert len(labels) == len(fitnesses) == len(positions)
    positions, fitnesses = np.array(positions), np.array(fitnesses)

    labels = force_small_sample_clustering( list(labels), positions, fitnesses )

    k = max(labels) + 1

    clusters_positions, clusters_fitnesses = [], []
    for i in range(int(k)):
        #print(i)
        indices = np.where(labels==i)[0]
        clusters_positions.append( positions[indices] ) 
        clusters_fitnesses.append( fitnesses[indices] ) 

    #draw( clusters_positions, clusters_fitnesses, obj, fig_name = 'test.png' )

    original_score = MDL(clusters_positions, clusters_fitnesses)


    scores = calculate_MDL_scores( positions, fitnesses, labels )
    merge_index = np.unravel_index(scores.argmin(), scores.shape) 
    min_score = scores[merge_index] 
    #print(labels)
    #print('scores:\n', scores) 
    #print('original_score:', original_score)

   
    while min_score < original_score:
        #print('merge:', merge_index, 'min_score:', min_score)
        max_indices = np.where(labels==max(labels))[0]
        merge_indices = np.where(labels==max(merge_index))[0]
        labels[max_indices] = max(merge_index)
        labels[merge_indices] = min(merge_index)
        #print(labels)

        original_score = min_score

        scores = calculate_MDL_scores( positions, fitnesses, labels )
        merge_index = np.unravel_index(scores.argmin(), scores.shape) 
        min_score = scores[merge_index]
        #print('scores:\n', scores) 
        #print('original_score:', original_score)
        
        ###########################################
        '''
        from boundary import Boundary
        dimension = 2
        obj = CEC2005(dimension)[function_id].objective_function
        min_bounds = Boundary(dimension, function_id).min_bounds
        max_bounds = Boundary(dimension, function_id).max_bounds
        clusters_positions, clusters_fitnesses = [], []
        for i in range(k):
            indices = np.where(labels==i)[0]
            clusters_positions.append( positions[indices] ) 
            clusters_fitnesses.append( fitnesses[indices] ) 
        k = max(labels)+1
        fig_name = '2017_07_03_MDL_selection/F%d_%d.png'%(function_id+1, k)
        print('K = %d, drawing'%k, fig_name)
        draw( clusters_positions, clusters_fitnesses, obj, 
              fig_name = fig_name,
              xlim = [ min_bounds[0], max_bounds[0] ],
              ylim = [ min_bounds[1], max_bounds[1] ] )
        '''
        ###########################################

    return np.array(labels)



def draw_gaussian(ax, color, mean, cov):

    from matplotlib.patches import Ellipse

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    for nstd in range(1,3):
        width, height = 2 * nstd * np.sqrt(abs(vals))
        ellip = Ellipse( xy = mean,
                         width = width,
                         height = height,
                         angle = theta )
        ellip.set_alpha(0.2)
        ellip.set_facecolor(color)
        ax.add_artist(ellip)


def draw( clusters_positions, clusters_fitnesses, obj, fig_name, **kwargs ):

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
    fig_dir = kwargs.get( 'fig_dir', './' )
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig = plt.figure(figsize=plt.figaspect(0.85))
    ax = fig.add_subplot(111, aspect=1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cmap = cm.coolwarm
    scatter_cmap = cm.jet( np.linspace(0.1, 0.9, len(clusters_positions)) )
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
    colors = iter(scatter_cmap)
    for positions, fitnesses in zip(clusters_positions, clusters_fitnesses):
        color = next(colors)
        ax.scatter(positions[:,0], positions[:,1], color=color,  s=10)

        mean, cov = weighted_gaussian( positions, fitnesses ) 
        #print('mean:\n', mean)
        #print('cov:\n', cov)
        draw_gaussian(ax, color, mean, cov)

    plt.savefig('%s/%s' % (fig_dir, fig_name) )
    plt.close(fig)


def manhalanobis_distance(x, mean, cov):
    assert len(x) == len(mean) == len(cov) == len(cov[0])
    xm = x - mean
    inv_cov = np.linalg.inv( cov + np.eye(cov.shape[1])*1e-8 )
    return xm.dot(inv_cov).dot(xm.T)


def run(function_id, fig_dir):
    from boundary import Boundary
    dimension = 2
    n_points = 200 
    function_id = function_id-1
    obj = CEC2005(dimension)[function_id].objective_function
    min_bounds = Boundary(dimension, function_id).min_bounds
    max_bounds = Boundary(dimension, function_id).max_bounds
    '''
    positions = np.array([
                          [0.75, -1.4],
                          [0.90, -1.4],
                          [1.05, -1.4],
                          [1.75, -1.4],
                          [1.90, -1.4],
                          [2.05, -1.4],
                          [0.75, -1.6],
                          [0.90, -1.6],
                          [1.05, -1.6],
                          [1.75, -1.6],
                          [1.90, -1.6],
                          [2.05, -1.6],
                        ])
    '''
    positions = np.random.uniform( min_bounds, max_bounds, size=(n_points, dimension) )
    fitnesses = np.array([ obj(p) for p in positions ])

    indices = fitnesses.argsort()
    selected = indices[:int(len(positions)/2)]
    positions, fitnesses = positions[selected], fitnesses[selected]
    #mean, cov = weighted_gaussian( positions, fitnesses ) 

    labels = hierarchical_clustering( positions, fitnesses )
    k = max(labels) + 1

    clusters_positions, clusters_fitnesses = [], []
    for i in range(k):
        indices = np.where(labels==i)[0]
        clusters_positions.append( positions[indices] ) 
        clusters_fitnesses.append( fitnesses[indices] ) 


    fig_name = 'F%d_init.png' % (function_id+1)
    print('K = %d, drawing'%k, fig_name)

    draw( clusters_positions, clusters_fitnesses, obj, 
          fig_name = fig_name,
          fig_dir = fig_dir,
          xlim = [ min_bounds[0], max_bounds[0] ],
          ylim = [ min_bounds[1], max_bounds[1] ] )



    labels = trim_by_MDL( positions, fitnesses, labels ) 
    #labels = clustering( positions, fitnesses )
    k = max(labels) + 1

    clusters_positions, clusters_fitnesses = [], []
    for i in range(k):
        #print(i)
        indices = np.where(labels==i)[0]
        clusters_positions.append( positions[indices] ) 
        clusters_fitnesses.append( fitnesses[indices] ) 

        #mean, cov = weighted_gaussian( positions[indices], fitnesses[indices] ) 
        #distances = [ manhalanobis_distance(x, mean, cov) for x in positions[indices] ]
        #idx = np.argsort(distances)
        #for i in idx:
        #    print( positions[indices][i], fitnesses[indices][i], distances[i] )

        #h = -fitnesses[indices][idx] - min(-fitnesses[indices])
        #h = h/sum(h)
        #print(h)
        #n, bins, patches = plt.hist(h, len(h)

        #print()


    fig_name = 'F%d_MDL_GMM.png' % (function_id+1)
    print('K = %d, drawing'%k, fig_name)

    draw( clusters_positions, clusters_fitnesses, obj, 
          fig_name = fig_name,
          fig_dir = fig_dir,
          xlim = [ min_bounds[0], max_bounds[0] ],
          ylim = [ min_bounds[1], max_bounds[1] ] )


if __name__ == '__main__':


    if len(sys.argv) < 2:
        print()
        print('         python3 cluster.py <function_id (1~25)> <dir(default: ./)>')
        print('Example: python3 cluster.py 1')
        print('Example: python3 cluster.py 25 test/')
        print()
    else:

        function_id = 1 # F1 ~ F25
        fig_dir = './'

        if len(sys.argv) == 2:
            function_id = int(sys.argv[1])
        elif len(sys.argv) == 3:
            function_id = int(sys.argv[1])
            fig_dir = sys.argv[2]
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

        run(function_id, fig_dir = fig_dir)




