from itertools import combinations
from operator import itemgetter
from scipy.stats import rankdata
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
        #print(i, closest_indices)

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

    #print(directed_edges)
    #print(labels)
    return np.array(labels)
    


def weighted_gaussian(positions, fitnesses):

    # Inverse ranks so that min fitness has max value of rank
    fitnesses = -1*np.array(fitnesses)
    inverse_ranks = rankdata(fitnesses, method='ordinal')
    weights = 1.0/sum(inverse_ranks) * inverse_ranks
    #print(weights)

    #n = len(fitnesses)
    #weights = np.log(n + 0.5) - np.log(inverse_ranks)
    #weights = weights / sum(weights)
    #print(weights)
    #print(positions) 

    mean = np.ma.average( positions, axis = 0, weights = weights )
    #print(mean)
    xm = positions - mean
    #print(xm)
    #print(weights[:,None] * xm )
    covariance = (weights[:,None] * xm).T.dot(xm)
    #print(covariance)
    return mean, covariance 


def MDL(clusters_positions, clusters_fitnesses):

    # J: number of clusters
    J = len(clusters_positions)
    # D: number of dimension
    D = len(clusters_positions[0][0])
    # I: total number of points
    I = sum(len(c) for c in clusters_positions)

    score = J * (D**2 + 3*D + 2) * np.log(I)/2

    #print('J:', J) 
    #print('D:', D) 
    #print('I:', I) 
    #print('score:', score) 

    for positions, fitnesses in zip(clusters_positions, clusters_fitnesses):
        # n: number of points in each cluster
        n = len(positions) 
        mean, covariance = weighted_gaussian(positions, fitnesses)
        det = np.linalg.det(covariance)
        if n > 0 and det > 0:
            score -= n * np.log( n*n/det )
        '''
        else:
            print('n:', n)
            print('det:', det)
            print('cov:\n', covariance)
        '''
        #assert n != 0
        #assert det != 0
        #score -= n * np.log( n*n/det )

        #print('n:', n) 
        #print('cov:\n', covariance) 
        #print('|cov|:', np.linalg.det(covariance)) #print(n * np.log( n*n/np.linalg.det(covariance) ) )
        #print('score:', score) 
    
    return score 



def calculate_MDL_scores( positions, fitnesses, labels ): 
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

        clusters_positions.append( merged_positions )
        clusters_fitnesses.append( merged_fitnesses )

        scores[combination] = MDL(clusters_positions, clusters_fitnesses)

    return scores



def clustering(positions, fitnesses):

    labels = hierarchical_clustering( positions, fitnesses )
    k = max(labels) + 1

    clusters_positions, clusters_fitnesses = [], []
    for i in range(k):
        #print(i)
        indices = np.where(labels==i)[0]
        #print(indices)
        clusters_positions.append( positions[indices] ) 
        clusters_fitnesses.append( fitnesses[indices] ) 

    #draw( clusters_positions, clusters_fitnesses, obj, fig_name = 'test.png' )


    original_score = MDL(clusters_positions, clusters_fitnesses)


    scores = calculate_MDL_scores( positions, fitnesses, labels )
    merge_index = np.unravel_index(scores.argmin(), scores.shape) 
    min_score = scores[merge_index] 
    print(labels)
    print('scores:\n', scores) 
    print('min_score:', scores[merge_index])
    print('original_score:', original_score)

   
    while min_score < original_score:
        print('merge:', merge_index)
        max_indices = np.where(labels==max(labels))[0]
        merge_indices = np.where(labels==max(merge_index))[0]
        labels[max_indices] = max(merge_index)
        labels[merge_indices] = min(merge_index)
        print(labels)

        original_score = min_score

        scores = calculate_MDL_scores( positions, fitnesses, labels )
        merge_index = np.unravel_index(scores.argmin(), scores.shape) 
        min_score = scores[merge_index]
        print('scores:\n', scores) 
        print('min_score:', scores[merge_index])
        print('original_score:', original_score)

    return labels



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


if __name__ == '__main__':

    from boundary import Boundary
    function_id = 10 
    dimension = 2
    n_points = 200
    obj = CEC2005(dimension)[function_id].objective_function
    min_bounds = Boundary(dimension, function_id).min_bounds
    max_bounds = Boundary(dimension, function_id).max_bounds
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
        #print(i)
        indices = np.where(labels==i)[0]
        #print(indices)
        clusters_positions.append( positions[indices] ) 
        clusters_fitnesses.append( fitnesses[indices] ) 

    draw( clusters_positions, clusters_fitnesses, obj, fig_name = 'init.png',
          xlim = [ min_bounds[0], max_bounds[0] ],
          ylim = [ min_bounds[1], max_bounds[1] ] )



    labels = clustering( positions, fitnesses )
    k = max(labels) + 1

    clusters_positions, clusters_fitnesses = [], []
    for i in range(k):
        #print(i)
        indices = np.where(labels==i)[0]
        #print(indices)
        clusters_positions.append( positions[indices] ) 
        clusters_fitnesses.append( fitnesses[indices] ) 

    draw( clusters_positions, clusters_fitnesses, obj, fig_name = 'test.png',
          xlim = [ min_bounds[0], max_bounds[0] ],
          ylim = [ min_bounds[1], max_bounds[1] ] )






