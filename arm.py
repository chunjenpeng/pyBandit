from copy import deepcopy
from operator import attrgetter
import numpy as np 

from CMAES import CMA
from SPSO2011 import PSO
from matrix import Matrix


class Arm:
    def __init__(self, obj, init_positions, init_fitnesses):

        self.matrix = Matrix(deepcopy(init_positions))
        self.obj = obj

        # Transform points onto subspace and start algorithm
        trans_positions = self.matrix.transform(init_positions) 
        assert (trans_positions.all() <= 1)
        assert (trans_positions.all() >= 0)

        n_points = len(init_fitnesses)
        dimension = len(trans_positions[0])
        self.algo = CMA(self.transform_obj, n_points, dimension, 
                        init_positions=trans_positions, 
                        init_fitnesses=init_fitnesses )

    def transform_obj(self, X):
        original_X = self.matrix.inverse_transform(X) 
        return self.obj(original_X)

    def get_positions(self):
        return self.matrix.inverse_transform( self.algo.get_positions() )
    def get_fitnesses(self):
        return self.algo.get_fitnesses()

    def update_matrix(self, positions_in, fitnesses, positions_out):
        self.matrix.update(positions_in, fitnesses, positions_out)


    def pull(self):
        best_position, best_fitness = self.algo.run() 
        return self.matrix.inverse_transform(best_position), best_fitness
        
    def stop(self):
        return self.algo.stop()


    def reached_border(self):
        margin = 0.05

        trans_positions = self.algo.get_positions()
        trans_mean_position = np.mean(trans_positions, axis=0)

        best_index = np.argmax(self.algo.get_fitnesses())  
        trans_best_position = trans_positions[best_index]

        if ((trans_best_point < margin).any() or (trans_best_point > 1.0-margin).any()) and \
           ((trans_mean_position < 2.0*margin).any() or (trans_mean_position > 1.0-2.0*margin).any()):
            return True
        return False

def draw_arms(function_id, arms, **kwargs):

    import os
    from optproblems.cec2005 import CEC2005
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from boundary import Boundary

    # Parameters
    dim = 2
    function = CEC2005(dim)[function_id].objective_function
    optimal_pos = CEC2005(dim)[function_id].get_optimal_solutions()[0].phenome
    boundary = Boundary(dim, function_id)

    k = len(arms)
    inch_size = 4
    fig_w = k + 1
    fig_h = 1
    fig = plt.figure(figsize = (fig_w * inch_size, fig_h * inch_size))
    cmap = cm.coolwarm
    scatter_cmap = cm.jet(np.linspace(0.1, 0.9, k))
    angle = kwargs.get('angle', 240)
    rotate = kwargs.get('rotate', False)
    fig_name = kwargs.get('fig_name', None)
    fig_title = kwargs.get('fig_title', 'F%d'%(function_id+1))
    fig_dir = kwargs.get('fig_dir', 'test_arms')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Get Mesh Solutions for contour
    step = (boundary.max_bounds[0] - boundary.min_bounds[0]) / 100
    X = np.arange(boundary.min_bounds[0], boundary.max_bounds[0] + step, step)
    Y = np.arange(boundary.min_bounds[1], boundary.max_bounds[1] + step, step)
    (X, Y) = np.meshgrid(X, Y)
    positions = [ [x, y] for x, y in zip(X.ravel(), Y.ravel()) ]
    Z = np.array( [ function(position) for position in positions ] )

    # Reset colormap to get rid of extreme colors
    vmin = min(Z)
    vmax = max(Z)
    vmin = vmin - (vmax - vmin) * 0.2
    vmax = vmax + (vmax - vmin) * 0.2
    Z = Z.reshape(X.shape)

    # Plot contour
    ax = fig.add_subplot(fig_h, fig_w, 1)
    ax.set_xlim([ boundary.min_bounds[0], boundary.max_bounds[0]])
    ax.set_ylim([ boundary.min_bounds[1], boundary.max_bounds[1]])
    cset = ax.contourf(X, Y, Z, cmap = cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(cset, aspect = 20)


    # Plot scatter points in each arm
    colors = iter(scatter_cmap)
    for arm in arms:
        color = next(colors)
        positions = arm.get_positions()
        ax.scatter(positions[:,0], positions[:,1], color = color, marker = 'o', s = 10)

        # Plot borders on original boundary
        subspace_border = np.array([ [ 0, 0], [ 1, 0], [ 1, 1], [ 0, 1], [ 0, 0]])
        border = arm.matrix.inverse_transform( subspace_border )
        ax.plot(border[:, 0], border[:, 1], color = color)
    
    # Plot optimal solution as a big white 'X'
    ax.scatter(optimal_pos[0], optimal_pos[1], color = 'w', marker = 'x', s = 100)


    # Plot from each arm's perspective
    for (i, arm) in enumerate(arms):

        color = scatter_cmap[i]
        ax = fig.add_subplot(fig_h, fig_w, i + 2)
        ax.set_xlim([ -0.01, 1.01])
        ax.set_ylim([ -0.01, 1.01])

        # Plot contour
        (X, Y) = np.meshgrid( np.arange(0, 1.01, 0.01), np.arange(0, 1.01, 0.01) )

        positions = [ [x, y] for x, y in zip(X.ravel(), Y.ravel())]
        original_positions = arm.matrix.inverse_transform(positions)

        Z = np.array( [ function(position) for position in original_positions ] )
        Z = Z.reshape(X.shape)

        cset = ax.contourf(X, Y, Z, cmap = cmap, vmin = vmin, vmax = vmax)


        # Plot scatter points in each arm
        trans_X = arm.matrix.transform( arm.get_positions() )
        ax.scatter(trans_X[:, 0], trans_X[:, 1], color = color, marker = 'o', s = 10)

        # Plot border
        cord = np.array([ [0, 0], [1, 0], [1, 1], [0, 1]])
        ax.plot(cord[[0, 1, 2, 3, 0], 0], cord[[0, 1, 2, 3, 0], 1], color = color)
    



    fig.tight_layout()
    st = fig.suptitle(fig_title, fontsize = 16)
    st.set_y(0.95)
    fig.subplots_adjust(top = 0.85)
    if fig_name is not None:
        plt.savefig( '%s/%s' % (fig_dir,fig_name) )
    else:
        plt.show()
        input('Press Enter to continue...')
    plt.close(fig)


def testArm(plot=False):
    from optproblems.cec2005 import CEC2005
    from sklearn.cluster import KMeans

    function_id = 0
    dimension = 2 
    n_points = 60
    k = 3
    function = CEC2005(dimension)[function_id].objective_function

    init_positions = np.random.uniform(-100,100,size=(n_points, dimension))
    init_fitnesses = np.array([function(p) for p in init_positions])

    index = init_fitnesses.argsort()
    selected = index[:int(n_points/2)]
    positions, fitnesses = init_positions[selected], init_fitnesses[selected]
    labels = KMeans( n_clusters=k ).fit_predict(positions)


    it = 0
    arms = []
    for i in range(k):
        indices = np.where(labels==i)[0]
        arms.append( Arm(function, positions[indices], fitnesses[indices]) )


    if plot: draw_arms( function_id, arms, fig_name='it_%d.png' % it )

    while True:
        for arm in arms:
            if not arm.stop():
                it += 1
                best_position, best_fitness = arm.pull()
                print('Iter', it, best_fitness, best_position)
            if plot: draw_arms( function_id, arms, fig_name='it_%d.png' % it )


if __name__ == '__main__':
    #testArm(plot=True)
    for i in range(100):
        testArm()
