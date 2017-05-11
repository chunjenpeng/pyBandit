from copy import deepcopy
from operator import attrgetter
import numpy as np 

from CMA import CMA
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



def testArm():
    from plot import draw_arms
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

    draw_arms( function_id, arms, fig_name='test_plot/it_%d.png' % it )
    while True:
        for arm in arms:
            if not arm.stop():
                it += 1
                best_position, best_fitness = arm.pull()
                print('Iter', it, best_fitness, best_position)
            draw_arms( function_id, arms, fig_name='test_plot/it_%d.png' % it )


if __name__ == '__main__':
    #testArm()
    for i in range(100):
        testArm()
