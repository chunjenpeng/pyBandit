# Source Generated with Decompyle++
# File: cluster.cpython-34.pyc (Python 3.4)

from scipy.spatial import ConvexHull
from skimage.transform import ProjectiveTransform
import numpy as np

class Cluster:
    __qualname__ = 'Cluster'
    
    def __init__(self, population, ranks, **kwargs):
        self.population = population
        self.ranks = ranks
        dim = len(population[0].phenome)
        self.matrix = self.init_matrix(dim)

    
    def init_matrix(self, dim):
        positions = np.array((lambda .0: continue[ p.phenome for p in .0 ])(self.population))
        amax = np.amax(positions, axis = 0)
        amin = np.amin(positions, axis = 0)
        scale = amax - amin
        scale_matrix = np.eye(dim + 1)
        for i in range(dim):
            if scale[i] != 0:
                scale_matrix[(i, i)] = 1 / scale[i]
                continue
        translate_matrix = np.eye(dim + 1)
        translate_matrix[(0:-1, -1)] = -(amin.T)
        return np.dot(scale_matrix, translate_matrix)

    
    def transform(self, positions, **kwargs):
        get_original = kwargs.get('get_original', False)
        positions = np.array(positions)
        positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
        trans_positions = np.dot(self.matrix, positions.T).T
        if not get_original:
            w = trans_positions[(:, -1)]
            trans_positions = trans_positions[(:, :-1)] / w[(:, None)]
        return trans_positions

    
    def transform_inverse(self, positions, **kwargs):
        get_original = kwargs.get('get_original', False)
        positions = np.array(positions)
        positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
        trans_positions = np.dot(np.linalg.inv(self.matrix), positions.T).T
        if not get_original:
            w = trans_positions[(:, -1)]
            trans_positions = trans_positions[(:, :-1)] / w[(:, None)]
        return trans_positions


