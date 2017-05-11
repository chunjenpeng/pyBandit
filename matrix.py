# Source Generated with Decompyle++
# File: matrix.cpython-34.pyc (Python 3.4)

from scipy.spatial import ConvexHull
from skimage.transform import ProjectiveTransform
import numpy as np

class Matrix:
    def __init__(self, positions):
        self.delta = 1e-08
        self.matrix = self.init_matrix(positions)

    
    def init_matrix(self, positions):
        dim = len(positions[0])
        amax = np.amax(positions, axis = 0) + self.delta
        amin = np.amin(positions, axis = 0) - self.delta

        scale = amax - amin
        scale_matrix = np.eye(dim + 1)
        for i in range(dim):
            if scale[i] != 0:
                scale_matrix[i, i] = 1.0 / scale[i]


        translate_matrix = np.eye(dim + 1)
        translate_matrix[:-1, -1] = -(amin.T)

        return np.dot(scale_matrix, translate_matrix)

    
    def transform(self, positions, **kwargs):
        get_original = kwargs.get('get_original', False)

        positions = np.array(positions)
        positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
        trans_positions = np.dot(self.matrix, positions.T).T

        if not get_original:
            w = trans_positions[:, -1]
            w[w == 0] = self.delta
            trans_positions = trans_positions[:, :-1] / w[:, None]

        #assert (trans_positions <= 1).all() 
        #assert (trans_positions >= 0).all() 
        return trans_positions

    
    def inverse_transform(self, positions, **kwargs):
        get_original = kwargs.get('get_original', False)

        positions = np.array(positions)
        positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
        trans_positions = np.dot(np.linalg.inv(self.matrix), positions.T).T

        if not get_original:
            w = trans_positions[:, -1]
            w[w == 0] = self.delta
            trans_positions = trans_positions[:, :-1] / w[:, None]

        return trans_positions

    
    def update(self, positions_in, fitnesses, positions_out):
        pass


if __name__ == '__main__':

    dimension = 2
    n_points = 5 
    X = np.random.uniform(-100, 100, size = (n_points, dimension))
    m = Matrix(X)
    print('\nTransformation Matrix:')
    print(m.matrix)

    print('\noriginal points:')
    print(X)

    print('\ntransformed points:')
    trans_X = m.transform(X, get_original=True) 
    print(trans_X)

    print('\nnormalized transformed points:')
    trans_X = m.transform(X)
    print(trans_X)

    print('\nrestored points:')
    restored_X = m.inverse_transform(trans_X, get_original=True) 
    print(restored_X)
        
    print('\nnormalized restored points:')
    restored_X = m.inverse_transform(trans_X) 
    print(restored_X)
    
