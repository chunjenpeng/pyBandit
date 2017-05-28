from copy import deepcopy
import numpy as np

from OnePlusOne_ES import OnePlusOne_ES

class Matrix:
    def __init__(self, positions, matrix=None):
        self.delta = 1e-08
        if matrix is None:
            self.matrix = self.init_matrix( np.array(positions) )
        else:
            self.matrix = matrix


    
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
        if len(positions) == 0:
            return []

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
        if len(positions) == 0:
            return []

        positions = np.array(positions)
        positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
        trans_positions = np.dot(np.linalg.inv(self.matrix), positions.T).T

        if not get_original:
            w = trans_positions[:, -1]
            w[w == 0] = self.delta
            trans_positions = trans_positions[:, :-1] / w[:, None]

        return trans_positions



    def optimize(self, best, include, exclude, min_bounds, max_bounds, max_evaluation_num = 10000):

        dimension = self.matrix.shape[0] - 1
        n_samples = 100 * dimension 

        # Repeatedly used parameters in evaluate
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.original_best_position = deepcopy(best)
        self.original_positions_in  = deepcopy(include)
        self.original_positions_out = deepcopy(exclude)
        self.samples = np.random.uniform(0, 1, size=(n_samples, dimension))

        best_solution = self.matrix.ravel()
        best_score = self.evaluate( best_solution )
        #print( 'Init score:', self.evaluate(best_solution, debug=True) )

        # Use (1+1)-ES to optimize matrix
        repeat = 10
        for _ in range(repeat):
            es = OnePlusOne_ES( self.evaluate, len(best_solution), 
                                parent = self.matrix.ravel(),
                                #parent = best_solution, 
                                step = 0.1,
                                max_iteration = max_evaluation_num / repeat )
            while not es.stop():
                x_best, score = es.run()
            if score < best_score:
                best_score = score
                best_solution = x_best
            #print( 'Final score:', score) 


        self.matrix = np.array(best_solution).reshape( self.matrix.shape )
        #print( 'Final score:', self.evaluate(best_solution, debug=True) )


    def evaluate(self, X, debug=False):
        self.matrix = np.array(X).reshape( self.matrix.shape )

        trans_best = self.transform([self.original_best_position])[0]
        
        trans_in   = self.transform(self.original_positions_in)

        if self.original_positions_out.any():
            trans_out  = self.transform(self.original_positions_out)
            trans_out  = trans_out[ np.all( trans_out >= 0, axis=1) ]
            trans_out  = trans_out[ np.all( trans_out <= 1, axis=1) ]


        ori_samples = self.inverse_transform(self.samples)

        out_min_bounds = self.min_bounds - ori_samples
        out_min_bounds = out_min_bounds[ out_min_bounds > 0 ]
        dist_out_min_bounds = sum( out_min_bounds )

        out_max_bounds = ori_samples - self.max_bounds
        out_max_bounds = out_max_bounds[ out_max_bounds > 0 ]
        dist_out_max_bounds = sum( out_max_bounds )


        # Features to be minimized
        dist_best_to_center = np.linalg.norm( trans_best - 0.5 )

        dist_should_be_in   = abs(sum( trans_in[ np.where(trans_in > 1.0) ] - 1.0 ))
        dist_should_be_in  += abs(sum( trans_in[ np.where(trans_in < 0.0) ] ))

        dist_should_be_out = 0 
        if self.original_positions_out.any():
            lower_half = np.where( np.logical_and( trans_out >= 0.0, trans_out < 0.5 ) )
            upper_half = np.where( np.logical_and( trans_out >= 0.5, trans_out <= 1.0 ) )
            dist_should_be_out  = abs(sum( trans_out[ lower_half ] ))
            dist_should_be_out += abs(sum( trans_out[ upper_half ] - 0.5 ))

        
        reconstruct = self.inverse_transform( np.clip(trans_in, 0, 1) )
        reconstruct_error = sum( np.linalg.norm( p1 - p2 ) \
                                 for p1, p2 in zip(reconstruct, self.original_positions_in) )

        trans_std = np.std( (trans_in - trans_best), axis=0 )
        dist_std = sum( abs(trans_std - 0.3) ) 
        

        score  = 0.0
        score += 100*reconstruct_error 
        # Limit in global boundary
        score += dist_should_be_in 
        score += dist_should_be_out 
        # Split point in and out of cluster
        score += dist_out_min_bounds
        score += dist_out_max_bounds
        # Approximate a Normal distribution centering at trans_best
        score += dist_best_to_center
        score += dist_std

        if not debug:
            return score 
        else:
            #print('trans_out:\n', trans_out)
            #print('trans_in:\n', trans_in)
            #print('trans_best:\n', trans_best)
            #print('original:\n', self.original_positions_in)
            #print('reconstruct:\n', reconstruct)
            #if reconstruct_error > 1:
            if True:
                print('dist_in  :', dist_should_be_in)
                print('dist_min :', dist_out_min_bounds)
                print('dist_max :', dist_out_max_bounds)
                print('dist_out :', dist_should_be_out)
                print('dist_best:', dist_best_to_center)
                print('std      :', trans_std)
                print('dist_std :', dist_std)
                print('error    :', reconstruct_error)
                print('score    :', score)
                #print(self.matrix)
                subspace_border = np.array([ [ 0, 0], [ 1, 0], [ 1, 1], [ 0, 1] ])
                border = self.inverse_transform( subspace_border )
                #print(subspace_border)
                #print(border)
                #print()
            return score 
            #return reconstruct_error



if __name__ == '__main__':

    dimension = 2
    n_points = 5 
    X = np.random.uniform(-100, 100, size = (n_points, dimension))
    m = Matrix(X)
    m.matrix = np.random.uniform(-100, 100, size = (dimension+1, dimension+1) )

    print('\nTransformation Matrix:')
    print(m.matrix)
    print('\nInverse Matrix:')
    print(np.linalg.inv(m.matrix))

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
    
