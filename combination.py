import sys
import numpy as np

class Combination:
    
    def __init__(self, f_left, n_clusters, n_points, clusters_ranks, **kwargs):
        
        # Sort ranks 
        for ranks in clusters_ranks:
            ranks.sort()

        self.f_left = int(f_left)
        self.n_clusters = int(n_clusters)
        self.n_points = int(n_points)
        self.model = kwargs.get('model', 'uniform')
        self.verbose= kwargs.get('verbose', False)
        self.combination = self.calc_combination(clusters_ranks)

    
    def CDF(self, x):
        x = float(x)
        if self.model == 'quadratic':
            return 1.0 - x * x / (x * (x + 1.0) * (2.0 * x + 1.0) / 6.0)
        elif self.model == 'linear':
            return 1.0 - x / (x * (x + 1.0) / 2.0)
        return (x - 1.0) / x

    
    def buildArm(self, ranks):
        Arm = {}
        Min = self.f_left + ranks[-1]
        Max = 2 * self.f_left + self.n_points

        Arm['vector'] = np.array([ self.CDF(x) for x in range(Min, Max + 1) ])
        Arm['nModels'] = float(self.f_left + (self.n_points + 1 - ranks[-1])) \
                               * (self.f_left + ranks[0])
        Arm['nOnes'] = Arm['nModels'] - Max + 1 - Min
        return Arm

    
    def findMin(self, cache):
        base = np.prod(cache, axis = 0)[0]
        value = base * cache[:, 1] / cache[:, 0]
        return np.argmin(value)

    
    def calc_combination(self, clusters_ranks):
        minValue = 1.0
        combination = [0] * self.n_clusters
        A = [ self.buildArm(ranks) for ranks in clusters_ranks ]
        V = [ A[i]['vector'] for i in range(self.n_clusters) ]
        cache = np.ones((self.n_clusters, 2))

        #'''
        bug = False
        for i in range(self.n_clusters):
            if A[i]['nModels'] <= 0:
                bug = True
                break
        if bug:
            print('\nBUG!!!')
            print('f_left:%d, n_points:%d' % (self.f_left, self.n_points))
            for i in range(self.n_clusters):
                print('cluster %d: ' % i)
                print('ranks:', clusters_ranks[i])
                print('vector:', A[i]['vector'])
                print('nModels:', A[i]['nModels'])
                print('nOnes:', A[i]['nOnes'])
            raise Exception('Bug in combination.py, nModel <= 0')
        #'''

        for i in range(self.n_clusters):
            cache[i, 1] = float(np.sum(V[i]) + A[i]['nOnes']) / A[i]['nModels']
        
        for i in range(self.f_left):
            m = self.findMin(cache)
            combination[m] += 1
            minValue *= cache[m, 1] / cache[m, 0]

            V[m] = V[m] * A[m]['vector']
            cache[m, 0] = cache[m, 1]
            cache[m, 1] = float(np.sum(V[m]) + A[m]['nOnes']) / A[m]['nModels']

            if self.verbose:
                print( 'calc_combination:', *combination, sep=' | ', end=' |\r'),
                sys.stdout.flush()

        if self.verbose:
            print( 'calc_combination:', *combination, sep=' | ', end=' | \n')

        return np.array(combination)


if __name__ == '__main__':
    f_left = 100
    n_clusters = 2
    n_points = 12 
    clusters_ranks = np.array([ [6, 2, 3, 4, 5, 1], [7, 9, 8, 10, 11, 12] ])
    combination = Combination(f_left, 
                              n_clusters,
                              n_points,
                              clusters_ranks,

                              # Optional
                              model = 'uniform',
                              verbose = True
                              )
