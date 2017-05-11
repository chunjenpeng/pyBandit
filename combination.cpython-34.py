# Source Generated with Decompyle++
# File: combination.cpython-34.pyc (Python 3.4)

import sys
import numpy as np

class Combination:
    __qualname__ = 'Combination'
    
    def __init__(self, f_left, n_clusters, n_points, clusters_ranks, **kwargs):
        self.f_left = f_left
        self.n_clusters = n_clusters
        self.n_points = n_points
        self.model = kwargs.get('model', 'linear')
        self.debug = kwargs.get('debug', False)
        self.combination = self.calc_combination(clusters_ranks)

    
    def CDF(self, x):
        if self.model == 'quadratic':
            return 1 - x * x / x * (x + 1) * (2 * x + 1) / 6
        if None.model == 'linear':
            return 1 - x / x * (x + 1) / 2
        return (None - 1) / x

    
    def buildArm(self, ranks):
        Arm = { }
        Min = self.f_left + ranks[-1]
        Max = 2 * self.f_left + self.n_points
        Arm['vector'] = (np.array,)((lambda .0: continue[ self.CDF(float(x)) for x in .0 ])(range(Min, Max + 1)))
        Arm['nModels'] = float(self.f_left + (self.n_points + 1 - ranks[-1])) * (self.f_left + ranks[0])
        Arm['nOnes'] = Arm['nModels'] - Max + 1 - Min
        return Arm

    
    def findMin(self, cache):
        base = np.prod(cache, axis = 0)[0]
        value = base * cache[(:, 1)] / cache[(:, 0)]
        return np.argmin(value)

    
    def calc_combination(self, clusters_ranks):
        minValue = 1
        combination = [
            0] * self.n_clusters
        A = (lambda .0: continue[ self.buildArm(ranks) for ranks in .0 ])(clusters_ranks)
        V = (lambda .0: continue[ A[i]['vector'] for i in .0 ])(range(self.n_clusters))
        cache = np.ones((self.n_clusters, 2))
        bug = False
        for i in range(self.n_clusters):
            if A[i]['nModels'] <= 0:
                bug = True
                break
                continue
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
        for i in range(self.n_clusters):
            cache[(i, 1)] = float(np.sum(V[i]) + A[i]['nOnes']) / A[i]['nModels']
        
        for i in range(self.f_left):
            m = self.findMin(cache)
            combination[m] += 1
            minValue *= cache[(m, 1)] / cache[(m, 0)]
            V[m] = V[m] * A[m]['vector']
            cache[(m, 0)] = cache[(m, 1)]
            cache[(m, 1)] = float(np.sum(V[m]) + A[m]['nOnes']) / A[m]['nModels']
            if self.debug:
                sys.stdout.write('\rcalc_combination: [%s]' % ','.join((lambda .0: continue)(combination)))
                sys.stdout.flush()
                continue
        if self.debug:
            print('')
        return combination


if __name__ == '__main__':
    f_left = 20000
    n_clusters = 2
    n_points = 3
    clusters_ranks = [
        [
            1,
            3],
        [
            2]]
    combination = Combination(f_left, n_clusters, n_points, clusters_ranks, model = 'uniform', debug = True)
