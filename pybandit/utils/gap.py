# gap.py
# (c) 2013 Mikael Vejdemo-Johansson
# BSD License
#
# SciPy function to compute the gap statistic for evaluating k-means clustering.
# Gap statistic defined in
# Tibshirani, Walther, Hastie:
#  Estimating the number of clusters in a data set via the gap statistic
#  J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423
import warnings
import numpy as np
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
from sklearn.cluster import KMeans
dst = scipy.spatial.distance.euclidean

def gap(data, refs=None, nrefs=20, max_num_clusters=5):
    """
    Compute the Gap statistic for an nxm dataset in data.
    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.
    Give the list of k-values for which you want to compute the statistic in ks.
    """
    ks = range(1, max_num_clusters+1)
    shape = data.shape
    if refs==None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))
    

        rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs

    gaps = scipy.zeros((len(ks),))
    stds = scipy.zeros((len(ks),))
    for (i,k) in enumerate(ks):

        if k == 1:
            kmc = np.mean(data, axis=0).reshape(shape[1],1)
            kml = np.zeros(len(data), dtype=np.int32)
        else:
            kmeans = KMeans(n_clusters=k).fit(data)
            kmc = kmeans.cluster_centers_
            kml = kmeans.labels_

        #(kmc,kml) = scipy.cluster.vq.kmeans2(data, k)
        #print('kmc\n',kmc1)
        #print('kml\n',kml1)
        #input('continue...')


        disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])

        refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                (kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k)


            refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
            

        gaps[i] = scipy.mean(scipy.log(refdisps))-scipy.log(disp)

        l_mean = scipy.mean(scipy.log(refdisps))
        sd = np.sqrt( scipy.mean( np.power( (scipy.log(refdisps) - l_mean), 2) ) )
        stds[i] = sd * np.sqrt( 1.0 + 1.0/rands.shape[2] ) 

    delta = [ stds[i] - stds[i+1] for i in range( len(ks) - 1 ) ]

    first_std = max_num_clusters
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i+1] - stds[i+1]:
            first_std = i+1
            break

    #num_clusters = np.argmax(delta) + 1
    num_clusters = np.argmax(gaps) + 1
    #num_clusters = first_std 
    print('gap max delta:', np.argmax(delta)+1)
    print('gap max gap  :', np.argmax(gaps)+1)
    print('gap first std:', first_std)


    return num_clusters



def test():
    data = np.random.uniform( size=(100,2) ) 
    #data = np.random.normal( size=(100,2) ) 
    print(data)
    print(gap(data, ks = range(1,8)))

if __name__ == '__main__':
    test()
