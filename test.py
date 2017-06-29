from itertools import combinations
from operator import itemgetter
from scipy.stats import rankdata, kstest, shapiro, ks_2samp, norm, shapiro, normaltest, multivariate_normal
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from optproblems.cec2005 import CEC2005
import matplotlib.pyplot as plt
import numpy as np

def draw( obj, fig_name, **kwargs ):

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
    #scatter_cmap = cm.jet( np.linspace(0.1, 0.9, len(clusters_positions)) )
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

    plt.savefig('%s/%s' % (fig_dir, fig_name))
    plt.close(fig)

def manhalanobis_distance( x, mean, cov ):
    assert len(x) == len(mean) == len(cov) == len(cov[0])
    xm = x - mean
    inverse_cov = np.linalg.inv(cov)
    return xm.dot(inverse_cov).dot(xm.T)


obj = CEC2005(2)[8].objective_function
xlim = [1.5 ,2.5]
ylim = [-2, -1]
positions = np.random.uniform([xlim[0], ylim[0]], [xlim[1], ylim[1]], size=(20,2))
#draw( obj, 'test.png', xlim=xlim, ylim=ylim )

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(111, aspect=0.5)

from scipy import stats

a, b = 2, 2
x = np.linspace(stats.beta.ppf(0, a, b),
                stats.beta.ppf(1, a, b), 100)

norm = stats.norm(0.5, 0.2)
#ax.plot(x, norm.pdf(x), 'k-', label='normal pdf') 

beta = stats.beta(a,b)
ax.plot(x, beta.pdf(x), 'r-', label='beta pdf') 

x = np.random.uniform(0,1,10)
Pr = norm.pdf(x)
F = beta.pdf(x)
#ax.scatter(x, Pr) 

A = np.vstack([Pr, np.ones(len(Pr))]).T
m, c = np.linalg.lstsq(A, F)[0]
print(m, c)

new_Pr = m*Pr + c

errors = F - new_Pr
print(errors)
std = np.std(errors)
yerr = [3*std] * len(errors)
ax.errorbar(x, new_Pr, yerr=yerr, fmt='o') 

x = np.linspace(stats.beta.ppf(0, a, b),
                stats.beta.ppf(1, a, b), 100)
kernel = m*norm.pdf(x) + c
ax.plot(x, kernel, 'b-', label='linear transformed normal pdf') 


ax.legend()
plt.savefig('test.png')
plt.close(fig)



'''
fitnesses = np.array([ obj(x) for x in x1 ])

for x, f in zip(x1, fitnesses):
    print(x,f)

from cluster import weighted_gaussian
mean, cov = weighted_gaussian(x1, fitnesses)
rv = multivariate_normal( mean, cov )
print(rv.pdf(mean))

ds = [ manhalanobis_distance(x, mean, cov) for x in x1 ]


print('\n')
indices = np.argsort(ds)
for i in indices:
    print(x1[i], fitnesses[i], ds[i]) 

import matplotlib.pyplot as plt

plt.scatter(x1[:,0], x1[:,1])
plt.savefig('test.png')

counts, bin_edges = np.histogram(

from cluster import weighted_gaussian
mean, cov = weighted_gaussian(x1, fitnesses)
rv = multivariate_normal( mean, cov )

print(fitnesses)
ranks = rankdata(fitnesses, method='ordinal')
print(ranks)
weights = np.log(len(fitnesses)+0.5) - np.log(ranks)
weights = weights / sum(weights)
#print(weights)
for x, f, w in zip(x1, fitnesses, weights):
    pr = rv.logpdf(x)
    print(f, w, pr)

weights = np.log(len(fitnesses)+0.5) - np.log(ranks)
weights = weights / min(weights)
xs = []
#print(weights)
for x, r in zip(x1, ranks):
    #print(x, r)
    xs.extend( [x,]* (len(x1)-r) )
xs = np.array(xs)
#print(xs)
#print(shapiro(xs)) 



df_pair = [ (manhalanobis_distance(x, mean, cov), f(x)) for x in x1 ]
df_pair = [ manhalanobis_distance(x, mean, cov) for x in x1 ]
#print(df_pair)
for x in x1:
    print(manhalanobis_distance(x, mean, cov)) 

for x, f in zip(x1, fitnesses):
    print( manhalanobis_distance(x, mean, cov) )
    print(x, f)
'''

#print(normaltest(x1))
'''
print(x1[:,0])
print(ks_2samp(x1[:,0], x2[:,0]))
input()

for f, x in zip(fs, xs):
    print(f, x)
print()

for d in range(2):
    indices = np.argsort(xs[:,d])
    for i in indices: 
        print(fs[i], xs[i])
    norm_fs = ( fs[indices] - min(fs) ) * (1/(max(fs) - min(fs)))
    print(fs[indices])
    print(kstest(fs[indices], 'norm'))
    print(shapiro(fs[indices]))
    print(norm_fs)
    print(kstest(norm_fs, 'norm'))
    print(shapiro(norm_fs))
    print()
'''
