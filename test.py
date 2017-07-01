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
    scatter = kwargs.get( 'scatter', None )
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

    if scatter is not None:
        color = 'r'
        ax.scatter( scatter[:,0], scatter[:,1], color=color, s=10 )


    plt.savefig('%s/%s' % (fig_dir, fig_name))
    plt.close(fig)



def manhalanobis_distance( x, mean, cov ):
    assert len(x) == len(mean) == len(cov) == len(cov[0])
    xm = x - mean
    inverse_cov = np.linalg.inv(cov)
    return xm.dot(inverse_cov).dot(xm.T)

function_id = 8
dimension = 2
obj = CEC2005(dimension)[function_id].objective_function
xlim = [1.5 ,3.25]
ylim = [-2, -1]
positions = np.random.uniform([xlim[0], ylim[0]], [xlim[1], ylim[1]], size=(20,2))
fitnesses = np.array([ obj(x) for x in positions ])
draw( obj, 'test.png', xlim=xlim, ylim=ylim, scatter = positions )
from cluster import weighted_gaussian
mean, cov = weighted_gaussian( positions, fitnesses )
from scipy.stats import multivariate_normal
rv = multivariate_normal(mean, cov)
from cluster import clustering
labels = clustering(positions, fitnesses)
print('mean:', mean)
print('cov:\n', cov)
print(labels)


x, y = np.mgrid[xlim[0]:xlim[1]:((xlim[1]-xlim[0])/100), ylim[0]:ylim[1]:((ylim[1]-ylim[0])/100)] 
pos = np.empty(x.shape+(2,))
pos[:,:,0] = x
pos[:,:,1] = y
plt.contourf(x, y, rv.pdf(pos))
plt.savefig('contourf.png')
plt.close()


new_p = np.array([2.35, -1.6])
p1 = np.random.uniform([1.75, -1.7], [2.05,-1.4], size=(10,2))
p2 = np.random.uniform([2.75, -1.7], [3.05,-1.4], size=(10,2))
positions = np.concatenate((p1, p2))
print(positions.shape)
#positions[0] = np.random.uniform([xlim[0], ylim[0]], [xlim[1], ylim[1]], size=(1,2))
draw( obj, 'test1.png', xlim=xlim, ylim=ylim, scatter = positions )
fitnesses = np.array([ obj(x) for x in positions ])
for pos in positions:
    print(rv.pdf(pos), rv.logpdf(pos))
new_mean, new_cov = weighted_gaussian( positions, fitnesses )
print('mean:', new_mean)
print('cov:\n', new_cov)
labels = clustering(positions, fitnesses)
print(labels)

from scipy import stats
wilks_statistics = len(positions) * manhalanobis_distance( new_mean, mean, cov )
dof = dimension
p_value = stats.chisqprob(wilks_statistics, dof)
print(p_value)

wilks_statistics = manhalanobis_distance( new_p, mean, cov )
dof = dimension
p_value = stats.chisqprob(wilks_statistics, dof)
print(p_value)
#fig = plt.figure(figsize=plt.figaspect(1))
#ax = fig.add_subplot(111, aspect=0.5)

cm = mean - new_mean
inverse_cov = np.linalg.inv( 0.5*(cov + new_cov) )
z = cm.dot(inverse_cov).dot(cm.T)
p_value = stats.chisqprob(z, dof)
print(p_value)

rv = multivariate_normal(new_mean, new_cov)
x, y = np.mgrid[xlim[0]:xlim[1]:((xlim[1]-xlim[0])/100), ylim[0]:ylim[1]:((ylim[1]-ylim[0])/100)] 
pos = np.empty(x.shape+(2,))
pos[:,:,0] = x
pos[:,:,1] = y
plt.contourf(x, y, rv.pdf(pos))
plt.savefig('contourf1.png')
plt.close()


'''
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
yerr = [2*std] * len(errors)
ax.errorbar(x, new_Pr, yerr=yerr, fmt='o') 

x = np.linspace(stats.beta.ppf(0, a, b),
                stats.beta.ppf(1, a, b), 100)
kernel = m*norm.pdf(x) + c
ax.plot(x, kernel, 'b-', label='linear transformed normal pdf') 


ax.legend()
plt.savefig('test.png')
plt.close(fig)



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
