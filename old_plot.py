# Source Generated with Decompyle++
# File: plot.cpython-34.pyc (Python 3.4)

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from operator import itemgetter
from sklearn.cluster import KMeans
from optproblems.cec2005 import CEC2005
from optproblems import Individual, Problem
from cluster import Cluster
from boundary import Boundary

def draw_surface3d(function_id, **kwargs):
    dim = 2
    func = CEC2005(dim)[function_id]
    clusters = kwargs.get('clusters', [])
    angle = kwargs.get('angle', 240)
    rotate = kwargs.get('rotate', False)
    fig_name = kwargs.get('fig_name', None)
    boundary = Boundary(func)
    step = (boundary.max_bounds[0] - boundary.min_bounds[0]) / 100
    X = np.arange(boundary.min_bounds[0], boundary.max_bounds[0] + step, step)
    Y = np.arange(boundary.min_bounds[1], boundary.max_bounds[1] + step, step)
    (X, Y) = np.meshgrid(X, Y)
    positions = (lambda .0: continue[ [
x,
y] for (x, y) in .0 ])(zip(X.ravel(), Y.ravel()))
    solutions = (lambda .0: continue[ Individual(position) for position in .0 ])(positions)
    problem = Problem(func.objective_function)
    problem.batch_evaluate(solutions)
    Z = np.array((lambda .0: continue[ solution.objective_values for solution in .0 ])(solutions))
    Z = Z.reshape(X.shape)
    fig = plt.figure(figsize = plt.figaspect(0.5))
    fig.suptitle(str(func))
    ax = fig.add_subplot(1, 2, 1)
    cset = ax.contourf(X, Y, Z, cmap = cm.coolwarm)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(clusters))))
    for cluster in clusters:
        color = next(colors)
        x = np.array((lambda .0: continue[ individual.phenome[0] for individual in .0 ])(cluster.population))
        y = np.array((lambda .0: continue[ individual.phenome[1] for individual in .0 ])(cluster.population))
        ax.scatter(x, y, color = color, marker = 'o', s = 10)
        x_border = (lambda .0: continue[ vertice[0] for vertice in .0 ])(cluster.border)
        x_border.append(x_border[0])
        y_border = (lambda .0: continue[ vertice[1] for vertice in .0 ])(cluster.border)
        y_border.append(y_border[0])
        ax.plot(x_border, y_border, color = color)
    
    optimal_pos = func.get_optimal_solutions()[0].phenome
    ax.scatter(optimal_pos[0], optimal_pos[1], color = 'w', marker = 'x', s = 100)
    ax = fig.add_subplot(1, 2, 2, projection = '3d')
    surf = ax.plot_surface(X, Y, Z, alpha = 0.8, zorder = -1, cmap = cm.coolwarm, linewidth = 0, edgecolors = 'k', antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    cset = ax.contourf(X, Y, Z, zdir = 'z', zorder = -2, offset = np.amin(Z), cmap = cm.coolwarm)
    if rotate:
        for ang in range(angle, angle + 360, 10):
            ax.view_init(30, ang)
            plt.draw()
            plt.pause(0.001)
        
    else:
        ax.view_init(30, angle)
        plt.show()
        if fig_name is not None:
            plt.savefig(fig_name)
        input('Press Enter to continue...')


def draw_cluster(function_id, clusters, k, **kwargs):
    dim = 2
    func = CEC2005(dim)[function_id]
    angle = kwargs.get('angle', 240)
    fig_name = kwargs.get('fig_name', None)
    cluster = clusters[k]
    color = cm.rainbow(np.linspace(0, 1, len(clusters)))[k]
    X = np.arange(0, 1.01, 0.01)
    Y = np.arange(0, 1.01, 0.01)
    (X, Y) = np.meshgrid(X, Y)
    positions = (lambda .0: continue[ [
x,
y] for (x, y) in .0 ])(zip(X.ravel(), Y.ravel()))
    original_positions = cluster.transform_inverse(positions)
    solutions = (lambda .0: continue[ Individual(position) for position in .0 ])(original_positions)
    problem = Problem(func.objective_function)
    problem.batch_evaluate(solutions)
    Z = np.array((lambda .0: continue[ solution.objective_values for solution in .0 ])(solutions))
    Z = Z.reshape(X.shape)
    fig = plt.figure(figsize = plt.figaspect(0.5))
    fig.suptitle(str(func))
    ax = fig.add_subplot(1, 2, 1)
    cset = ax.contourf(X, Y, Z, cmap = cm.coolwarm)
    positions = np.array((lambda .0: continue[ p.phenome for p in .0 ])(cluster.population))
    transformed_positions = cluster.transform(positions)
    x = transformed_positions.T[0]
    y = transformed_positions.T[1]
    ax.scatter(x, y, color = color, marker = 'o', s = 10)
    cord = np.array([
        [
            0,
            0],
        [
            1,
            0],
        [
            1,
            1],
        [
            0,
            1]])
    ax.plot(cord[([
        0,
        1,
        2,
        3,
        0], 0)], cord[([
        0,
        1,
        2,
        3,
        0], 1)], color = color)
    ax = fig.add_subplot(1, 2, 2, projection = '3d')
    surf = ax.plot_surface(X, Y, Z, alpha = 0.8, zorder = -1, cmap = cm.coolwarm, linewidth = 0, edgecolors = 'k', antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    cset = ax.contourf(X, Y, Z, zdir = 'z', zorder = -2, offset = np.amin(Z), cmap = cm.coolwarm)
    ax.view_init(30, angle)
    plt.show()
    if fig_name is not None:
        plt.savefig(fig_name)
    input('Press Enter to continue...')


def draw_all(function_id, **kwargs):
    dim = 2
    func = CEC2005(dim)[function_id]
    clusters = kwargs.get('clusters', [])
    angle = kwargs.get('angle', 240)
    rotate = kwargs.get('rotate', False)
    fig_name = kwargs.get('fig_name', None)
    cmap = cm.coolwarm
    scatter_cmap = cm.jet(np.linspace(0.1, 0.9, len(clusters)))
    boundary = Boundary(func)
    step = (boundary.max_bounds[0] - boundary.min_bounds[0]) / 100
    X = np.arange(boundary.min_bounds[0], boundary.max_bounds[0] + step, step)
    Y = np.arange(boundary.min_bounds[1], boundary.max_bounds[1] + step, step)
    (X, Y) = np.meshgrid(X, Y)
    positions = (lambda .0: continue[ [
x,
y] for (x, y) in .0 ])(zip(X.ravel(), Y.ravel()))
    solutions = (lambda .0: continue[ Individual(position) for position in .0 ])(positions)
    problem = Problem(func.objective_function)
    problem.batch_evaluate(solutions)
    Z = np.array((lambda .0: continue[ solution.objective_values for solution in .0 ])(solutions))
    Z = Z.reshape(X.shape)
    fig_w = len(clusters) + 1
    fig_h = 2
    fig = plt.figure(figsize = plt.figaspect(float(fig_h) / fig_w))
    fig.suptitle(str(func))
    ax = fig.add_subplot(fig_h, fig_w, 1, projection = '3d')
    surf = ax.plot_surface(X, Y, Z, alpha = 0.8, zorder = -1, cmap = cmap, linewidth = 0, edgecolors = 'k', antialiased = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    cset = ax.contourf(X, Y, Z, zdir = 'z', zorder = -2, offset = np.amin(Z), cmap = cmap)
    ax.view_init(30, angle)
    ax = fig.add_subplot(fig_h, fig_w, fig_w + 1)
    cset = ax.contourf(X, Y, Z, cmap = cmap)
    colors = iter(scatter_cmap)
    for cluster in clusters:
        color = next(colors)
        x = np.array((lambda .0: continue[ individual.phenome[0] for individual in .0 ])(cluster.population))
        y = np.array((lambda .0: continue[ individual.phenome[1] for individual in .0 ])(cluster.population))
        ax.scatter(x, y, color = color, marker = 'o', s = 10)
        x_border = (lambda .0: continue[ vertice[0] for vertice in .0 ])(cluster.border)
        x_border.append(x_border[0])
        y_border = (lambda .0: continue[ vertice[1] for vertice in .0 ])(cluster.border)
        y_border.append(y_border[0])
        ax.plot(x_border, y_border, color = color)
    
    optimal_pos = func.get_optimal_solutions()[0].phenome
    ax.scatter(optimal_pos[0], optimal_pos[1], color = 'w', marker = 'x', s = 100)
    for k in range(len(clusters)):
        cluster = clusters[k]
        color = scatter_cmap[k]
        X = np.arange(0, 1.01, 0.01)
        Y = np.arange(0, 1.01, 0.01)
        (X, Y) = np.meshgrid(X, Y)
        positions = (lambda .0: continue[ [
x,
y] for (x, y) in .0 ])(zip(X.ravel(), Y.ravel()))
        original_positions = cluster.transform_inverse(positions)
        solutions = (lambda .0: continue[ Individual(position) for position in .0 ])(original_positions)
        problem = Problem(func.objective_function)
        problem.batch_evaluate(solutions)
        Z = np.array((lambda .0: continue[ solution.objective_values for solution in .0 ])(solutions))
        Z = Z.reshape(X.shape)
        ax = fig.add_subplot(fig_h, fig_w, k + 2, projection = '3d')
        surf = ax.plot_surface(X, Y, Z, alpha = 0.8, zorder = -1, cmap = cmap, linewidth = 0, edgecolors = 'k', antialiased = False)
        fig.colorbar(surf, shrink = 0.5, aspect = 5)
        cset = ax.contourf(X, Y, Z, zdir = 'z', zorder = -2, offset = np.amin(Z), cmap = cmap)
        ax.view_init(30, angle)
        ax = fig.add_subplot(fig_h, fig_w, fig_w + k + 2)
        cset = ax.contourf(X, Y, Z, cmap = cmap)
        positions = np.array((lambda .0: continue[ p.phenome for p in .0 ])(cluster.population))
        transformed_positions = cluster.transform(positions)
        x = transformed_positions.T[0]
        y = transformed_positions.T[1]
        ax.scatter(x, y, color = color, marker = 'o', s = 10)
        cord = np.array([
            [
                0,
                0],
            [
                1,
                0],
            [
                1,
                1],
            [
                0,
                1]])
        ax.plot(cord[([
            0,
            1,
            2,
            3,
            0], 0)], cord[([
            0,
            1,
            2,
            3,
            0], 1)], color = color)
    
    fig.tight_layout()
    plt.show()
    if fig_name is not None:
        plt.savefig(fig_name)
    input('Press Enter to continue...')


def draw_contour(function_id, **kwargs):
    dim = 2
    func = CEC2005(dim)[function_id]
    clusters = kwargs.get('clusters', [])
    angle = kwargs.get('angle', 240)
    rotate = kwargs.get('rotate', False)
    fig_name = kwargs.get('fig_name', None)
    fig_title = kwargs.get('fig_title', str(func))
    cmap = cm.coolwarm
    scatter_cmap = cm.jet(np.linspace(0.1, 0.9, len(clusters)))
    boundary = Boundary(dim, function_id)
    step = (boundary.max_bounds[0] - boundary.min_bounds[0]) / 100
    X = np.arange(boundary.min_bounds[0], boundary.max_bounds[0] + step, step)
    Y = np.arange(boundary.min_bounds[1], boundary.max_bounds[1] + step, step)
    (X, Y) = np.meshgrid(X, Y)
    positions = (lambda .0: continue[ [
x,
y] for (x, y) in .0 ])(zip(X.ravel(), Y.ravel()))
    solutions = (lambda .0: continue[ Individual(position) for position in .0 ])(positions)
    problem = Problem(func.objective_function)
    problem.batch_evaluate(solutions)
    Z = np.array((lambda .0: continue[ solution.objective_values for solution in .0 ])(solutions))
    vmin = min(Z)
    vmax = max(Z)
    vmin = vmin - (vmax - vmin) * 0.2
    vmax = vmax + (vmax - vmin) * 0.2
    Z = Z.reshape(X.shape)
    inch_size = 4
    fig_w = len(clusters) + 1
    fig_h = 1
    fig = plt.figure(figsize = (fig_w * inch_size, fig_h * inch_size))
    ax = fig.add_subplot(fig_h, fig_w, 1)
    ax.set_xlim([
        boundary.min_bounds[0],
        boundary.max_bounds[0]])
    ax.set_ylim([
        boundary.min_bounds[1],
        boundary.max_bounds[1]])
    cset = ax.contourf(X, Y, Z, cmap = cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(cset, aspect = 20)
    colors = iter(scatter_cmap)
    for cluster in clusters:
        color = next(colors)
        x = np.array((lambda .0: continue[ individual.phenome[0] for individual in .0 ])(cluster.population))
        y = np.array((lambda .0: continue[ individual.phenome[1] for individual in .0 ])(cluster.population))
        ax.scatter(x, y, color = color, marker = 'o', s = 10)
        border = cluster.transform_inverse(np.array([
            [
                0,
                0],
            [
                1,
                0],
            [
                1,
                1],
            [
                0,
                1],
            [
                0,
                0]]))
        ax.plot(border[(:, 0)], border[(:, 1)], color = color)
    
    optimal_pos = func.get_optimal_solutions()[0].phenome
    ax.scatter(optimal_pos[0], optimal_pos[1], color = 'w', marker = 'x', s = 100)
    for k in range(len(clusters)):
        cluster = clusters[k]
        color = scatter_cmap[k]
        X = np.arange(0, 1.01, 0.01)
        Y = np.arange(0, 1.01, 0.01)
        (X, Y) = np.meshgrid(X, Y)
        positions = (lambda .0: continue[ [
x,
y] for (x, y) in .0 ])(zip(X.ravel(), Y.ravel()))
        original_positions = cluster.transform_inverse(positions)
        solutions = (lambda .0: continue[ Individual(position) for position in .0 ])(original_positions)
        problem = Problem(func.objective_function)
        problem.batch_evaluate(solutions)
        Z = np.array((lambda .0: continue[ solution.objective_values for solution in .0 ])(solutions))
        Z = Z.reshape(X.shape)
        ax = fig.add_subplot(fig_h, fig_w, k + 2)
        ax.set_xlim([
            -0.01,
            1.01])
        ax.set_ylim([
            -0.01,
            1.01])
        cset = ax.contourf(X, Y, Z, cmap = cmap, vmin = vmin, vmax = vmax)
        positions = np.array((lambda .0: continue[ p.phenome for p in .0 ])(cluster.population))
        transformed_positions = cluster.transform(positions)
        x = transformed_positions.T[0]
        y = transformed_positions.T[1]
        ax.scatter(x, y, color = color, marker = 'o', s = 10)
        cord = np.array([
            [
                0,
                0],
            [
                1,
                0],
            [
                1,
                1],
            [
                0,
                1]])
        ax.plot(cord[([
            0,
            1,
            2,
            3,
            0], 0)], cord[([
            0,
            1,
            2,
            3,
            0], 1)], color = color)
    
    fig.tight_layout()
    st = fig.suptitle(fig_title, fontsize = 16)
    st.set_y(0.95)
    fig.subplots_adjust(top = 0.85)
    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()
        input('Press Enter to continue...')
    plt.close(fig)


def draw_original_contour(function_id, **kwargs):
    dim = 2
    func = CEC2005(dim)[function_id]
    population = kwargs.get('population', [])
    angle = kwargs.get('angle', 240)
    rotate = kwargs.get('rotate', False)
    fig_name = kwargs.get('fig_name', None)
    fig_title = kwargs.get('fig_title', str(func))
    cmap = cm.coolwarm
    scatter_cmap = cm.jet(np.linspace(0.1, 0.9, len(population)))
    boundary = Boundary(dim, function_id)
    step = (boundary.max_bounds[0] - boundary.min_bounds[0]) / 100
    X = np.arange(boundary.min_bounds[0], boundary.max_bounds[0] + step, step)
    Y = np.arange(boundary.min_bounds[1], boundary.max_bounds[1] + step, step)
    (X, Y) = np.meshgrid(X, Y)
    positions = (lambda .0: continue[ [
x,
y] for (x, y) in .0 ])(zip(X.ravel(), Y.ravel()))
    solutions = (lambda .0: continue[ Individual(position) for position in .0 ])(positions)
    problem = Problem(func.objective_function)
    problem.batch_evaluate(solutions)
    Z = np.array((lambda .0: continue[ solution.objective_values for solution in .0 ])(solutions))
    vmin = min(Z)
    vmax = max(Z)
    vmin = vmin - (vmax - vmin) * 0.2
    vmax = vmax + (vmax - vmin) * 0.2
    Z = Z.reshape(X.shape)
    inch_size = 4
    fig_w = 1
    fig_h = 1
    fig = plt.figure(figsize = (fig_w * inch_size, fig_h * inch_size))
    ax = fig.add_subplot(fig_h, fig_w, 1)
    ax.set_xlim([
        boundary.min_bounds[0],
        boundary.max_bounds[0]])
    ax.set_ylim([
        boundary.min_bounds[1],
        boundary.max_bounds[1]])
    cset = ax.contourf(X, Y, Z, cmap = cmap, vmin = vmin, vmax = vmax)
    fig.colorbar(cset, aspect = 20)
    colors = iter(scatter_cmap)
    color = next(colors)
    x = np.array((lambda .0: continue[ individual.phenome[0] for individual in .0 ])(population))
    y = np.array((lambda .0: continue[ individual.phenome[1] for individual in .0 ])(population))
    ax.scatter(x, y, color = color, marker = 'o', s = 10)
    for opt in func.get_optimal_solutions():
        optimal_pos = opt.phenome
        ax.scatter(optimal_pos[0], optimal_pos[1], color = 'w', marker = 'x', s = 100)
    
    fig.tight_layout()
    st = fig.suptitle(fig_title, fontsize = 16)
    st.set_y(0.95)
    fig.subplots_adjust(top = 0.85)
    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()
        input('Press Enter to continue...')
    plt.close(fig)


def draw_arms(function_id, arms, **kwargs):

    # Parameters
    dim = 2
    function = CEC2005(dim)[function_id].objective_function

    k = len(arms)
    inch_size = 4
    fig_w = k + 1
    fig_h = 1
    fig = plt.figure(figsize = (fig_w * inch_size, fig_h * inch_size))
    angle = kwargs.get('angle', 240)
    rotate = kwargs.get('rotate', False)
    fig_name = kwargs.get('fig_name', None)
    fig_title = kwargs.get('fig_title', str(func))
    cmap = cm.coolwarm
    scatter_cmap = cm.jet(np.linspace(0.1, 0.9, k))
    boundary = Boundary(dim, function_id)


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
    optimal_pos = func.get_optimal_solutions()[0].phenome
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
        ax.scatter(trans_X[(:, 0)], trans_X[(:, 1)], color = color, marker = 'o', s = 10)

        # Plot border
        cord = np.array([ [0, 0], [1, 0], [1, 1], [0, 1]])
        ax.plot(cord[[0, 1, 2, 3, 0], 0], cord[[0, 1, 2, 3, 0], 1], color = color)
    



    fig.tight_layout()
    st = fig.suptitle(fig_title, fontsize = 16)
    st.set_y(0.95)
    fig.subplots_adjust(top = 0.85)
    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()
        input('Press Enter to continue...')
    plt.close(fig)


if __name__ == '__main__':
    nPoints = 40
    dim = 2
    function_id = 8
    n_clusters = 3
    func = CEC2005(dim)[function_id]
    positions = np.zeros((nPoints, dim))
    boundary = Boundary(dim, function_id)
    for d in range(dim):
        positions[(:, d)] = np.random.uniform(boundary.min_bounds[d], boundary.max_bounds[d], nPoints)
    
    population = (lambda .0: continue[ Individual(position) for position in .0 ])(positions)
    problem = Problem(func.objective_function)
    problem.batch_evaluate(population)
    population = sorted(population, key = (lambda p: p.objective_values))
    population = population[:len(population) // 2]
    X = np.array((lambda .0: continue[ individual.phenome for individual in .0 ])(population))
    labels = KMeans(n_clusters = n_clusters).fit_predict(X)
    clusters = []
    for k in range(n_clusters):
        cluster_population = []
        for i in range(len(population)):
            if labels[i] == k:
                cluster_population.append(population[i])
                continue
        clusters.append(Cluster(cluster_population, boundary))
    
    draw_contour(function_id, clusters = clusters)
