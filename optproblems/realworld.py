"""
Some real-world problems that can be easily defined and quickly evaluated.

We do not really care about the meaningfulness of the actual application
here, but are rather interested in the landscapes of the generated objective
functions.

"""
import random
import math

import numpy as np
from diversipy.distance import calc_euclidean_dist_matrix

from optproblems.base import Problem, BoundConstraintsChecker
from optproblems.continuous import Himmelblau


def approx_gradient(position, obj_at_pos, function, step_size):
    """Approximate gradient with forward differences.

    Parameters
    ----------
    position : list
        The position where the gradient is approximated.
    obj_at_pos : float
        The objective value at this position (to save one evaluation).
    function : callable
        The objective function.
    step_size : float
        Step size for each dimension.

    Returns
    -------
    grad : list
        Approximated gradient.

    """
    dim = len(position)
    grad = [0.0] * dim
    for i in range(dim):
        offset_point = list(position)
        offset_point[i] += step_size
        quotient = (function(offset_point) - obj_at_pos) / step_size
        grad[i] = quotient
    return grad



def descent_step(position, obj_at_pos, function, grad_function, step_size):
    """Do a small step in the direction of steepest descent.

    Parameters
    ----------
    position : list
        The current incumbent solution.
    obj_at_pos : float
        The objective value at this position (to save one evaluation).
    function : callable
        The objective function.
    grad_function : callable
        A function returning the gradient.
    step_size : float
        Step size for each dimension.

    Returns
    -------
    proposed_point : list
        A point which is hopefully an improvement.
    norm : float
        The length of the gradient vector (useful as stopping criterion).

    """
    gradient = grad_function(position, obj_at_pos, function, step_size)
    norm = math.sqrt(math.fsum(g ** 2 for g in gradient))
    step_vector = [step_size * g / norm for g in gradient]
    proposed_point = [p - s for p, s in zip(position, step_vector)]
    return proposed_point, norm



def gradient_method(obj_function,
                    start_point,
                    init_step_size,
                    shrink_factor,
                    grad_function=approx_gradient,
                    step_function=descent_step,
                    max_iterations=float("inf"),
                    stop_norm=1e-8):
    """A simple gradient method.

    Parameters
    ----------
    obj_function : callable
        The objective function.
    start_point : list
        The starting point for the optimization.
    init_step_size : float
        Indicates how far we move into the descent direction.
    shrink_factor : float
        In case of an unsuccessful move, the step size is reduced by
        multiplying with this value.
    grad_function : callable, optional
        A function returning the gradient.
    step_function : callable, optional
        A function carrying out one optimization step.
    max_iterations : int, optional
        The maximal number of iterations to carry out.
    stop_norm : float, optional
        Optimization stops when the norm of the gradient goes below this
        value.

    Returns
    -------
    current_pos : list
        The current incumbent solution.
    current_obj : float
        The objective value of the current solution.

    """
    current_pos = start_point
    current_obj = obj_function(current_pos)
    step_size = init_step_size
    iteration = 0
    while iteration < max_iterations:
        try:
            new_pos, grad_norm = step_function(current_pos,
                                               current_obj,
                                               obj_function,
                                               grad_function,
                                               step_size)
        except ZeroDivisionError:
            # the norm of the gradient may become zero
            # use this as stopping criterion
            break
        if stop_norm is not None and grad_norm <= stop_norm:
            # stop when norm of gradient drops below threshold
            break
        if new_pos == current_pos:
            # stop because no new point is generated any more
            # (probably because step size is too small)
            break
        new_obj = obj_function(new_pos)
        obj_diff = current_obj - new_obj
        if obj_diff < 0:
            # the new point was not an improvement
            step_size *= shrink_factor
        else:
            current_pos = new_pos
            current_obj = new_obj
        iteration += 1
    return current_pos, current_obj



class GradientMethodConfiguration(Problem):
    """An algorithm configuration problem.

    This problem interprets the parameters of a simple gradient method as
    the search space of an optimization problem. If we fix the starting
    point, the resulting optimization problem is deterministic thanks to
    the deterministic behavior of the gradient method. Alternatively, we
    can generate a noisy problem by drawing a random starting point for each
    evaluation. This problem can either act as a single-objective problem
    (only objective value) or as a bi-objective problem (objective value and
    consumed function evaluations). Both quantities must be minimized. Also
    the search space dimension can be varied by selecting between one and
    four algorithm parameters to adjust.

    """
    def __init__(self, num_objectives=1,
                 noisy=False,
                 algo_max_it=100,
                 params_to_optimize=None,
                 internal_problem=None,
                 start_point=None,
                 phenome_preprocessor=None,
                 **kwargs):
        """Constructor.

        .. note:: If the starting point is not given, this constructor
            creates a randomly initialized problem instance by selecting
            the starting point randomly.

        Parameters
        ----------
        num_objectives : int, optional
            The number of objectives to consider. Must be one (default) or
            two. The first objective is the best objective value obtained by
            the algorithm, the second one is the number of function
            evaluations consumed.
        noisy : bool, optional
            If True, a new starting point is drawn for each algorithm run.
            Otherwise, the same starting point is used for all runs.
        algo_max_it : int, optional
            The maximal number of iterations to execute for the algorithm.
            This should be kept relatively low to keep the function
            evaluation of this problem cheap.
        params_to_optimize : list, optional
            This problem has at most four real-valued decision variables. By
            providing this argument, a subset and the order can be selected.
            The object provided here must have the format described in
            :func:`get_default_params <optproblems.realworld.GradientMethodConfiguration.get_default_params>`.
        internal_problem : optproblems.Problem, optional
            The problem the gradient method has to optimize. By default, the
            :class:`Himmelblau <optproblems.continuous.Himmelblau>` problem
            is chosen.
        start_point : list, optional
            The starting point for the gradient method. If none is provided,
            the starting point is drawn with the method
            :func:`create_start_point <optproblems.realworld.GradientMethodConfiguration.create_start_point>`.
            If ``noisy == True``, this argument is ignored.
        phenome_preprocessor : callable, optional
            A callable potentially applying transformations or checks to
            the phenome. Modifications should only be applied to a copy
            of the input. The (modified) phenome must be returned. When
            this pre-processing raises an exception, no function
            evaluations are counted. By default, no pre-processing is
            applied.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        assert num_objectives in (1, 2)
        self.noisy = noisy
        if algo_max_it is None:
            algo_max_it = float("inf")
        self.algo_max_it = algo_max_it
        if params_to_optimize is None:
            params_to_optimize = self.get_default_params()
        self.params_to_optimize = params_to_optimize
        self.num_variables = len(params_to_optimize)
        self.min_bounds = [params_to_optimize[i][1][0] for i in range(self.num_variables)]
        self.max_bounds = [params_to_optimize[i][1][1] for i in range(self.num_variables)]
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        Problem.__init__(self, self.obj_function,
                         num_objectives,
                         phenome_preprocessor=preprocessor,
                         **kwargs)
        if internal_problem is None:
            internal_problem = Himmelblau()
        if not isinstance(internal_problem, Problem):
            internal_problem = Problem(internal_problem)
        self.internal_problem = internal_problem
        if start_point is None:
            start_point = self.create_start_point()
        self.start_point = start_point


    @staticmethod
    def get_default_params():
        """Return the list of default decision variables.

        The returned list contains a tuple of the variable name and lower
        and upper bounds for each decision variable. The default setting is
        ``[("step_size", [0.0, None]),
        ("shrink_factor", [0.0, 1.0]),
        ("grad_step", [0.0, None]),
        ("stop_norm", [0.0, None])]``

        """
        default_params = [("step_size", [0.0, None]),
                          ("shrink_factor", [0.0, 1.0]),
                          ("grad_step", [0.0, None]),
                          ("stop_norm", [0.0, None])]
        return default_params


    def create_start_point(self):
        """Create a random starting point.

        This method draws points uniformly from :math:`[-6, 6]^n`. If a
        different internal problem is provided and this setting does not
        make sense, this function should be overridden.

        """
        dim = self.internal_problem.num_variables
        return [(random.random() - 0.5) * 12.0 for _ in range(dim)]


    def obj_function(self, phenome):
        """The objective function of this configuration problem.

        This method runs the
        :func:`gradient_method <optproblems.realworld.gradient_method>` on
        the chosen problem and returns some statistics on the run.

        Parameters
        ----------
        phenome : list
            The configuration setting for the gradient method.

        Returns
        -------
        opt_obj_value : float
            The best found objective value.
        evaluations : float, optional
            If two objectives were requested, also the number of consumed
            objective function evaluations is returned.

        """
        algo_max_it = self.algo_max_it
        # default values
        param_dict = {"step_size": 1.0, "shrink_factor": 0.5, "grad_step": 0.0, "stop_norm": 0}
        # find out which parameters are actively optimized
        for i, param_name, _ in enumerate(self.params_to_optimize):
            param_dict[param_name] = phenome[i]
        if param_dict["grad_step"] <= 0.0:
            grad_func = approx_gradient
        else:
            def fixed_step_approx_grad(position, obj_at_pos, function, step_size=None):
                return approx_gradient(position, obj_at_pos, function, param_dict["grad_step"])

            grad_func = fixed_step_approx_grad
        if self.noisy:
            # noise is introduced by drawing a random starting point
            self.start_point = self.create_start_point()
        # reset, so that evaluations are properly counted
        self.internal_problem.consumed_evaluations = 0
        _, opt_obj_value = gradient_method(self.internal_problem,
                                           self.start_point,
                                           param_dict["step_size"],
                                           param_dict["shrink_factor"],
                                           grad_function=grad_func,
                                           max_iterations=algo_max_it,
                                           stop_norm=param_dict["stop_norm"])
        if self.num_objectives == 2:
            evaluations = float(self.internal_problem.consumed_evaluations)
            return opt_obj_value, evaluations
        else:
            return opt_obj_value



class UniformityOptimization(Problem):
    """Optimize the uniformity of a point set in the unit hypercube.

    To make this problem interesting, it was given some unconventional
    properties: 1) existing points can be incorporated into the distance
    computations, to make several different instances possible, and 2) the
    objective function does (in general) not return a single scalar, but
    a list of sorted nearest neighbor distances. So, it can be treated as
    a multiobjective problem or, with lexicographic ordering of objective
    values, a single-objective problem. Random noise can be added by
    perturbing the points before computing distances.

    """
    def __init__(self, num_points=10,
                 dimension=2,
                 existing_points=None,
                 noise_strength=0.0,
                 dist_matrix_function=None,
                 phenome_preprocessor=None,
                 **kwargs):
        """Constructor.

        .. note:: If existing points are not given, this constructor
            creates a randomly initialized problem instance by drawing
            `num_points` randomly. (Note that in general, the number of
            existing points does not necessarily have to be equal to the
            number of optimized points.)

        Parameters
        ----------
        num_points : int, optional
            The number of points whose uniformity is to be optimized.
        dimension : int, optional
            The dimension of the unit cube.
        existing_points : list, optional
            A set of existing points influencing the optimized points. By
            default `num_points` are drawn randomly.
        noise_strength : float, optional
            If this value is larger than zero, points are perturbed by a
            multivariate Gaussian distribution with this standard deviation,
            before the distances are measured.
        dist_matrix_function : callable, optional
            A callable that returns a distance matrix for two sequences of
            points as input. Default is Euclidean distance.
        phenome_preprocessor : callable, optional
            A callable potentially applying transformations or checks to
            the phenome. Modifications should only be applied to a copy
            of the input. The (modified) phenome must be returned. When
            this pre-processing raises an exception, no function
            evaluations are counted. By default, no pre-processing is
            applied.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        assert num_points > 0
        assert dimension > 0
        assert noise_strength >= 0.0
        self.num_points = num_points
        self.dimension = dimension
        if existing_points is None:
            existing_points = self.create_existing_points()
        self.existing_points = np.asarray(existing_points)
        self.noise_strength = noise_strength
        if dist_matrix_function is None:
            dist_matrix_function = calc_euclidean_dist_matrix
        self.dist_matrix_function = dist_matrix_function
        self.num_variables = num_points * dimension
        self.min_bounds = [0.0] * self.num_variables
        self.max_bounds = [1.0] * self.num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        Problem.__init__(self, self.obj_function,
                         num_objectives=num_points,
                         phenome_preprocessor=preprocessor,
                         **kwargs)


    def create_existing_points(self):
        """Create random existing points in the unit hypercube."""
        dim = self.dimension
        existing_points = []
        for i in range(self.num_points):
            existing_points.append([random.random() for _ in range(dim)])
        return existing_points


    def obj_function(self, phenome):
        """The objective function of this problem.

        Parameters
        ----------
        phenome : list
            The concatenated list of points.

        Returns
        -------
        nn_dists : list
            A list of negated nearest-neighbor distances, sorted from
            absolute largest to absolute smallest. These can be
            interpreted as objective values by applying a lexicographic
            order.

        """
        dim = self.dimension
        num_points = self.num_points
        noise_strength = self.noise_strength
        assert dim * num_points == self.num_variables
        assert len(phenome) == self.num_variables
        if num_points < 2 and len(self.existing_points) == 0:
            return 0.0
        points = []
        for i in range(num_points):
            points.append(phenome[i*dim:(i+1)*dim])
        points = np.asarray(points)
        all_points = np.vstack((points, self.existing_points))
        if noise_strength > 0.0:
            shape = all_points.shape
            all_points += noise_strength * np.random.standard_normal(shape)
        dist_matrix = self.dist_matrix_function(points, all_points)
        for i in range(num_points):
            dist_matrix[i, i] = np.inf
        nn_dists = -dist_matrix.min(axis=1)
        return sorted(nn_dists.tolist())
