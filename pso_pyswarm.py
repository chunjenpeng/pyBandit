from functools import partial
import numpy as np


class PSO:

    def _obj_wrapper(self, func, args, kwargs, x):
        return func(x, *args, **kwargs)

    def _is_feasible_wrapper(self, func, x):
        return np.all(func(x)>=0)

    def _cons_none_wrapper(self, x):
        return np.array([0])

    def _cons_ieqcons_wrapper(self, ieqcons, args, kwargs, x):
        return np.array([y(x, *args, **kwargs) for y in ieqcons])

    def _cons_f_ieqcons_wrapper(self, f_ieqcons, args, kwargs, x):
        return np.array(f_ieqcons(x, *args, **kwargs))
    
    def __init__(self, func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
                swarmsize=12, omega=0.5, phip=0.5, phig=0.5, maxiter=100000, 
                minstep=1e-12, minfunc=1e-12, debug=False, processes=1,
                particle_output=False):
        """
        Perform a particle swarm optimization (PSO)
       
        Parameters
        ==========
        func : function
            The function to be minimized
        lb : array
            The lower bounds of the design variable(s)
        ub : array
            The upper bounds of the design variable(s)
       
        Optional
        ========
        ieqcons : list
            A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
            a successfully optimized problem (Default: [])
        f_ieqcons : function
            Returns a 1-D array in which each element must be greater or equal 
            to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
            ieqcons is ignored (Default: None)
        args : tuple
            Additional arguments passed to objective and constraint functions
            (Default: empty tuple)
        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions (Default: empty dict)
        swarmsize : int
            The number of particles in the swarm (Default: 100)
        omega : scalar
            Particle velocity scaling factor (Default: 0.5)
        phip : scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        phig : scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        maxiter : int
            The maximum number of iterations for the swarm to search (Default: 100)
        minstep : scalar
            The minimum stepsize of swarm's best position before the search
            terminates (Default: 1e-8)
        minfunc : scalar
            The minimum change of swarm's best objective value before the search
            terminates (Default: 1e-8)
        debug : boolean
            If True, progress statements will be displayed every iteration
            (Default: False)
        processes : int
            The number of processes to use to evaluate objective function and 
            constraints (default: 1)
        particle_output : boolean
            Whether to include the best per-particle position and the objective
            values at those.
       
        Returns
        =======
        g : array
            The swarm's best known position (optimal design)
        f : scalar
            The objective value at ``g``
        p : array
            The best known position per particle
        pf: arrray
            The objective values at each position in p
       
        """
       
        assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'

        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.maxiter = maxiter
        self.minstep = minstep
        self.minfunc = minfunc
        self.debug = debug
        self.processes = processes
        self.particle_output = particle_output
       
        vhigh = np.abs(ub - lb)
        vlow = -vhigh
    
        # Initialize objective function
        self.obj = partial(self._obj_wrapper, func, args, kwargs)
        
        # Check for constraint function(s) #########################################
        if f_ieqcons is None:
            if not len(ieqcons):
                if debug:
                    print('No constraints given.')
                cons = self._cons_none_wrapper
            else:
                if debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(self._cons_ieqcons_wrapper, ieqcons, args, kwargs)
        else:
            if debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(self._cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
        self.is_feasible = partial(self._is_feasible_wrapper, cons)
    
        # Initialize the multiprocessing module if necessary
        if processes > 1:
            import multiprocessing
            self.mp_pool = multiprocessing.Pool(processes)
            
        # Initialize the particle swarm ############################################
        self.S = swarmsize
        self.D = len(lb)  # the number of dimensions each particle has
        self.x = np.random.rand(self.S, self.D)  # particle positions
        self.v = np.zeros_like(self.x)  # particle velocities
        self.p = np.zeros_like(self.x)  # best particle positions
        self.fx = np.zeros(self.S)  # current particle function values
        self.fs = np.zeros(self.S, dtype=bool)  # feasibility of each particle
        self.fp = np.ones(self.S)*np.inf  # best particle function values
        self.g = []  # best swarm position
        self.fg = np.inf  # best swarm position starting value
        
        # Initialize the particle's position
        self.x = lb + self.x*(ub - lb)
    
        # Calculate objective and constraints for each particle
        if self.processes > 1:
            self.fx = np.array(self.mp_pool.map(self.obj, self.x))
            self.fs = np.array(self.mp_pool.map(self.is_feasible, self.x))
        else:
            for i in range(self.S):
                self.fx[i] = self.obj(self.x[i, :])
                self.fs[i] = self.is_feasible(self.x[i, :])
           
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((self.fx < self.fp), self.fs)
        self.p[i_update, :] = self.x[i_update, :].copy()
        self.fp[i_update] = self.fx[i_update]
    
        # Update swarm's best position
        i_min = np.argmin(self.fp)
        if self.fp[i_min] < self.fg:
            self.fg = self.fp[i_min]
            self.g = self.p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give self.it a temporary "best" point since self.it's likely to change
            self.g = self.x[0, :].copy()
           
        # Initialize the particle's velocity
        self.v = vlow + np.random.rand(self.S, self.D)*(vhigh - vlow)
           
        # Iterate until termination criterion met ##################################
        self.it = 1
        self.should_terminate = False

    def stop(self):
        return self.should_terminate

    def run(self):

        rp = np.random.uniform(size=(self.S, self.D))
        rg = np.random.uniform(size=(self.S, self.D))

        # Update the particles velocities
        self.v = self.omega*self.v + self.phip*rp*(self.p - self.x) + self.phig*rg*(self.g - self.x)
        # Update the particles' positions
        self.x = self.x + self.v
        # Correct for bound violations
        maskl = self.x < self.lb
        masku = self.x > self.ub
        self.x = self.x*(~np.logical_or(maskl, masku)) + self.lb*maskl + self.ub*masku

        # Update objectives and constraints
        if self.processes > 1:
            self.fx = np.array(self.mp_pool.map(self.obj, self.x))
            self.fs = np.array(self.mp_pool.map(self.is_feasible, self.x))
        else:
            for i in range(self.S):
                self.fx[i] = self.obj(self.x[i, :])
                self.fs[i] = self.is_feasible(self.x[i, :])

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((self.fx < self.fp), self.fs)
        self.p[i_update, :] = self.x[i_update, :].copy()
        self.fp[i_update] = self.fx[i_update]

        # Compare swarm's best position with global best position
        i_min = np.argmin(self.fp)
        if self.fp[i_min] < self.fg:
            if self.debug:
                print('New best for swarm at iteration {:}: {:} {:}'\
                    .format(self.it, self.p[i_min, :], self.fp[i_min]))

            p_min = self.p[i_min, :].copy()
            stepsize = np.sqrt(np.sum((self.g - p_min)**2))
            if np.abs(self.fg - self.fp[i_min]) <= self.minfunc:
                self.should_terminate = True 
                print('Stopping search: Swarm best objective change less than {:}'\
                    .format(self.minfunc))
                if self.particle_output:
                    return p_min, self.fp[i_min], self.p, self.fp
                else:
                    return p_min, self.fp[i_min]

            elif stepsize <= self.minstep:
                self.should_terminate = True 
                print('Stopping search: Swarm best position change less than {:}'\
                    .format(self.minstep))
                if self.particle_output:
                    return p_min, self.fp[i_min], self.p, self.fp
                else:
                    return p_min, self.fp[i_min]
            else:
                self.g = p_min.copy()
                self.fg = self.fp[i_min]

        if self.debug:
            print('Best after iteration {:}: {:} {:}'.format(self.it, self.g, self.fg))
        self.it += 1

        if self.particle_output:
            return self.g, self.fg, self.p, self.fp
        else:
            return self.g, self.fg



