import numpy as np
from optproblems.cec2005 import CEC2005

class Boundary:
    
    def __init__(self, dim, function_id):
        if function_id == 6:
            self.min_bounds = np.array([-300] * dim)
            self.max_bounds = np.array([600] * dim)
            self.init_min_bounds = np.array([0] * dim)
            self.init_max_bounds = np.array([600] * dim)
        #elif function_id == 8:
        #    self.min_bounds = np.array([0.5, -2])
        #    self.max_bounds = np.array([2.5, 0])
        #    self.init_min_bounds = np.array([0.5, -2])
        #    self.init_max_bounds = np.array([2.5, 0])
        elif function_id == 24:
            self.min_bounds = np.array([-5] * dim)
            self.max_bounds = np.array([10] * dim)
            self.init_min_bounds = np.array([2] * dim)
            self.init_max_bounds = np.array([5] * dim)
        else:
            self.min_bounds = np.array(CEC2005(dim)[function_id].min_bounds)
            self.max_bounds = np.array(CEC2005(dim)[function_id].max_bounds)
            self.init_min_bounds = np.array(CEC2005(dim)[function_id].min_bounds)
            self.init_max_bounds = np.array(CEC2005(dim)[function_id].max_bounds)

    
    def in_boundary(self, point):
        if np.any(np.less(point, self.min_bounds)):
            return False
        if None.any(np.greater(point, self.max_bounds)):
            return False


