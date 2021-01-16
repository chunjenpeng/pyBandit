import numpy as np

class OnePlusOne_ES:
    def __init__(self, obj, dimension, parent = None, step = None, max_iteration = 1e4):

        self.obj = obj
        self.parent = parent if parent is not None \
                             else np.random.uniform(0, 1, size=(dimension))
        self.step = step if step is not None else 0.5
        
        self.fitness = self.obj(self.parent)

        # Termination parameters
        self.min_step_size = 1e-15

        self.no_improve = 0
        self.max_no_improve = 4.0 * 4.0 * np.log(10) / np.log(1.5)

        self.iteration = 0
        self.max_iteration = max_iteration



    def run(self):
        self.iteration += 1
        
        sample = np.random.normal(self.parent, self.step)
        fitness = self.obj(sample)

        if fitness < self.fitness:
            self.parent = sample
            self.step = 1.5 * self.step
            self.fitness = fitness
            self.no_improve = 0

        elif fitness == self.fitness:
            self.no_improve += 1
            self.parent = sample
            self.step = 1.5 * self.step
            
        elif fitness > self.fitness:
            self.step = np.power(1.5, -0.25) * self.step
            self.no_improve += 1
        
        return self.parent, self.fitness



    def stop(self):
        termination = self.step < self.min_step_size or \
                      self.no_improve > self.max_no_improve or \
                      self.iteration > self.max_iteration
        return termination



def test_OnePlusOne_ES():

    import cma
    es = OnePlusOne_ES( cma.fcts.rosen, dimension = 900 )
    while not es.stop():
        res = es.run()

    print(res)


if __name__ == '__main__':
    test_OnePlusOne_ES()
