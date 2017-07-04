import numpy as np


class Control:
    def __init__(self, params, population):
        self.initial_value = float(params['initial_value'])
        self.start_time = params['start_time']
        self.population = population
        self.tau = params['tau']
        self.dt = population.dt
        self.tt = population.tt
        self.history = np.zeros(population.history.shape)

    def __call__(self, t):
        pass

    def update_gain(self, i):
        pass

    def get_index_from_time(self, t):
        index = min(np.where(self.tt >= t)[0])
        if self.tt[index] == t:
            return [index]
        else:
            return [index - 1, index]


class AdaptiveProportionalControl(Control):
    def __init__(self, params, population):
        super().__init__(params, population)
        self.gain = self.initial_value * np.ones(self.population.last_state().shape)

    def __call__(self, tt):
        t = tt - self.dt
        if t < self.start_time:
            return self.initial_value
        else:
            i = self.get_index_from_time(t)
            if len(i) == 1:
                return np.array([-self.population(x, t) for x in self.population.substrate_grid] * self.history[i][0])

    def update_gain(self, i):
        self.gain += self.dt / self.tau * np.absolute(self.population.last_state())
        self.history[i] = self.gain


class ZeroControl(Control):
    def __init__(self, population):
        super().__init__({'initial_value': 0, 'start_time': 0, 'tau': 0}, population)

    def __call__(self, t):
        return np.zeros(self.population.last_state().shape)
