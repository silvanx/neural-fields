import numpy as np
from scipy.interpolate import interp1d


class State:

    def __init__(self, params, max_delta):
        self.tt = np.arange(-max_delta, params['tstop'], params['dt'])
        self.history = np.zeros((len(self.tt), 3))
        self.max_t = 0
        self.init_state = [params["x10"], params["x20"], params["theta0"]]

    def __call__(self, t):
        if t < 0:
            return self.initial_state()
        elif t > self.max_t:
            raise IndexError("Time {} is not yet evaluated (last evaluated: {})".format(t, self.max_t))
        else:
            return self.state_from_history(t)

    def initial_state(self):
        return self.init_state

    def state_from_history(self, t):
        index = min(np.where(self.tt >= t)[0])
        if self.tt[index] == t:
            return np.array(self.history[index])
        else:
            x = [self.tt[index - 1], self.tt[index]]
            y = self.history[x]
            interpolator = interp1d(x, y, axis=0)
            return np.array(interpolator(t))

    def add_to_history(self, t, value):
        self.history[np.where(self.tt == t)] = value
        self.max_t = t