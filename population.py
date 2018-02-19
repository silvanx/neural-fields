import matplotlib.pyplot as py
import numpy as np
from scipy.interpolate import interp1d, interp2d

from control import *


class Population:

    def __init__(self, name, params, substrate):
        self.max_index = 0
        self.name = name
        initial_value = params['x0']
        self.physical_size = np.array(params['x_size'], dtype=float).flatten()
        self.starting_point = np.array(params['starting_point'], dtype=float).flatten()
        if len(self.physical_size) != len(self.starting_point):
            raise ValueError("Dimensions of the population and the starting point do not agree")
        self.dimensions = len(self.physical_size)
        self.substrate_grid = substrate.place_population(self)
        self.tt = substrate.tt
        self.substrate = substrate
        self.max_t = substrate.tt[0]
        self.history = np.zeros((len(self.tt), len(self.substrate_grid)))
        if len(np.array([initial_value]).flatten()) == 1:
            self._initial_state = initial_value * np.ones(self.substrate_grid.shape)
        elif np.array(initial_value).shape != self.substrate_grid.shape:
            raise ValueError("Shape of initial value vector: {} doesn't match the underlying grid: {}".format(
                np.array(initial_value).shape, self.substrate_grid.shape))
        else:
            self._initial_state = initial_value
        self.history[self.tt <= 0, :] = self._initial_state
        self.axonal_speed = params['axonal_speed']
        self.m = params['m']
        self.b = params['b']
        self.dt = substrate.dt
        self.mu = params['starting_point'] + params['x_size'] / 2
        self.tau = params['tau']
        if 'control' in params.keys():
            if params['control']['type'] == 'adaptive':
                self.control = AdaptiveProportionalControl(params['control'], self)
            elif params['control']['type'] == 'alpha':
                self.control = AlphaControl(params['control'], self)
            elif params['control']['type'] == 'dissipative':
                self.control = DissipativeControl(params['control'], self)
            elif params['control']['type'] == 'zero':
                self.control = ZeroControl(self)
        else:
            self.control = ZeroControl(self)

    def sigmoid(self, x):
        numerator = self.m * self.b
        denominator = self.b + (self.m - self.b) * np.array([np.exp(r) for r in (-4 * (x + self.m * np.log((self.m - self.b) / self.b) / 4) / self.m)])
        return numerator / denominator.astype(float) - self.m/2

    # def sigmoid(self, x):
    #     numerator = self.m * self.b
    #     denominator = self.b + (self.m - self.b) * np.array([np.exp(r) for r in (-4 * x / self.m)])
    #     return numerator / denominator.astype(float)

    def delay(self, r1, r2):
        return abs(r1 - r2) / self.axonal_speed

    def initial_state(self, x):
        index = self.get_index_from_position(x)
        if len(index) == 1:
            return self._initial_state[index[0]]
        else:
            inter = interp1d([self.substrate_grid[i] for i in index], [self._initial_state[i] for i in index])
            return inter(x)

    def get_index_from_position(self, x):
        if x < self.starting_point or x > self.starting_point + self.physical_size:
            raise IndexError("Position not within the population")
        index = np.where(self.substrate_grid >= x)[0][0]
        if self.substrate_grid[index] == x:
            return [index]
        else:
            return [index - 1, index]

    def get_index_from_time(self, t):
        index = np.where(self.tt >= t)[0][0]
        if self.tt[index] == t:
            return [index]
        else:
            return [index - 1, index]

    def state_from_history(self, x, t):
        x_index = self.get_index_from_position(x)
        t_index = self.get_index_from_time(t)
        if len(x_index) == 1 and len(t_index) == 1:
            return self.history[t_index, x_index]
        elif len(x_index) == 2 and len(t_index) == 2:
            interpolator = interp2d(t_index, x_index, [[self.history[t, x] for x in x_index] for t in t_index])
            return interpolator(t, x)
        elif len(x_index) == 1:
            interpolator = interp1d([self.tt[s] for s in t_index], [self.history[s, x_index[0]] for s in t_index])
            return interpolator(t)
        else:
            interpolator = interp1d([self.substrate_grid[r] for r in x_index], [self.history[t_index[0], r] for r in x_index])
            return interpolator(x)

    def update_state(self, t, state):
        index = self.get_index_from_time(t)
        if len(index) == 1:
            self.history[index] = state
            self.max_t = t
            self.max_index = index[0]
            # self.control.update_gain(t)
        else:
            raise ValueError("Selected time not in the grid")

    def plot_history(self, show=True):
        py.plot(self.tt, self.history[:, :])
        if show:
            py.show()

    def plot_history_average(self, show=True):
        py.plot(self.tt, self.history.mean(axis=1))
        if show:
            py.show()

    def get_tail(self, length):
        return self.history.mean(axis=1)[self.max_index - length : self.max_index]

    def __call__(self, x, t):
        if t <= 0:
            return self.initial_state(x)
        if t > self.max_t:
            raise IndexError("Time {} is not yet evaluated (last evaluated: {})".format(t, self.max_t))
        else:
            return np.array([self.state_from_history(x, t)]).flatten()[0]

    def delayed_activity(self, r, tn):
        """
        Create a vector with activity from entire population converging on point r at time t
        :param r:
        :param tn:
        :return:
        """
        t = tn - self.substrate.dt
        timepoints = [t - self.delay(r, x) for x in self.substrate_grid]
        return [self.__call__(x, s) for x, s in zip(self.substrate_grid, timepoints)]

    def last_state(self):
        index = min(np.where(self.tt >= self.max_t)[0])
        return self.history[index, :]


class Population1D(Population):
    pass


class Population2D(Population):
    pass


class Population3D(Population):
    pass
