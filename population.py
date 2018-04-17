import matplotlib.pyplot as py
from scipy import signal
from scipy.interpolate import interp1d

from control import *


class Population:

    def __init__(self, name, params, substrate):
        self.max_index = 0
        self.name = name
        initial_value = params['x0']
        self.order = params['order']
        self.physical_size = np.array(params['x_size'], dtype=float).flatten()
        self.starting_point = np.array(params['starting_point'],
                                       dtype=float).flatten()
        if len(self.physical_size) != len(self.starting_point):
            raise ValueError("Dimensions of the population and the starting \
                             point do not agree")
        self.dimensions = len(self.physical_size)
        self.substrate_grid = substrate.place_population(self)
        self.tt = substrate.tt
        self.substrate = substrate
        self.max_t = substrate.tt[0]
        self.history = np.zeros((len(self.tt), len(self.substrate_grid)))
        if len(np.array([initial_value]).flatten()) == 1:
            self._initial_state = \
                initial_value * np.ones(self.substrate_grid.shape)
        elif np.array(initial_value).shape != self.substrate_grid.shape:
            raise ValueError("Shape of initial value vector: {} doesn't match \
                             the underlying grid: {}".format(
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
        self.control = ZeroControl(self)

    def sigmoid(self, x):
        numerator = self.m * self.b
        denominator = self.b + (self.m - self.b) * \
                      np.array([np.exp(r)
                                for r in (-4 * (x + self.m *
                                                np.log((self.m - self.b) / self.b) / 4) /
                                          self.m)])
        return numerator / denominator.astype(float) - self.m/2

    def delay(self, r1, r2):
        return abs(r1 - r2) / self.axonal_speed

    def initial_state(self, x):
        index = self.get_index_from_position(x)
        if len(index) == 1:
            return self._initial_state[index[0]]
        else:
            inter = interp1d([self.substrate_grid[i] for i in index],
                             [self._initial_state[i] for i in index])
            return inter(x)

    def get_index_from_position(self, x):
        if x < self.starting_point or \
                x > self.starting_point + self.physical_size:
            raise IndexError("Position not within the population")
        return [int((x - self.starting_point) / self.substrate.dx)]

    def get_index_from_time(self, t):
        return [int(t / self.dt + self.substrate.max_delta)]

    def state_from_history(self, x, t_index):
        x_index = self.get_index_from_position(x)
        return self.history[t_index, x_index]

    def update_state(self, index, state):
        self.history[index] = state
        self.max_index = index
        # self.control.update_gain(t)

    def plot_history(self, show=True):
        py.plot(self.tt, self.history[:, :])
        if show:
            py.show()

    def plot_history_average(self, show=True):
        py.plot(self.tt, self.history.mean(axis=1))
        if show:
            py.show()

    def get_tail(self, l):
        return self.history.mean(axis=1)[self.max_index - l:self.max_index]

    def __call__(self, x, i):
        return np.array([self.state_from_history(x, i)]).flatten()[0]

    def delayed_activity(self, r, index):
        """
        Create a vector with activity from entire population converging on \
        point r at time index index
        :param r:
        :param index:
        :return:
        """
        i = index - 1
        timepoints = [i - int(self.delay(r, x) // self.substrate.dt)
                      for x in self.substrate_grid]
        return [self.__call__(x, s)
                for x, s in zip(self.substrate_grid, timepoints)]

    def last_state(self):
        index = self.max_index
        # index = min(np.where(self.tt >= self.max_t)[0])
        return self.history[index, :]

    def tail_amplitude(self, length):
        tail = self.get_tail(length)
        return tail.ptp()

    def filtered_tail_amplitude(self, length, order, cutoff):
        fs = 1000 / self.substrate.dt
        nyq = fs / 2

        cutoff_norm = cutoff / nyq
        tail = self.get_tail(length)
        b, a = signal.butter(order, cutoff_norm)
        filtered_tail = signal.lfilter(b, a, tail)
        return filtered_tail.ptp()


class Population1D(Population):
    pass


class Population2D(Population):
    pass


class Population3D(Population):
    pass
