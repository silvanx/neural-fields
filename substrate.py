import numpy as np


class Substrate(object):

    def __init__(self, dim, params, max_delta):
        """Create a new substrate with specified number of spatial dimensions, \
        physical size and grain
        """
        if dim not in [1, 2, 3]:
            raise ValueError("Dimension should be 1, 2 or 3")
        self.dimensions = dim
        self.physical_size = np.array([params['x_size']],
                                      dtype=float).flatten()
        if self.physical_size.shape[0] != dim:
            raise TypeError("physical_size should have {} elements, \
                            has {}".format(dim, self.physical_size.shape[0]))
        self.dt = params['dt']
        self.dx = params['dx']
        self.grid = self.generate_grid(float(self.dx))
        self.populations = []
        self.tt = np.arange(-max_delta, params['tstop'], self.dt)
        self.max_delta = max_delta

    def generate_grid(self, dx):
        axes = [np.arange(0, self.physical_size[i] + dx, dx)
                for i in range(self.dimensions)]
        return np.meshgrid(*axes)

    def place_population(self, population):
        if population.dimensions != self.dimensions:
            raise ValueError("Number of dimensions of the population does not \
                             match the number of dimensions of the substrate")
        if np.any(population.physical_size > self.physical_size):
            raise ValueError("All dimensions of the population have to be \
                             smaller than the substrate")
        if np.any(np.array(population.starting_point) +
                  np.array(population.physical_size) > self.physical_size):
            raise ValueError("With this starting point the population exceeds \
                             the substrate")
        self.populations.append(population)


class Substrate1D(Substrate):
    def __init__(self, *args):
        super().__init__(1, *args)

    def place_population(self, population):
        super().place_population(population)
        return self.grid[0][np.all([self.grid[0] >= population.starting_point,
                                    self.grid[0] <= population.starting_point +
                                    population.physical_size], axis=0)]


class Substrate2D(Substrate):
    def __init__(self, *args):
        super().__init__(2, *args)

    def place_population(self, population):
        super().place_population(population)
        # TODO: Implement returning grid for 2 dimensions
        raise NotImplementedError("2D and 3D populations not functional yet")


class Substrate3D(Substrate):
    def __init__(self, *args):
        super().__init__(3, *args)

    def place_population(self, population):
        super().place_population(population)
        # TODO: Implement returning grid for 3 dimensions
        raise NotImplementedError("2D and 3D populations not functional yet")
