import unittest
from substrate import Substrate
from population import Population
import numpy as np


class InitializationTestCase(unittest.TestCase):
    def setUp(self):
        self.s1d = Substrate(1, 10, 0.5)

    def testInitTooManyDimensions(self):
        self.assertRaises(ValueError, Substrate, 4, 10, 1)

    def testInitDimensionMismatch(self):
        self.assertRaises(TypeError, Substrate, 3, 10, 1)

    @staticmethod
    def testInit1D():
        Substrate(1, 10, 1)

    @staticmethod
    def testInit2D():
        Substrate(2, [10, 11], 1)

    @staticmethod
    def testInit3D():
        Substrate(3, [10, 12, 14], 1)

    def testSubstrateGrid1D(self):
        np.testing.assert_array_equal(self.s1d.grid[0], np.arange(0, 10, 0.5))

    def testSubstratePopulationGrid1D(self):
        p = Population(2, 2, self.s1d, 20)
        np.testing.assert_array_equal(p.substrate_grid, np.arange(2, 4, 0.5))


if __name__ == '__main__':
    unittest.main()
