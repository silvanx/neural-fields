import unittest
from substrate import Substrate
from population import Population
import numpy as np


class InitializationTestCase(unittest.TestCase):
    def setUp(self):
        self.s1d = Substrate(1, 10, 1)

    def tearDown(self):
        del self.s1d

    # Test to see if the population is larger than the substrate

    def testPopulationTooBig1D(self):
        self.assertRaises(ValueError, Population, 20, 1, self.s1d, 20)

    # Test successful creation of a population

    def testPopulation1D(self):
        Population(9, 1, self.s1d, 20)

    # Test population spanning the entire substrate

    def testPopulation1DFull(self):
        Population(10, 0, self.s1d, 20)

    # Test placing population out of bounds

    def testPopulationOutOfBounds1D(self):
        self.assertRaises(ValueError, Population, 5, 7, self.s1d, 20)


class InitialStateTestCase(unittest.TestCase):
    def setUp(self):
        self.substrate = Substrate(1, 20, 1)
        self.population = Population(10, 0, self.substrate, 20, initial_value=[i for i in range(1, 11)])

    def testSingleValue(self):
        population = Population(10, 0, self.substrate, 20, initial_value=1)
        np.testing.assert_array_equal(np.ones(10,), population._initial_state)

    def testArrayValue(self):
        np.testing.assert_array_equal([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], self.population._initial_state)

    def testInterpolate(self):
        self.assertEqual(4.5, self.population.initial_state(3.5))

if __name__ == '__main__':
    unittest.main()
