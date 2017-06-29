import unittest
from state import State
import json
import numpy as np


class MyTestCase(unittest.TestCase):
    def setUp(self):
        f = open("../simulation_params")
        self.params = json.load(f)
        self.maxDelay = 6
        self.state = State(self.params, self.maxDelay)
        for i, t in enumerate(self.state.tt):
            x = self.state(t - self.params['dt'])
            self.state.add_to_history(t, x)

    def tearDown(self):
        del self.state
        del self.params

    def testInitialCondition(self):
        self.assertListEqual(self.state(-1), [self.params["x10"], self.params["x20"], self.params["theta0"]])

    def testInitialConditionOutOfBounds(self):
        self.assertListEqual(self.state(-self.maxDelay * 2), [self.params["x10"], self.params["x20"],
                                                              self.params["theta0"]])

    def testInitialConditionInterpolate(self):
        self.assertListEqual(self.state(-self.maxDelay - self.params['dt'] / 2),
                             [self.params["x10"], self.params["x20"], self.params["theta0"]])

    def testStableState(self):
        np.testing.assert_array_equal(self.state(self.params["tstop"] - self.params["dt"]),
                                      np.array([self.params["x10"], self.params["x20"], self.params["theta0"]]))

    def testStableStateInterpolate(self):
        np.testing.assert_array_equal(self.state(self.params["tstop"] - self.params["dt"] * 1.5),
                                      np.array([self.params["x10"], self.params["x20"], self.params["theta0"]]))


if __name__ == '__main__':
    unittest.main()
