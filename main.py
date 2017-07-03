import numpy as np
from substrate import Substrate1D
from population import Population1D
import json

if __name__ == "__main__":
    print('Delayed Neural Fields')
    f = open('simulation_params', 'r')
    params = json.load(f)
    print(params)
    max_delta = 20
    dx = params["substrate"]["dx"]

    substrate = Substrate1D(params['substrate'], max_delta)

    populations = {name: Population1D(name, params['populations'][name], substrate) for name in params['populations']}
    states = {population.name: np.zeros(population.substrate_grid.shape) for population in populations.values()}

    stn = populations['stn2']
    gpe = populations['gpe2']

    state_stn = states['stn2']
    state_gpe = states['gpe2']

    def g12(r1, r2, params):
        x = abs(abs(r1 - params["mu1"]) - abs(r2 - params["mu2"]))
        return -params["K12"] * np.exp(-x ** 2 / (2 * params["sigma12"]))

    def g22(r1, r2, params):
        x = abs(r1 - r2)
        return -abs(r1 - r2) * params["K22"] * np.exp(-x ** 2 / (2 * params["sigma22"]))

    def g21(r1, r2, params):
        x = abs(abs(r1 - params["mu1"]) - abs(r2 - params["mu2"]))
        return params["K21"] * np.exp(-x ** 2 / (2 * params["sigma21"]))


    w11 = np.zeros((state_stn.shape[0], state_stn.shape[0]))
    w12 = np.array([[g12(r1, r2, params) for r2 in gpe.substrate_grid] for r1 in stn.substrate_grid])
    w21 = np.array([[g21(r1, r2, params) for r2 in stn.substrate_grid] for r1 in gpe.substrate_grid])
    w22 = np.array([[g22(r1, r2, params) for r2 in gpe.substrate_grid] for r1 in gpe.substrate_grid])

    for i, t in enumerate(substrate.tt):
        state_stn = stn.last_state()
        state_gpe = gpe.last_state()
        if t > 0:
            inputs_to_stn = np.array([np.dot(w11[ri, :], gpe.delayed_activity(r, t)) +
                                      np.dot(w12[ri, :], gpe.delayed_activity(r, t)) +
                                      27 * 12.5
                                      for ri, r in enumerate(stn.substrate_grid)])
            inputs_to_gpe = np.array([np.dot(w22[ri, :], gpe.delayed_activity(r, t)) +
                                      np.dot(w21[ri, :], stn.delayed_activity(r, t)) -
                                      110 * 2
                                      for ri, r in enumerate(gpe.substrate_grid)])
            state_stn += params['substrate']['dt']/params['populations']['stn']['tau'] \
                         * (-state_stn + stn.sigmoid(inputs_to_stn))
            state_gpe += params['substrate']['dt']/params['populations']['gpe']['tau'] \
                         * (-state_gpe + gpe.sigmoid(inputs_to_gpe))
        stn.update_state(t, state_stn)
        gpe.update_state(t, state_gpe)

    print("simulation finished")
    stn.plot_history_average()
    gpe.plot_history_average()
