import numpy as np
from substrate import Substrate1D
from population import Population1D
import json


def model(x, t, params):
    invtau = 1 / params['tau1']
    if t > 0:
        return -invtau * x
    else:
        return np.zeros(np.shape(x))


if __name__ == "__main__":
    print('Delayed Neural Fields')
    f = open('simulation_params', 'r')
    params = json.load(f)
    print(params)
    max_delta = 20
    dx = params["substrate"]["dx"]

    substrate = Substrate1D(params['substrate'], max_delta)

    stn = Population1D(params['populations']['stn'], substrate)
    gpe = Population1D(params['populations']['gpe'], substrate)

    state_stn = np.zeros(stn.substrate_grid.shape)
    state_gpe = np.zeros(stn.substrate_grid.shape)

    for i, t in enumerate(substrate.tt):
        state_stn = stn.last_state()
        state_gpe = gpe.last_state()
        if t > 0:
            state_stn += - params['substrate']['dt']/params['populations']['stn']['tau'] \
                         * (state_stn)
            state_gpe += -params['substrate']['dt']/params['populations']['gpe']['tau'] \
                         * (state_gpe)
        stn.update_state(t, state_stn)
        gpe.update_state(t, state_gpe)

    print("simulation finished")
    stn.plot_history()
    gpe.plot_history()
