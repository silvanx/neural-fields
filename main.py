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

    stn = Population1D(params['populations']['stn'], substrate)
    gpe = Population1D(params['populations']['gpe'], substrate)

    state_stn = np.zeros(stn.substrate_grid.shape)
    state_gpe = np.zeros(stn.substrate_grid.shape)

    w12 = -np.ones((state_stn.shape[0], state_gpe.shape[0]))
    w21 = np.ones((state_gpe.shape[0], state_stn.shape[0]))
    w22 = -0.1 * np.ones((state_gpe.shape[0], state_gpe.shape[0]))

    for i, t in enumerate(substrate.tt):
        state_stn = stn.last_state()
        state_gpe = gpe.last_state()
        if t > 0:
            inputs_to_stn = np.array([np.dot(w12[ri, :], gpe.delayed_activity(r, t))
                                      for ri, r in enumerate(stn.substrate_grid)])
            inputs_to_gpe = np.array([np.dot(w22[ri, :], gpe.delayed_activity(r, t)) +
                                      np.dot(w21[ri, :], stn.delayed_activity(r, t))
                                      for ri, r in enumerate(gpe.substrate_grid)])
            state_stn += params['substrate']['dt']/params['populations']['stn']['tau'] \
                         * (-state_stn + stn.sigmoid(inputs_to_stn))
            state_gpe += params['substrate']['dt']/params['populations']['gpe']['tau'] \
                         * (-state_gpe + gpe.sigmoid(inputs_to_gpe))
        stn.update_state(t, state_stn)
        gpe.update_state(t, state_gpe)

    print("simulation finished")
    stn.plot_history()
    gpe.plot_history()
