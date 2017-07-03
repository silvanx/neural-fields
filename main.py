import json

import matplotlib.pyplot as py
import numpy as np

from population import Population1D
from substrate import Substrate1D


def g12(r1, r2, params):
    x = abs(abs(r1 - params["mu1"]) - abs(r2 - params["mu2"]))
    return -params["K12"] * np.exp(-x ** 2 / (2 * params["sigma12"]))


def g22(r1, r2, params):
    x = abs(r1 - r2)
    return -abs(r1 - r2) * params["K22"] * np.exp(-x ** 2 / (2 * params["sigma22"]))


def g21(r1, r2, params):
    x = abs(abs(r1 - params["mu1"]) - abs(r2 - params["mu2"]))
    return params["K21"] * np.exp(-x ** 2 / (2 * params["sigma21"]))


def get_connectivity(kernel, key, column, shape):
    if key in kernel:
        return kernel[key][column, :]
    else:
        return np.zeros(shape)


if __name__ == "__main__":
    print('Delayed Neural Fields')
    f = open('simulation_params', 'r')
    params = json.load(f)
    print(params)
    max_delta = 20
    dx = params["substrate"]["dx"]

    substrate = Substrate1D(params['substrate'], max_delta)

    populations = {name: Population1D(name, params['populations'][name], substrate) for name in params['populations']}

    w = dict()
    w[('stn', 'gpe')] = np.array([[g12(r1, r2, params) for r2 in populations['gpe'].substrate_grid]
                                  for r1 in populations['stn'].substrate_grid])
    w[('gpe', 'stn')] = np.array([[g21(r1, r2, params) for r2 in populations['stn'].substrate_grid]
                                  for r1 in populations['gpe'].substrate_grid])
    w[('gpe', 'gpe')] = np.array([[g22(r1, r2, params) for r2 in populations['gpe'].substrate_grid]
                                  for r1 in populations['gpe'].substrate_grid])
    w[('stn2', 'gpe2')] = np.array([[g12(r1, r2, params) for r2 in populations['gpe2'].substrate_grid]
                                    for r1 in populations['stn2'].substrate_grid])
    w[('gpe2', 'stn2')] = np.array([[g21(r1, r2, params) for r2 in populations['stn2'].substrate_grid]
                                    for r1 in populations['gpe2'].substrate_grid])
    w[('gpe2', 'gpe2')] = np.array([[g22(r1, r2, params) for r2 in populations['gpe2'].substrate_grid]
                                    for r1 in populations['gpe2'].substrate_grid])

    for i, t in enumerate(substrate.tt):
        states = {p.name: p.last_state() for p in populations.values()}
        if t > 0:
            inputs = dict()
            for pop in populations.keys():
                inputs[pop] = np.array([np.sum([np.dot(get_connectivity(w, (pop, p), ri, states[p].shape),
                                                       populations[p].delayed_activity(r, t))
                                                for p in populations.keys()]) + populations[pop].external_input(t)
                                        for ri, r in enumerate(populations[pop].substrate_grid)])
            for p in populations.keys():
                states[p] += substrate.dt/populations[p].tau * (-states[p] + populations[p].sigmoid(inputs[p]))
        for p in populations.keys():
            populations[p].update_state(t, states[p])

    print("simulation finished")
    populations['stn'].plot_history_average(False)
    populations['gpe'].plot_history_average(False)
    populations['stn2'].plot_history_average(False)
    populations['gpe2'].plot_history_average(False)
    py.legend(['stn', 'gpe', 'stn2', 'gpe2'])
    py.show()
    average = np.mean([np.mean(p.history, axis=1) for p in populations.values()], axis=0)
    py.subplot(2, 1, 1)
    py.plot(substrate.tt, average)
    Fs = 1000 / substrate.dt
    py.subplot(2, 1, 2)
    Pxx, freqs, bins, im = py.specgram(average, NFFT=1024, Fs=Fs, noverlap=10)
    py.show()
