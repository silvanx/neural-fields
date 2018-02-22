import argparse
import json
import pprint

import matplotlib.pyplot as py

import utils
from control import *
from population import Population1D
from substrate import Substrate1D


def g12(r1, r2, mu1, mu2, params):
    # x = abs(abs(r1 - mu1) - abs(r2 - mu2))
    x = r2 - r1 - (2 * mu1 + 10)
    return np.float(-params["K12"] * np.exp(-x ** 2 / (2 * params["sigma12"])))


def g22(r1, r2, mu1, mu2, params):
    x = abs(r1 - r2)
    return np.float(-abs(r1 - r2) * params["K22"] * np.exp(-x ** 2 / (2 * params["sigma22"])))


def g21(r1, r2, mu1, mu2, params):
    # r1 -> gpe; r2 -> stn
    # x = abs(abs(r1 - mu1) - abs(r2 - mu2))
    x = r2 - r1 + (2 * mu1 + 10)
    return np.float(params["K21"] * np.exp(-x ** 2 / (2 * params["sigma21"])))


def get_connectivity(kernel, key, column, shape):
    if key in kernel:
        return kernel[key][column, :]
    else:
        return np.zeros(shape)

if __name__ == "__main__":
    print('Delayed Neural Fields')
    parser = argparse.ArgumentParser(description='Delayed Neural Fields simulation')
    parser.add_argument('params', type=str, help='File with parameters of the simulation')
    args = parser.parse_args()
    f = open(args.params, 'r')
    params = json.load(f)
    pprint.pprint(params)
    # TODO: Calculate max delta
    max_delta = 20
    dx = params["substrate"]["dx"]
    plot_connectivity = False
    feedback = True
    average_feedback = False

    substrate = Substrate1D(params['substrate'], max_delta)

    populations = {name: Population1D(name, params['populations'][name], substrate) for name in params['populations']}

    theta_history = np.zeros(substrate.tt.shape)
    feedback_start_time = params['feedback_start_time']
    theta = params['theta0']
    sigma = params['sigma']
    tau_theta = params['tau_theta']

    w = dict()
    w[('stn', 'gpe')] = np.array([[g12(r1, r2, populations['stn'].mu, populations['gpe'].mu, params) for r2 in populations['gpe'].substrate_grid]
                                  for r1 in populations['stn'].substrate_grid])
    w[('gpe', 'stn')] = np.array([[g21(r1, r2, populations['stn'].mu, populations['gpe'].mu, params) for r2 in populations['stn'].substrate_grid]
                                  for r1 in populations['gpe'].substrate_grid])
    w[('gpe', 'gpe')] = np.array([[g22(r1, r2, populations['gpe'].mu, populations['gpe'].mu, params) for r2 in populations['gpe'].substrate_grid]
                                  for r1 in populations['gpe'].substrate_grid])


    # py.figure()
    # wtotal = np.array([[g21(r2, r1, populations['stn'].mu, populations['gpe'].mu, params) for r2 in np.arange(0, 15 + dx, dx)]
    #                               for r1 in np.arange(0, 15 + dx, dx)])
    # py.imshow(wtotal)

    wmin = np.min([np.min(ww) for ww in w.values()])
    wmax = np.max([np.max(ww) for ww in w.values()])

    if plot_connectivity:
        py.figure()
        py.subplot(222)
        py.imshow(w[('stn', 'gpe')], cmap='RdBu', vmin=wmin, vmax=wmax)
        py.title('stn-gpe')
        py.subplot(223)
        py.imshow(w[('gpe', 'stn')], cmap='RdBu', vmin=wmin, vmax=wmax)
        py.title('gpe-stn')
        py.subplot(224)
        py.imshow(w[('gpe', 'gpe')], cmap='RdBu', vmin=wmin, vmax=wmax)
        py.title('gpe-gpe')

    print('The norm of w22 is ' + str(np.sum(w[('gpe', 'gpe')] ** 2) * dx ** 2))
    print('The norm of w12 is ' + str(np.sum(w[('stn', 'gpe')] ** 2) * dx ** 2))
    print('The norm of w21 is ' + str(np.sum(w[('gpe', 'stn')] ** 2) * dx ** 2))

    for i, t in enumerate(substrate.tt):
        if i % 200 == 0:
            print('Time: {} ms'.format(t))
        states = {p.name: p.last_state() for p in populations.values()}
        if t >= 0:
            inputs = dict()
            for pop in populations.keys():
                inputs[pop] = np.array([np.sum([np.dot(get_connectivity(w, (pop, p), ri, states[p].shape),
                                                       populations[p].delayed_activity(r, i)) * dx
                                                for p in populations.keys()])
                                        for ri, r in enumerate(populations[pop].substrate_grid)])
            if feedback and t >= feedback_start_time:
                inputs['stn'] -= theta * states['stn']
                theta_history[i] = theta
                theta += substrate.dt / tau_theta * (np.mean(states['stn']) ** 2 - sigma * theta)
            for p in populations.keys():
                states[p] += substrate.dt/populations[p].tau * (-states[p] + populations[p].sigmoid(inputs[p]))
        for p in populations.keys():
            populations[p].update_state(i, states[p])

    print("Simulation finished")

    utils.plot_simulation_results(populations, theta_history)
    utils.save_simulation_results(populations, theta_history, params)
