import argparse
import json
import pprint

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


def generate_kernels(populations, params):
    W_scale = params['W_scale']
    w = dict()
    w[('stn', 'gpe')] = W_scale * np.array([[g12(r1, r2, populations['stn'].mu, populations['gpe'].mu, params)
                                             for r2 in populations['gpe'].substrate_grid]
                                            for r1 in populations['stn'].substrate_grid])
    w[('gpe', 'stn')] = W_scale * np.array([[g21(r1, r2, populations['stn'].mu, populations['gpe'].mu, params)
                                             for r2 in populations['stn'].substrate_grid]
                                            for r1 in populations['gpe'].substrate_grid])
    w[('gpe', 'gpe')] = W_scale * np.array([[g22(r1, r2, populations['gpe'].mu, populations['gpe'].mu, params)
                                             for r2 in populations['gpe'].substrate_grid]
                                            for r1 in populations['gpe'].substrate_grid])
    return w


def calculate_norm(ar, dx):
    return np.sum(np.square(ar)) * dx ** 2


def calculate_dtheta(params, theta, states, ampl):
    dt = params['substrate']['dt']
    tau_theta = params['tau_theta']
    sigma = params['sigma']
    # if abs(np.mean(states['stn'])) < params['feedback_threshold']:
    if ampl < params['feedback_threshold']:
        dtheta = substrate.dt / tau_theta * (- tau_theta * sigma * theta)
    else:
        # dtheta = substrate.dt / tau_theta * (np.mean(states['stn']) ** 2 - tau_theta * sigma * theta)
        dtheta = substrate.dt / tau_theta * (ampl ** 2 - tau_theta * sigma * theta)
    return dtheta


def calculate_ctx_suppression(t, params):
    amplitude = params["ctx_suppression"]
    start = params["ctx_suppression_start"]
    periodic = params["ctx_suppression_periodic"] == 1
    period = params["ctx_suppression_period"]
    dc = params["ctx_suppression_duty_cycle"]

    if t > start and amplitude > 0:
        if periodic:
            t_elapsed = t - start
            point_in_cycle = t_elapsed % period
            if point_in_cycle > dc * period:
                return 0
            else:
                return amplitude
        else:
            return amplitude
    else:
        return 0


def run_simulation(substrate, params):
    populations = {p.name: p for p in substrate.populations}
    w = generate_kernels(populations, params)
    theta_history = np.zeros(substrate.tt.shape)
    ctx_history = np.zeros(substrate.tt.shape)
    ampl_history = np.zeros(substrate.tt.shape)
    feedback_start_time = params['feedback_start_time']
    theta = params['theta0']
    ampl = 0
    dt = params['substrate']['dt']
    filtering = params['filtering'] == 1

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
                inputs['stn'] -= theta * states['stn'] + params['ctx_nudge']
                theta_history[i] = theta
                if filtering:
                    ampl = populations['stn'].filtered_tail_amplitude(int(params['filter_length'] / dt), params['filter_order'], params['filter_cutoff'])
                else:
                    ampl = populations['stn'].tail_amplitude(int(max_delta / dt))
                theta += calculate_dtheta(params, theta, states, ampl)
            ctx_history[i] = calculate_ctx_suppression(t, params)
            ampl_history[i] = ampl
            inputs['stn'] -= ctx_history[i]
            for p in populations.keys():
                states[p] += substrate.dt / populations[p].tau * (-states[p] + populations[p].sigmoid(inputs[p]))
        for p in populations.keys():
            populations[p].update_state(i, states[p])

    return populations, w, theta_history, ctx_history, ampl_history

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
    feedback = params['feedback'] == 1
    average_feedback = False

    params['average_feedback'] = average_feedback

    substrate = Substrate1D(params['substrate'], max_delta)

    populations = {name: Population1D(name, params['populations'][name], substrate) for name in params['populations']}

    populations, w, theta_history, ctx_history, ampl_history = run_simulation(substrate, params)

    print("Simulation finished")

    print('The norm of w22 is ' + str(calculate_norm(w[('gpe', 'gpe')], dx)))
    print('The norm of w12 is ' + str(calculate_norm(w[('stn', 'gpe')], dx)))
    print('The norm of w21 is ' + str(calculate_norm(w[('gpe', 'stn')], dx)))

    if plot_connectivity:
        utils.plot_connectivity(w)
    utils.plot_simulation_results(populations, theta_history, ctx_history, ampl_history, params)
    utils.save_simulation_results(populations, theta_history, params)
