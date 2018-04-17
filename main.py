import argparse
import json
import pprint
from collections import OrderedDict

import utils
from control import *
from population import Population1D
from substrate import Substrate1D


def g12(r1, r2, mu1, mu2, params):
    x = r2 - r1 - (2 * mu1 + 10)
    return np.float(-params["K12"] * np.exp(-x ** 2 / (2 * params["sigma12"])))


def g22(r1, r2, mu1, mu2, params):
    x = abs(r1 - r2)
    return np.float(-abs(r1 - r2) * params["K22"] *
                    np.exp(-x ** 2 / (2 * params["sigma22"])))


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
    w_scale = params['W_scale']
    w = dict()
    pop_names = list(OrderedDict(sorted(populations.items(),
                                        key=lambda t: t[1].order)).keys())
    p1 = pop_names[0]
    p2 = pop_names[1]
    print('Generating kernels for populations: {} {}'.format(p1, p2))
    w[(p1, p2)] = w_scale * np.array(
        [[g12(r1, r2, populations[p1].mu, populations[p2].mu, params)
          for r2 in populations[p2].substrate_grid]
         for r1 in populations[p1].substrate_grid])
    w[(p2, p1)] = w_scale * np.array(
        [[g21(r1, r2, populations[p1].mu, populations[p2].mu, params)
          for r2 in populations[p1].substrate_grid]
         for r1 in populations[p2].substrate_grid])
    w[(p2, p2)] = w_scale * np.array(
        [[g22(r1, r2, populations[p2].mu, populations[p2].mu, params)
          for r2 in populations[p2].substrate_grid]
         for r1 in populations[p2].substrate_grid])
    print('Populations: {} {}'.format(p1, p2))
    print('The norm of w22 is ' + str(calculate_norm(w[(p2, p2)], dx)))
    print('The norm of w12 is ' + str(calculate_norm(w[(p1, p2)], dx)))
    print('The norm of w21 is ' + str(calculate_norm(w[(p2, p1)], dx)))
    return w


def add_field(field, substrate, params):
    p = {
        name: Population1D(name, field[name], substrate)
        for name in field
    }
    w = generate_kernels(p, params)
    return {"p": p, "w": w}


def calculate_norm(ar, dx):
    return np.sum(np.square(ar)) * dx ** 2


def calculate_dtheta(params, theta, states, ampl, p1):
    dt = params['substrate']['dt']
    tau_theta = params['tau_theta']
    sigma = params['sigma']
    # if abs(np.mean(states['stn'])) < params['feedback_threshold']:
    if ampl < params['feedback_threshold']:
        dtheta = substrate.dt / tau_theta * (- tau_theta * sigma * theta)
    else:
        dtheta = substrate.dt / tau_theta * (np.mean(states[p1]) ** 2 -
                                             tau_theta * sigma * theta)
        # dtheta = substrate.dt / tau_theta * (ampl ** 2 - tau_theta *
        #                                      sigma * theta)
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


def simulation_step(i, t, field, ctx_history, ampl_history, theta_history, dt):
    ampl = 0
    populations = field['p']
    pop_names = list(OrderedDict(sorted(populations.items(),
                                        key=lambda t: t[1].order)).keys())
    p1 = pop_names[0]
    p2 = pop_names[1]
    w = field['w']
    states = {p.name: p.last_state() for p in populations.values()}
    if t >= 0:
        inputs = dict()
        for pop in populations.keys():
            inputs[pop] = np.array([
                np.sum([np.dot(
                    get_connectivity(w, (pop, p), ri, states[p].shape),
                    populations[p].delayed_activity(r, i)
                ) * dx
                        for p in populations.keys()
                        ])
                for ri, r in enumerate(populations[pop].substrate_grid)
            ])
        if feedback and t >= feedback_start_time:
            inputs[p1] -= theta * states[p1] + params['ctx_nudge']
            theta_history[i] = theta
            if filtering:
                ampl = populations[p1].filtered_tail_amplitude(
                    int(params['filter_length'] / dt),
                    params['filter_order'],
                    params['filter_cutoff']
                )
            else:
                ampl = populations[p1].tail_amplitude(int(max_delta / dt))
            theta += calculate_dtheta(params, theta, states, ampl, p1)
        ctx_history[i] = calculate_ctx_suppression(t, params)
        ampl_history[i] = ampl
        inputs[p1] -= ctx_history[i]
        for p in populations.keys():
            states[p] += dt / populations[p].tau * \
                         (-states[p] + populations[p].sigmoid(inputs[p]))
    for p in populations.keys():
        populations[p].update_state(i, states[p])


def run_simulation(substrate, params, fields):
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
        for field in fields:
            simulation_step(i, t, field, ctx_history, ampl_history,
                            theta_history, dt)

    populations = substrate.populations
    w = field['w']

    return populations, w, theta_history, ctx_history, ampl_history


if __name__ == "__main__":
    print('Delayed Neural Fields')
    parser = argparse.ArgumentParser(description='Delayed Neural Fields')
    parser.add_argument('params', type=str,
                        help='File with parameters of the simulation')
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

    fields = [
        add_field(f, substrate, params)
        for f in params['fields']
    ]

    populations, w, theta_history, ctx_history, ampl_history = \
        run_simulation(substrate, params, fields)

    print("Simulation finished")

    if plot_connectivity:
        utils.plot_connectivity(w)
    utils.plot_simulation_results(populations, substrate, theta_history, ctx_history,
                                  ampl_history)
    utils.save_simulation_results(populations, theta_history, params)
