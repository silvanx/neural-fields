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
    dx = populations[p1].substrate.dx
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
        name: Population1D(name, field['populations'][name], substrate)
        for name in field['populations']
    }
    w = generate_kernels(p, field['connections'])
    c = field['cortex']
    return {"p": p, "w": w, "c": c}


def calculate_norm(ar, dx):
    return np.sum(np.square(ar)) * dx ** 2


def calculate_ctx(t, params):
    sup_amplitude = params["ctx_suppression"]
    sup_start = params["ctx_suppression_start"]
    sup_periodic = params["ctx_suppression_periodic"] == 1
    sup_period = params["ctx_suppression_period"]
    sup_dc = params["ctx_suppression_duty_cycle"]

    supression = 0
    if t > sup_start and sup_amplitude > 0:
        if sup_periodic:
            t_elapsed = t - sup_start
            point_in_cycle = t_elapsed % sup_period
            if point_in_cycle <= sup_dc * sup_period:
                supression = sup_amplitude
        else:
            supression = sup_amplitude

    en_amplitude = params["ctx_entrainment_amplitude"]
    en_frequency = params["ctx_entrainment_frequency"]
    en_phase = params["ctx_entrainment_phase"]
    en_start = params["ctx_entrainment_start"]
    tt = (t - en_start) / 1000
    entrainment = en_amplitude * np.sin(tt * 2 * np.pi * en_frequency + en_phase)

    return supression + entrainment


def simulation_step(i, t, field, ctx_history, dt,
                    feedback_start_time, theta, feedback, ctx_params):
    populations = field['p']
    pop_names = list(OrderedDict(sorted(populations.items(),
                                        key=lambda t: t[1].order)).keys())
    p1 = pop_names[0]
    w = field['w']
    dx = populations[p1].substrate.dx
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
            inputs[p1] -= theta * states[p1]
        ctx_history[i] = calculate_ctx(t, ctx_params) + ctx_params['ctx_nudge']
        inputs[p1] -= ctx_history[i]
        for p in populations.keys():
            states[p] += dt / populations[p].tau * \
                         (-states[p] + populations[p].sigmoid(inputs[p]))
    for p in populations.keys():
        populations[p].update_state(i, states[p])


def update_feedback_gain(t, params, substrate, theta):
    feedback = params['feedback'] == 1
    tau_theta = params['tau_theta']
    sigma = params['sigma']
    filtering = params['filtering'] == 1
    dt = params['substrate']['dt']
    tail_len = params['filter']['tail_len']
    deadzone = params['filter']['deadzone']
    i = t / dt
    npoints = int(np.floor(tail_len / dt))

    x1 = 0
    ampl = 0
    ptp = 0
    if filtering and i > npoints:
        b, _ = utils.make_filter(params)
        for p in substrate.populations:
            if p.order == 0:
                # TODO: Automatically count stn-like populations
                npop = 2
                x1 += p.get_tail(npoints) / npop
        measured_state = x1[-1]
        x1 = np.convolve(x1, b, mode='valid')
        ampl = x1[-1]
        ptp = np.ptp(x1)
        if ptp < deadzone:
            x1 = 0
        else:
            # x1 = x1[-1]
            x1 = ptp
    else:
        for p in substrate.populations:
            if p.order == 0:
                # TODO: Automatically count stn-like populations
                npop = 2
                x1 += p.get_tail(npoints) / npop
        measured_state = x1[-1] if len(x1) > 0 else 0
        x1 = measured_state

    dtheta = substrate.dt / tau_theta * (abs(x1) - sigma * theta)
    if feedback and t > params['feedback_start_time']:
        theta += dtheta

    return theta, ampl, measured_state, ptp


def run_simulation(substrate, params, fields):
    theta_history = np.zeros(substrate.tt.shape)
    ctx_history = [np.zeros(substrate.tt.shape) for f in fields]
    ampl_history = np.zeros(substrate.tt.shape)
    measured_state_history = np.zeros(substrate.tt.shape)
    ptp_history = np.zeros(substrate.tt.shape)
    theta0 = params['theta0']
    dt = params['substrate']['dt']
    feedback = params['feedback'] == 1
    feedback_start_time = params['feedback_start_time']

    theta = theta0
    for i, t in enumerate(substrate.tt):
        if i % 200 == 0:
            print('Time: {} ms'.format(t))

        theta, ampl, measured_state, ptp = update_feedback_gain(t, params, substrate, theta)
        for fi, field in enumerate(fields):
            simulation_step(i, t, field, ctx_history[fi], dt, feedback_start_time,
                            theta, feedback, field["c"])
        # TODO: save history in a separate function
        ampl_history[i] = ampl
        theta_history[i] = theta
        measured_state_history[i] = measured_state
        ptp_history[i] = ptp

    populations = substrate.populations
    w = fields[0]['w']

    return populations, w, theta_history, ctx_history, ampl_history, measured_state_history, ptp_history


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
    plot_filter = False
    average_feedback = False

    params['average_feedback'] = average_feedback
    b, ntaps = utils.make_filter(params)
    print('Filter order: {}'.format(ntaps))
    if plot_filter:
        utils.plot_filter_specs(b, params['substrate']['dt'], params['filter']['lowcut'], params['filter']['highcut'])
    params['filter']['ntaps'] = ntaps

    substrate = Substrate1D(params['substrate'], max_delta)

    fields = [
        add_field(f, substrate, params)
        for f in params['fields']
    ]

    populations, w, theta_history, ctx_history, ampl_history, measured_state_history, ptp_history = \
        run_simulation(substrate, params, fields)

    print("Simulation finished")

    if plot_connectivity:
        utils.plot_connectivity(w)
    utils.plot_simulation_results(populations, substrate, theta_history,
                                  ctx_history, ampl_history, measured_state_history, ptp_history, params)
    utils.save_simulation_results(populations, theta_history, ctx_history, ampl_history, measured_state_history,
                                  ptp_history, params)
