import operator
import pickle
import spectrum
import json

import matplotlib.pyplot as py
import numpy as np

from population import Population1D
from substrate import Substrate1D


def g12(r1, r2, mu1, mu2, params):
    x = abs(abs(r1 - mu1) - abs(r2 - mu2))
    return np.float(-params["K12"] * np.exp(-x ** 2 / (2 * params["sigma12"])))


def g22(r1, r2, mu1, mu2, params):
    x = abs(r1 - r2)
    return np.float(-abs(r1 - r2) * params["K22"] * np.exp(-x ** 2 / (2 * params["sigma22"])))


def g21(r1, r2, mu1, mu2, params):
    x = abs(abs(r1 - mu1) - abs(r2 - mu2))
    return np.float(params["K21"] * np.exp(-x ** 2 / (2 * params["sigma21"])))


def get_connectivity(kernel, key, column, shape):
    if key in kernel:
        return kernel[key][column, :]
    else:
        return np.zeros(shape)



if __name__ == "__main__":
    print('Delayed Neural Fields - freq test')
    f = open('simulation_params_2_populations', 'r')
    params = json.load(f)

    for k22 in np.arange(80, 100, 1):

        params['K22'] = k22

        # TODO: Calculate max delta
        max_delta = 20
        dx = params["substrate"]["dx"]

        substrate = Substrate1D(params['substrate'], max_delta)

        populations = {name: Population1D(name, params['populations'][name], substrate) for name in params['populations']}

        w = dict()
        w[('stn', 'gpe')] = np.array([[g12(r1, r2, populations['stn'].mu, populations['gpe'].mu, params) for r2 in populations['gpe'].substrate_grid]
                                      for r1 in populations['stn'].substrate_grid])
        w[('gpe', 'stn')] = np.array([[g21(r1, r2, populations['gpe'].mu, populations['stn'].mu, params) for r2 in populations['stn'].substrate_grid]
                                      for r1 in populations['gpe'].substrate_grid])
        w[('gpe', 'gpe')] = np.array([[g22(r1, r2, populations['gpe'].mu, populations['gpe'].mu, params) for r2 in populations['gpe'].substrate_grid]
                                      for r1 in populations['gpe'].substrate_grid])
        if 'stn2' in populations.keys():
            w[('stn2', 'gpe2')] = np.array([[g12(r1, r2, populations['stn2'].mu, populations['gpe2'].mu, params) for r2 in populations['gpe2'].substrate_grid]
                                            for r1 in populations['stn2'].substrate_grid])
            w[('gpe2', 'stn2')] = 0.02 * np.array([[g21(r1, r2, populations['gpe2'].mu, populations['stn2'].mu, params) for r2 in populations['stn2'].substrate_grid]
                                            for r1 in populations['gpe2'].substrate_grid])
            w[('gpe2', 'gpe2')] = np.array([[g22(r1, r2, populations['gpe2'].mu, populations['gpe2'].mu, params) for r2 in populations['gpe2'].substrate_grid]
                                            for r1 in populations['gpe2'].substrate_grid])

        for i, t in enumerate(substrate.tt):
            states = {p.name: p.last_state() for p in populations.values()}
            if t > 0:
                inputs = dict()
                for pop in populations.keys():
                    inputs[pop] = np.array([np.sum([np.dot(get_connectivity(w, (pop, p), ri, states[p].shape),
                                                           populations[p].delayed_activity(r, t))
                                                    for p in populations.keys()]) + populations[pop].external_input(t)
                                            for ri, r in enumerate(populations[pop].substrate_grid)]) + \
                                  populations[pop].control(t)
                for p in populations.keys():
                    states[p] += substrate.dt/populations[p].tau * (-states[p] + populations[p].sigmoid(inputs[p]))
            for p in populations.keys():
                populations[p].update_state(t, states[p])

        print("simulation finished")

        py.figure()
        populations['stn'].plot_history_average(False)
        populations['gpe'].plot_history_average(False)
        py.plot(populations['stn'].tt, np.mean(populations['stn'].control.history, axis=1))
        if 'stn2' in populations.keys():
            populations['stn2'].plot_history_average(False)
            populations['gpe2'].plot_history_average(False)
            py.legend(['stn', 'gpe', 'theta', 'stn2', 'gpe2'])
        else:
            py.legend(['stn', 'gpe', 'theta'])
        py.savefig("".join(['exploration/plots/k22-', str(k22), '-average.png']))

        py.figure()
        average = np.mean([np.mean(p.history, axis=1) for p in populations.values()], axis=0)
        py.subplot(2, 1, 1)
        py.plot(substrate.tt, average)
        py.xlim([0, 5000])
        py.subplot(2, 1, 2)
        # Fs = 1000 / substrate.dt
        # Pxx, freqs, bins, im = py.specgram(average, NFFT=512, Fs=Fs, noverlap=100)
        Pxx = spectrum.Periodogram(average[-1500:], 1000, detrend='mean')
        Pxx.run()
        Pxx.plot(marker='o')
        # py.show()
        py.savefig("".join(['exploration/plots/k22-', str(k22), '-spectrum.png']))

        try:
            with open('exploration/k22-dominant-freq.pickle', 'rb') as f:
                freqs = pickle.load(f)
        except (FileNotFoundError, EOFError):
            freqs = dict()

        max_index, _ = max(enumerate(Pxx.psd[5:]), key=operator.itemgetter(1))
        freqs[k22] = Pxx.frequencies()[max_index+5]

        with open('exploration/k22-dominant-freq.pickle', 'wb') as f:
            pickle.dump(freqs, f)

        py.close('all')
