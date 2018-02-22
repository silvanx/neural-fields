import datetime
import pickle

import matplotlib.pyplot as py


def plotFreqsPickle(filename):
    with open(filename, 'rb') as f:
        freqs = pickle.load(f)

    x, y = zip(*list(freqs.items()))
    py.scatter(x, y)
    py.show()


def plot_simulation_results(populations, theta_history):
    py.figure()
    py.subplot(211)
    populations['stn'].plot_history(False)
    py.legend([str(x) for x in populations['stn'].substrate_grid])
    py.subplot(212)
    populations['stn'].plot_history_average(False)
    py.legend(['stn avg'])
    py.figure()
    py.subplot(211)
    populations['gpe'].plot_history(False)
    py.legend([str(x) for x in populations['gpe'].substrate_grid])
    py.subplot(212)
    populations['gpe'].plot_history_average(False)
    py.legend(['gpe avg'])
    py.figure()
    py.plot(populations['stn'].substrate.tt, theta_history)

    py.show()


def save_simulation_results(populations, theta_history, config):
    filename = 'simulation_results/dnf_results_' + datetime.datetime.now().isoformat()
    result = {
        'populations': populations,
        'theta_history': theta_history,
        'config': config
    }
    with open(filename, 'wb') as f:
        pickle.dump(result, f)
