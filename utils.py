import datetime
import pathlib
import pickle

import matplotlib.pyplot as py
import numpy as np
from scipy import signal


def plot_simulation_results(populations, theta_history, ctx_history, ampl_history, params):
    py.figure()
    py.subplot(211)
    populations['stn'].plot_history(False)
    py.legend([str(x) for x in populations['stn'].substrate_grid])
    py.subplot(212)
    populations['stn'].plot_history_average(False)
    cutoff = params['filter_cutoff']
    order = params['filter_order']
    fs = 1000 / populations['stn'].substrate.dt
    nyq = fs / 2

    cutoff_norm = cutoff / nyq
    b, a = signal.butter(order, cutoff_norm)
    sig = np.mean(populations['stn'].history, axis=1)
    filtered_signal = signal.lfilter(b, a, sig)
    py.plot(populations['stn'].substrate.tt, filtered_signal)
    py.legend(['stn avg', 'x1_filtered'])
    py.ylim([-150, 150])
    py.figure()
    py.subplot(211)
    populations['gpe'].plot_history(False)
    py.legend([str(x) for x in populations['gpe'].substrate_grid])
    py.subplot(212)
    populations['gpe'].plot_history_average(False)
    py.legend(['gpe avg'])
    py.figure()
    py.plot(populations['stn'].substrate.tt, theta_history)
    py.legend(['theta'])
    py.figure()
    py.plot(populations['stn'].substrate.tt, ctx_history)
    py.legend(['ctx'])
    py.figure()
    py.plot(populations['stn'].substrate.tt, ampl_history)
    py.legend(['ampl'])


    py.show()


def plot_connectivity(w, show=False):
    # py.figure()
    # wtotal = np.array([[g21(r2, r1, populations['stn'].mu, populations['gpe'].mu, params) for r2 in np.arange(0, 15 + dx, dx)]
    #                               for r1 in np.arange(0, 15 + dx, dx)])
    # py.imshow(wtotal)
    #
    # py.figure()

    wmin = np.min([np.min(ww) for ww in w.values()])
    wmax = np.max([np.max(ww) for ww in w.values()])

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

    if show:
        py.show()


def save_simulation_results(populations, theta_history, config):
    path = pathlib.Path('simulation_results')
    if not path.exists():
        path.mkdir()
    file = 'dnf_results_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = path / file
    result = {
        'populations': populations,
        'theta_history': theta_history,
        'config': config
    }
    with filename.open(mode='wb') as f:
        pickle.dump(result, f)


def plot_steady_state_amplitude_vs_parameter(tag, param):
    results = []
    path = pathlib.Path('simulation_results')
    for file in path.iterdir():
        with file.open('rb') as f:
            res = pickle.load(f)
            if tag in res['config']['tags']:
                par = res['config'][param]
                ampl = res['populations']['stn'].get_tail(1000).ptp()
                results.append((par, ampl))
    results.sort(key=lambda tup: tup[0])
    p, a = zip(*results)
    py.plot(p, a, 'bo-')
    py.ylabel('Steady-state amplitude (peak-to-peak)')
    py.xlabel(param)
    py.show()


def plot_filter_specs(dt, cutoff, order):
    fs = 1000 / dt
    nyq = fs / 2

    cutoff_norm = cutoff / nyq

    b, a = signal.butter(order, cutoff_norm)
    w, h = signal.freqz(b, a)
    f = w * nyq / np.pi

    py.figure()
    py.subplot(211)
    py.plot(f, np.abs(h))
    py.xlim([0, 40])
    py.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    py.title('Amplitude response')
    py.ylabel('Relative amplitude')
    py.subplot(212)
    py.plot(f, 180 * np.angle(h) / np.pi)
    py.xlim([0, 40])
    py.title('Phase response')
    py.xlabel('Frequency [Hz]')
    py.show()
