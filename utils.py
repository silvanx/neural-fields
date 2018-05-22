import datetime
import pathlib
import pickle

import matplotlib.pyplot as py
import numpy as np
from scipy import signal


def plot_simulation_results(populations, substrate, theta_history, ctx_history,
                            ampl_history, measured_state_history, ptp_history, params):
    for p in populations:
        py.figure()
        py.subplot(211)
        p.plot_history(False)
        py.legend([str(x) for x in p.substrate_grid])
        py.subplot(212)
        p.plot_history_average(False)
        py.legend(['{} avg'.format(p.name)])
    py.figure()
    py.plot(substrate.tt, theta_history)
    py.legend(['theta'])
    py.figure()
    for ch in ctx_history:
        py.plot(substrate.tt, ch)
    py.legend(['ctx'])
    py.figure()
    plot_filter_comparison(populations, substrate, params, ampl_history, measured_state_history, ptp_history, show=True)


def plot_filter_comparison(populations, substrate, params, ampl_history, measured_state_history, ptp_history,
                           show=False):
    py.subplot(211)
    measured_signal = 0
    for p in populations:
        if p.order == 0:
            measured_signal += np.mean(p.history, axis=1) / 2
    py.plot(substrate.tt, measured_signal)
    py.plot(substrate.tt, measured_state_history, '--')
    b, _ = make_filter(params)
    filtered_signal = np.convolve(measured_signal, b, mode='valid')
    py.plot(substrate.tt[params['filter']['ntaps'] - 1:], filtered_signal)
    py.plot(substrate.tt, ampl_history, '--')
    py.legend(['Measured signal', 'Live measured signal', 'Filtered signal', 'Live filtered signal'])
    py.title('Comparison of filtering, window length = {}, taps = {}'.format(params['filter']['tail_len'],
                                                                             params['filter']['ntaps']))
    py.subplot(212)
    py.plot(substrate.tt, ptp_history)
    py.legend(['Running peak-to-peak amplitude of filtered signal'])
    if show:
        py.show()

def plot_connectivity(w, show=False):

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
    date_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file = 'dnf_results_' + date_string
    filename = path / file
    result = {
        'populations': populations,
        'theta_history': theta_history,
        'config': config
    }
    with filename.open(mode='wb') as f:
        pickle.dump(result, f)


def load_simulation_results(filename):
    path = pathlib.Path('simulation_results')
    fullpath = path / filename
    with fullpath.open(mode='rb') as f:
        results = pickle.load(f)
    return results


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


def plot_filter_specs(b, dt, lowcut=0, highcut=50):
    fs = 1000 / dt
    nyq = fs / 2

    w, h = signal.freqz(b, 1)
    h_dB = 20 * np.log10(abs(h))
    h_Phase = np.unwrap(np.arctan2(np.imag(h), np.real(h)))

    py.figure()
    # py.subplot(211)
    py.plot(w / max(w) * nyq, h_dB)
    # py.xlim([0, 40])
    py.title('Amplitude response, filter order: {}'.format(len(b) - 1))
    py.ylabel('Relative amplitude [dB]')
    py.xlabel('Frequency [Hz]')
    # py.subplot(212)
    # py.plot(w / max(w) * nyq, h_Phase)
    # py.xlim([0, 40])
    # py.title('Phase response')
    # py.xlabel('Frequency [Hz]')
    axes = py.gca()
    axes.axvspan(lowcut, highcut, color='grey', alpha=0.6)
    py.savefig('filter_specs.pdf', format='pdf', bbox_inches='tight')
    py.show()


def make_filter(params):
    dt = params['substrate']['dt']
    fs = 1000 / dt
    nyq = 0.5 * fs
    ntaps = params['filter']['ntaps']
    lowcut = params['filter']['lowcut']
    highcut = params['filter']['highcut']
    ntaps, beta = signal.kaiserord(22.0, 4 / nyq)
    b = signal.firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                      window=('kaiser', beta), scale=False)
    # window=('boxcar'), scale=False)
    # print('Num taps: {}'.format(ntaps))
    return b, ntaps
