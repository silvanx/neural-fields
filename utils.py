import datetime
import pathlib
import pickle

import matplotlib
import matplotlib.pyplot as py
import numpy as np
from scipy import signal


def plot_simulation_results(populations, substrate, theta_history, ctx_history, str_history,
                            ampl_history, measured_state_history, ptp_history, params):
    for p in populations:
        py.figure()
        py.subplot(211)
        p.plot_history(False)
        py.legend([str(x) for x in p.substrate_grid])
        py.subplot(212)
        p.plot_history_average(False)
        py.legend(['{} avg'.format(p.name)])
    # py.figure()
    # py.plot(substrate.tt, theta_history)
    # py.legend(['theta'])
    py.figure()
    for ch in ctx_history:
        py.plot(substrate.tt, ch)
    py.legend(['ctx'])
    py.figure()
    for sh in str_history:
        py.plot(substrate.tt, sh)
    py.legend(['str'])
    py.figure()
    plot_filter_comparison(populations, substrate, params, ampl_history, measured_state_history, ptp_history,
                           theta_history)
    # if params['feedback'] == 0:
    #     plot_fft(measured_state_history, params)
    # else:
    fs = 1000 / params['substrate']['dt']
    py.figure(figsize=(13.84, 4.83))
    py.subplot(211)
    f, t, Sxx = signal.spectrogram(populations[0].history.mean(axis=1), fs, detrend='linear', nfft=600)
    py.pcolormesh(t, f, Sxx)
    py.colorbar()
    py.ylim([0, 150])
    py.xlabel('Time [s]')
    py.ylabel('Frequency [Hz]')
    py.title('STN')
    py.subplot(212)
    f, t, Sxx = signal.spectrogram(populations[1].history.mean(axis=1), fs, detrend='linear', nfft=600)
    py.pcolormesh(t, f, Sxx)
    py.colorbar()
    py.ylim([0, 150])
    py.xlabel('Time [s]')
    py.ylabel('Frequency [Hz]')
    py.title('GPe')
    py.tight_layout()
    py.show()


def plot_fft(measured_state_history, params):
    py.figure()
    n = len(measured_state_history)
    k = np.arange(n)
    fs = 1000 / params['substrate']['dt']
    T = n / fs
    frq = k / T
    frq = frq[range(int(n / 2))]
    Y = np.fft.fft(measured_state_history) / n
    Y = Y[range(int(n / 2))]
    py.plot(frq, abs(Y), 'r')
    py.xlabel('Freq (Hz)')
    py.ylabel('|FFT(freq)|')


def plot_filter_comparison(populations, substrate, params, ampl_history, measured_state_history, ptp_history,
                           theta_history, show=False):
    py.figure(figsize=(19.0988, 10))
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    py.subplot(311)
    py.ylabel('Activity [spk/s]', fontsize=13)
    measured_signal = 0
    for p in populations:
        if p.order == 0:
            measured_signal += np.mean(p.history, axis=1)
    py.plot(substrate.tt, measured_signal)
    # py.plot(substrate.tt, measured_state_history)
    py.legend(['Measured signal'], fontsize=13)
    py.subplot(312)
    py.ylabel('Activity [spk/s]', fontsize=13)
    # b, _ = make_filter(params)
    # filtered_signal = np.convolve(measured_state_history, b, mode='valid')
    # py.plot(substrate.tt[params['filter']['ntaps'] - 1:], filtered_signal)
    py.plot(substrate.tt, ampl_history)
    py.legend(['Filtered signal'], fontsize=13)
    # py.legend(['Filtered signal', 'Live filtered signal'], fontsize=13)
    # py.title('Comparison of filtering, window length = {}, taps = {}'.format(params['filter']['tail_len'],
    #                                                                          params['filter']['ntaps']), fontsize=13)
    py.title('Bandpass-filtered signal')
    py.subplot(313)
    py.plot(substrate.tt, ptp_history, substrate.tt, theta_history)
    py.legend(['Running peak-to-peak amplitude of filtered signal', 'Theta'], fontsize=13)
    py.xlabel('Time [ms]', fontsize=13)
    if show:
        py.show()


def plot_connectivity(w, show=False):

    wmin = np.min([np.min(ww) for ww in w.values()])
    wmax = np.max([np.max(ww) for ww in w.values()])

    fig = py.figure(figsize=(15, 7.5))
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    ax1 = py.subplot2grid((2, 2), (0, 1))
    ax2 = py.subplot2grid((2, 2), (1, 0))
    ax3 = py.subplot2grid((2, 2), (1, 1))

    ax1.imshow(w[('stn', 'gpe')][::-1], cmap='RdBu', vmin=wmin, vmax=wmax, extent=[12.5, 15, 2.5, 0])
    ax1.set_title('STN-GPe', fontsize=13)
    ax2.imshow(w[('gpe', 'stn')][::-1], cmap='RdBu', vmin=wmin, vmax=wmax, extent=[0, 2.5, 15, 12.5])
    ax2.set_title('GPe-STN', fontsize=13)
    im = ax3.imshow(w[('gpe', 'gpe')], cmap='RdBu', vmin=wmin, vmax=wmax, extent=[12.5, 15, 12.5, 15])
    ax3.set_title('GPe-GPe', fontsize=13)
    fig.subplots_adjust(right=0.85, left=0.01)

    cbar_ax = fig.add_axes([0.90, 0.05, 0.05, 0.9])
    fig.colorbar(im, cax=cbar_ax)
    # py.colorbar()

    if show:
        py.show()


def save_simulation_results(populations, theta_history, ctx_history, ampl_history, measured_state_history,
                            ptp_history, config):
    path = pathlib.Path('simulation_results')
    if not path.exists():
        path.mkdir()
    date_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file = 'dnf_results_' + date_string
    filename = path / file
    result = {
        'populations': populations,
        'theta_history': theta_history,
        'ctx_history': ctx_history,
        'ampl_history': ampl_history,
        'measured_state_history': measured_state_history,
        'ptp_history': ptp_history,
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

    py.figure(figsize=(19.0988, 10))
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    # py.figure()
    # py.subplot(211)
    py.plot(w / max(w) * nyq, h_dB)
    # py.xlim([0, 40])
    py.title('Amplitude response, filter order: {}'.format(len(b) - 1), fontsize=13)
    py.ylabel('Relative amplitude [dB]', fontsize=-13)
    py.xlabel('Frequency [Hz]', fontsize=13)
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
    # ntaps = params['filter']['ntaps']
    lowcut = params['filter']['lowcut']
    highcut = params['filter']['highcut']
    ntaps, beta = signal.kaiserord(40, 8 / nyq)
    b = signal.firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                      window=('kaiser', beta), scale=False)
    return b, ntaps


def plot_filter_thing():
    path = pathlib.Path('simulation_results')

    stn_no_feedback = []
    gpe_no_feedback = []
    stn_feedback = []
    gpe_feedback = []
    for file in path.iterdir():
        if file.match('*_2018-06-06-*'):
            with file.open('rb') as f:
                res = pickle.load(f)
                params = res['config']
                freq = params['fields'][0]['cortex']['ctx_entrainment_frequency']
                dt = params['substrate']['dt']
                tags = params.get('tags')
                if tags and 'entrainment_freq_exploration' in tags:
                    stn_no_feedback.append((freq, res['populations'][0].tail_amplitude(int(1000 / dt))))
                    gpe_no_feedback.append((freq, res['populations'][1].tail_amplitude(int(1000 / dt))))
                elif tags and 'entrainment_freq_exploration_feedback_2' in tags:
                    stn_feedback.append((freq, res['populations'][0].tail_amplitude(int(1000 / dt))))
                    gpe_feedback.append((freq, res['populations'][1].tail_amplitude(int(1000 / dt))))
    stn_no_feedback.sort(key=lambda tup: tup[0])
    stn_feedback.sort(key=lambda tup: tup[0])
    gpe_no_feedback.sort(key=lambda tup: tup[0])
    gpe_feedback.sort(key=lambda tup: tup[0])

    py.subplots(2, 1, figsize=(16, 12))
    py.title('Effect of feedback stimulation')
    py.subplot(211)
    p1, a1 = zip(*stn_no_feedback)
    py.plot(p1, a1, 'b-')
    p2, a2 = zip(*stn_feedback)
    py.plot(p2, a2, 'r-')
    # py.xlabel('Frequency [Hz]')
    py.ylabel('Amplitude')
    py.legend(['no feedback', 'feedback'])
    py.title('STN')
    # py.figure()
    py.subplot(212)
    p3, a3 = zip(*gpe_no_feedback)
    py.plot(p3, a3, 'b-')
    p4, a4 = zip(*gpe_feedback)
    py.plot(p4, a4, 'r-')
    py.xlabel('Frequency [Hz]')
    py.ylabel('Amplitude')
    py.title('GPe')
    py.show()
