from functools import reduce

import matplotlib
import matplotlib.pyplot as py
from scipy.signal import butter, spectrogram

import utils


def unpack_simulation_results(results):
    config = results.get('config')
    populations = results.get('populations')
    ampl_history = results.get('ampl_history')
    ctx_history = results.get('ctx_history')
    measured_state_history = results.get('measured_state_history')
    ptp_history = results.get('ptp_history')
    theta_history = results.get('theta_history')

    return config, populations, ampl_history, ctx_history, measured_state_history, ptp_history, theta_history


def count_populations_type(populations):
    stn = []
    gpe = []
    for i, p in enumerate(populations):
        if p.order == 0:
            stn.append(i)
        else:
            gpe.append(i)

    return stn, gpe


def plot_population_activity(history, labels, tt, title):
    py.figure(figsize=(14.3241, 2.5))
    py.title(title)
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)

    fs = 1000 / populations[0].dt
    b, a = butter_highpass(0.1, fs)
    # history = filtfilt(b, a, history, axis=0)

    py.imshow(history.transpose(), aspect='auto', cmap='bwr', extent=(tt[0], tt[-1], len(labels), 0))
    # ylocs, _ = py.yticks()
    # py.yticks(ylocs, ['%.2f' % round(n, 2) for n in labels])
    py.yticks([])
    py.xlabel('Time [ms]')
    # py.ylabel('Position [mm]')
    py.colorbar()


def butter_highpass(cutoff, fs, order=7):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def plot_specgram(history, dt, title):
    fs = 1000 / dt
    avg_hist = history[:, 0]
    b, a = butter_highpass(2, fs)
    # avg_hist = filtfilt(b, a, avg_hist)
    fig = py.figure(figsize=(14.3241, 2.5))
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    # Pxx, freqs, bins, im = py.specgram(avg_hist, NFFT=750, Fs=fs, noverlap=500, cmap='rainbow')
    f, t, Sxx = spectrogram(avg_hist, fs, detrend='linear', nfft=600)
    py.pcolormesh(t, f, Sxx)
    py.colorbar()
    py.ylim([0, 150])
    py.title(title)
    py.ylabel('Frequency [Hz]')
    py.xlabel('Time [s]')


if __name__ == "__main__":
    filename = "dnf_results_2018-06-01-16-30-38"  # two fields, endogenous oscillations
    results = utils.load_simulation_results(filename)
    config, populations, ampl_history, ctx_history, measured_state_history, ptp_history, theta_history = \
        unpack_simulation_results(results)
    stn, gpe = count_populations_type(populations)

    py.figure(figsize=(14.3241, 2.5))
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    # py.plot(populations[stn[0]].tt, 1000 - ctx_history[0])
    py.fill_between(populations[stn[0]].tt, 1000 - ctx_history[0], alpha=0.5)
    py.yticks([])
    # py.plot(populations[stn[0]].tt, 1000 - ctx_history[1])
    py.fill_between(populations[stn[0]].tt, 1000 - ctx_history[1], alpha=0.5)
    py.legend(['beta', 'gamma'])
    py.yticks([])
    py.savefig('plots_antoine/beta_gamma', dpi=150, bbox_inches='tight')
    py.close()

    stn_history = reduce((lambda x, y: (x + y) / len(stn)), [populations[i].history for i in stn])
    gpe_history = reduce((lambda x, y: (x + y) / len(gpe)), [populations[i].history for i in gpe])
    plot_population_activity(stn_history, populations[stn[0]].substrate_grid, populations[stn[0]].tt, 'STN')
    py.savefig('plots_antoine/plot_stn', dpi=150, bbox_inches='tight')
    py.close()
    plot_population_activity(gpe_history, populations[gpe[0]].substrate_grid, populations[gpe[0]].tt, 'GPe')
    py.savefig('plots_antoine/plot_gpe', dpi=150, bbox_inches='tight')
    py.close()

    plot_specgram(stn_history, populations[0].dt, 'STN')
    py.savefig('plots_antoine/specgram_stn', dpi=150, bbox_inches='tight')
    py.close()
    plot_specgram(gpe_history, populations[0].dt, 'GPe')
    py.savefig('plots_antoine/specgram_gpe', dpi=150, bbox_inches='tight')
    py.close()

    py.figure(figsize=(14.3241, 2.5))
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    py.plot(populations[stn[0]].tt, theta_history)
    py.savefig('plots_antoine/plot_theta', dpi=150, bbox_inches='tight')
    py.close()
