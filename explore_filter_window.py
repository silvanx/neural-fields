import matplotlib.pyplot as py

from main import *


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
    average_feedback = False

    params['average_feedback'] = average_feedback

    for wl in range(1, 5, 1):
        params['filter']['tail_len'] = wl
        substrate = Substrate1D(params['substrate'], max_delta)

        fields = [
            add_field(f, substrate, params)
            for f in params['fields']
        ]
        print('running simulation with filter window = {}'.format(wl))
        populations, w, theta_history, ctx_history, ampl_history, measured_state_history = \
            run_simulation(substrate, params, fields)
        py.figure()
        utils.plot_filter_comparison(populations, substrate, params, ampl_history, measured_state_history, False)
        imgfile = '20180503_plots/filter_ntaps_{}_wl_{}.png'.format(params['filter']['ntaps'], params['filter']['tail_len'])
        py.savefig(imgfile, figsize=(10, 8), dpi=150)
        py.close()

    print("Simulation finished")
