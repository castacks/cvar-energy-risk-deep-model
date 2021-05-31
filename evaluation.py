import os
from scipy.integrate import simps
import liu_model
from data_utils import *
from deep_energy_model import *
import argparse
import numpy as np
from tabulate import tabulate

def get_pre_def_configs():
    # pre-defined configs for paper
    configs = {
        'b-LSTM': {'net_type': 'lstm', 'dropout': 0., 'units': 64, 'n_hidden': 3, 'optimizer': 'RMSprop', 'stateful': True, 'batch_size': 1},
        's-LSTM': {'net_type': 'lstm', 'dropout': 0., 'units': 16, 'n_hidden': 3, 'optimizer': 'RMSprop', 'stateful': True, 'batch_size': 1},
        'b-TCN': {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'n_layers': 5, 'total_epochs': 100, 'batch_size': 1},
        's-TCN': {'nb_filters': 16, 'kernel_size': 2, 'nb_stacks': 1, 'n_layers': 5, 'total_epochs': 100, 'batch_size': 1},
        'custom': {},
        'liu': {}
    }
    return configs

def get_flights(all_data):
    _, _, _, test_range, val_range, train_range = get_flights_and_ranges(all_data)
    flight_ranges = {
        'train': train_range,
        'test': test_range,
        'val': val_range,
        'random': get_random_flights()
    }
    return flight_ranges

def calc_energy(power, ts, cp):
    """
    power (list of float): power value at each time step
    ts (list of float): time step value
    cp (list of int): starting index of each regime + last index
    NOTE: make sure that indices are matching up for all the inputs
    Returns the energy for each regime of a flight
    """
    assert len(power) == len(ts)
    assert cp[-1] == len(power)
    
    actual_power = np.zeros(len(cp)-1)
    for idx, c in enumerate(cp[:-1]):
        actual_power[idx] = simps(power[c:cp[idx+1]], x=ts[c:cp[idx+1]], even='avg')/1000
    
    return actual_power

# if the file is execeuted as the main program
if __name__ == '__main__':
    
    # get default configs
    configs = get_pre_def_configs()
    
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default='./', help="Working directory for data")
    parser.add_argument("-m", "--model", default=['b-TCN'], nargs='*', help="The model we want to calculate values for")
    parser.add_argument("-D", "--data", action="store_true", help="Whether to download the data or not")
    parser.add_argument("-e", "--evaluate", default=['test'], nargs='*', help="Which dataset to evaluate on")
    parser.add_argument("--lookback", default=20, type=int, help="Size of lookback window")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of batch passed to `tf`.")
    parser.add_argument("--nb_filters", default=32, type=int, help="(For TCN) The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.")
    parser.add_argument("--kernel_size", default=3, type=int, help="(For TCN) The size of the kernel to use in each convolutional layer.")
    parser.add_argument("--nb_stacks", default=1, type=int, help="(For TCN) The number of stacks of residual blocks to use.")
    parser.add_argument("--n_layers", default=4, type=int, help="Number of layers in the network.")
    parser.add_argument("--total_epochs", default=10, type=int, help="Total number passes over the entire training data set.")
    parser.add_argument("--optimizer", default='Adam', help="Optimizer used by the neural network.")
    parser.add_argument("--dropout", default=0., type=float, help="Dropout used by the neural network.")
    parser.add_argument("--units", default=32, type=int, help="(For LSTM) The number of units in each layer.")
    parser.add_argument("--net_type", default='tcn', choices=['lstm', 'tcn'], help="The type of net to train. Either `lstm ` or `gru`.")
    parser.add_argument("--stateful", action="store_true", help="Whether to create a stateful neural net or not.")
    parser.add_argument("-c", "--ckpt", default='./', help="Checkpoint file location. Required if loading a `custom` model.")
    args = parser.parse_args()
    
    # validate directory
    directory = os.path.join(args.directory, '')
    if not os.path.exists(directory):
        raise ValueError('Invalid path.')
    
    # validate models
    if 'all' in args.model:
        models = ['b-LSTM', 's-LSTM', 'b-TCN', 's-TCN', 'liu']
    elif set(args.model).issubset(set(configs.keys())):
        models = list(set(args.model))
    else:
        raise ValueError('Unidentified model choice(s).')
    # validate checkpoint file
    if 'custom' in models:
        ckpt_file = args.ckpt
        if ckpt_file[-5:] == '.ckpt':
            raise ValueError('Invalid checkpoint file. Path should end with .ckpt')
        elif not os.path.exists(ckpt_file):
            raise ValueError('Invalid checkpoint file path.')
        configs['custom'] = {k: v for k, v in vars(args).items() if k not in ['directory', 'model', 'data', 'evaluate', 'ckpt']}
    
    # validate evaluation data
    if 'all' in args.evaluate:
        datasets = ['train', 'test', 'val', 'random']
    elif set(args.evaluate).issubset(set(['train', 'test', 'val', 'random'])):
        datasets = list(set(args.evaluate))
    else:
        raise ValueError('Unidentified evaluation dataset choice(s).')
    
    # download dataset if needed
    if args.data:
        get_dataset(directory)
    
    # load the data
    all_data = load_dataset(directory+'data/')
    
    # place holder to store all the results
    results = pd.DataFrame(columns=['Model']+[d for d in datasets])
    
    # first let us find the true stuff so we don't calculate that again and again
    flight_ranges = get_flights(all_data)
    
    test_range = list(np.concatenate([flight_ranges[d] for d in datasets])) # superset of all the flights for results
    
    first = 19 # lookback - 1
    change_points = find_regimes(all_data)

    flight_results = {
        k: {
            'True W': all_data[k]['power'].values[first:],
            'True E': calc_energy(all_data[k]['power'].values[first:], all_data[k]['time'].values[first:], change_points[k])
        } for k in test_range
    }
    
    for model in models:
        # special case for liu model
        if 'liu' == model:
            row = {'Model': 'Liu model'}
            popt = liu_model.optimum_values()
            for flight in test_range:
                y_pred = liu_model.power(all_data[flight][['vertspd', 'airspeed','aoa','payload', 'density']].T, popt[0], popt[1], popt[2], popt[3], popt[4])[first:]

                flight_results[flight]['MAPE'] = np.mean(np.abs((flight_results[flight]['True W'] - y_pred)/flight_results[flight]['True W']))*100
                flight_results[flight]['Joule'] = np.mean(np.abs((flight_results[flight]['True E'] - calc_energy(y_pred, all_data[flight]['time'].values[first:], change_points[flight]))/flight_results[flight]['True E']))*100
            for d in datasets:
                row[d] = '{:.2f}/{:.2f}'.format(np.mean([flight_results[f]['MAPE'] for f in flight_ranges[d]]), np.mean([flight_results[f]['Joule'] for f in flight_ranges[d]]))

            results = results.append(row, ignore_index=True)
        # regular deep models
        else:
            row = {'Model': model}
            
            # for the custom model
            if model == 'custom':
                model, inputs = load_model(configs[model], ckpt_file)
            else:
                model, inputs = load_model(configs[model], model)
            data, data_min, data_max, _, _, _ = process_data(all_data, **inputs)

            test_tensors = create_tensors(data, test_range, **inputs)

            for flight in test_range:
                model.reset_states()
                y_pred = model.predict(test_tensors[flight]).reshape(-1)
                y_pred = y_pred*(data_max[-1] - data_min[-1]) + data_min[-1]

                flight_results[flight]['MAPE'] = np.mean(np.abs((flight_results[flight]['True W'] - y_pred)/flight_results[flight]['True W']))*100
                flight_results[flight]['Joule'] = np.mean(np.abs((flight_results[flight]['True E'] - calc_energy(y_pred, all_data[flight]['time'].values[first:], change_points[flight]))/flight_results[flight]['True E']))*100
            
            for d in datasets:
                row[d] = '{:.2f}/{:.2f}'.format(np.mean([flight_results[f]['MAPE'] for f in flight_ranges[d]]), np.mean([flight_results[f]['Joule'] for f in flight_ranges[d]]))
                
            results = results.append(row, ignore_index=True)
    
    print(tabulate(results, headers=results.columns, tablefmt="fancy_grid", showindex=False))



