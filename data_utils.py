from scipy.spatial.transform import Rotation
import os
import pandas as pd
import numpy as np
import time
import MetarHandler
import tensorflow as tf
import datetime
import urllib.request
import zipfile
import ruptures as rpt


def get_dataset(directory='./'):
    # cleanup the save path
    directory = os.path.join(directory, '')
    save_path = '{}data.zip'.format(directory)
    # dropbox path to the zip file
    # url = 'https://www.dropbox.com/s/spoa84u8lb4xwe2/data.zip?dl=1'
    url = 'https://www.dropbox.com/s/vt5lbes7ojolto2/data_min.zip?dl=1'
    urllib.request.urlretrieve(url, save_path)
    # unzip the file; the data will be in a folder called `data`
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(directory)

def get_random_flights():
    random_flights = np.array(list(range(268, 274)) + list(range(276, 280)))
    return random_flights

def calc_aoa(row, degrees=False, index=1):
    r = Rotation.from_quat([row.x, row.y, row.z, row.w])
    return r.as_euler('xyz',
                      degrees=degrees)[index]  #rotated 90 from ENU (NWU)


def read_flight_sheet(directory):
    fname = '{}/Flight Sheet.xlsx'.format(directory)
    names = [
        "number", "route", "date", "time", "alt", "spd", "payload", "battery",
        "completed", "notes", "acc"
    ]
    cols = [0, 1, 3, 4, 5, 6, 7, 11, 13, 15, 16]
    flight_sheet = pd.read_excel(fname,
                                 header=0,
                                 names=names,
                                 usecols=cols,
                                 parse_dates={'datetime': ["date", "time"]})
    flight_sheet = flight_sheet.drop(
        flight_sheet[~flight_sheet.route.isin([5, 'New long route', 'Test'])].
        index)
    flight_sheet = flight_sheet.drop(
        flight_sheet[(flight_sheet.completed == "no")].index)
    flight_sheet = flight_sheet.drop(
        flight_sheet[(flight_sheet.acc == "no")].index)
    flight_sheet = flight_sheet.drop(flight_sheet[flight_sheet.spd > 12].index)
    flight_sheet["payload"] = flight_sheet["payload"] / 1000
    flight_sheet.datetime = pd.to_datetime(flight_sheet.datetime)

    return flight_sheet


# Read data sheets and save/load them
def load_all_data(directory, flight_sheet, mode='load'):
    '''
    mode ('load'/'reload'): default 'load'
        - 'load': defaults to loading the data for a flight and
                  calculates the data only for flights which
                  don't already have `processed.csv`
        - 'reload': forcibly calculates data for flight and saves 
                    them
    '''
    assert mode in ['load', 'reload']

    currentDate = None
    currentDensity = None

    col_names = {
        'time': 'time',
        'wind_speed': 'airspeed',
        'wind_angle': 'psi',
        'battery_voltage': 'voltage',
        'battery_current': 'current',
        'z': 'altitude',
        'x.1': 'x',
        'y.1': 'y',
        'z.1': 'z',
        'w': 'w',
        'z.2': 'vertspd'
    }
    all_data_list = {}

    if mode == 'load':
        load_flag = True
    else:
        load_flag = False

    for index, flight in flight_sheet.iterrows():

        save_fname = '{}/{}/processed.csv'.format(directory, flight.number)
        if os.path.exists(save_fname) and load_flag:
            # print("Using pre-existing data for flight {}".format(flight.number))
            flight_data = pd.read_csv(save_fname, header=0)
        else:
            if flight.datetime.date() != currentDate:
                currentDate = flight.datetime.date()
                currentDensity = MetarHandler.calculate_density(
                    flight.datetime)
                # print(currentDate, "  Flight number: ", flight.number)
            fname = '{}/{}/combined.csv'.format(directory, flight.number)
            flight_data_raw = pd.read_csv(fname, header=0)
            flight_data_raw = flight_data_raw[list(col_names.keys())]
            flight_data_raw.rename(columns=col_names, inplace=True)
            flight_data = pd.DataFrame()
            flight_data["time"] = flight_data_raw.time
            flight_data["airspeed"] = flight_data_raw.airspeed
            flight_data["vertspd"] = flight_data_raw.vertspd
            flight_data["psi"] = -1 * np.deg2rad(
                flight_data_raw.psi)  # wind_angle NED
            flight_data["aoa"] = flight_data_raw.apply(
                lambda row: calc_aoa(row, False, 0), axis=1)  #radians
            flight_data["theta"] = flight_data_raw.apply(
                lambda row: calc_aoa(row, False, 2), axis=1)  # radians
            flight_data[
                "diffalt"] = flight_data_raw.altitude - flight_data_raw[
                    "altitude"].values[0]
            flight_data["density"] = [currentDensity
                                      ] * flight_data_raw.time.count()
            flight_data["payload"] = [flight.payload
                                      ] * flight_data_raw.time.count()
            flight_data["power"] = flight_data_raw.apply(
                lambda row: (row.voltage * row.current), axis=1)
            flight_data["airspeed_x"] = flight_data["airspeed"] * np.cos(
                flight_data["psi"] - flight_data["theta"])
            flight_data["airspeed_y"] = flight_data["airspeed"] * np.sin(
                flight_data["psi"] - flight_data["theta"])
            # added a line due to the data process
            flight_data = flight_data[(flight_data.diffalt) > 7]
            flight_data.to_csv(save_fname, index=False)

        all_data_list.update({flight.number: flight_data})

    # repair the all data payload problem
    for flight, data in all_data_list.items():
        data.payload = data.apply(lambda row: row.payload / 1000
                                  if row.payload > 1 else row.payload,
                                  axis=1)

    return all_data_list


# ref: https://www.tensorflow.org/tutorials/structured_data/time_series
def multivariate_data(dataset,
                      target,
                      start_index,
                      end_index,
                      history_size,
                      target_size,
                      step,
                      single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])

    return np.array(data), np.array(labels)


def get_flights_and_ranges(data,
                           val_split=0.25,
                           test_split=0.2,
                           random_seed=42,
                           eval_mode=False,
                           **kwargs):
    
    if 'exclude_flights' not in kwargs:
        kwargs['exclude_flights'] = {}
    elif not isinstance(kwargs['exclude_flights'], set):
        kwargs['exclude_flights'] = set(kwargs['exclude_flights'])
    
    # find the number fo flights
    total_flights = list(
        set(data.keys()).difference(kwargs['exclude_flights']))
    num_total_flights = len(total_flights)
    
    # if `eval` mode, return all flights
    if eval_mode == True:
        num_test_flights = num_val_flights = num_train_flights = num_total_flights
        test_range = val_range = train_range = total_flights
        return num_test_flights, num_val_flights, num_train_flights, test_range, val_range, train_range
    
    
    num_test_flights = int(np.floor(num_total_flights * test_split))
    num_val_flights = int(
        np.ceil((num_total_flights - num_test_flights) * val_split))
    num_train_flights = num_total_flights - num_test_flights - num_val_flights

    # this test range is fixed for us, may want to take it out as a parameter
    test_range = np.array(list(range(268, 274)) + list(range(276, 280)))
    cur_num_test_flights = len(test_range)
    np.random.seed(random_seed)
    test_range = np.concatenate(
        (test_range,
         np.random.choice(total_flights,
                          size=num_test_flights - cur_num_test_flights,
                          replace=False)))
    total_flights = list(set(total_flights).difference(set(test_range)))

    # validation set
    np.random.seed(random_seed)
    val_range = np.random.choice(total_flights,
                                 size=num_val_flights,
                                 replace=False)
    total_flights = list(set(total_flights).difference(set(val_range)))

    # training set
    train_range = np.array(total_flights)

    return num_test_flights, num_val_flights, num_train_flights, test_range, val_range, train_range


def process_data(all_data,
                 lookback,
                 eval_mode=False,
                 val_split=0.25,
                 normalize=True,
                 auto_reg=False,
                 tv=0,
                 test_split=0.2,
                 random_seed=42,
                 **kwargs):
    # the normalize flag is just for the `power`. Input variables get normalized by default
    # the auto_reg flag includes the power in the input
    # this returns all the features, time variant or invariant
    # tv is the target value. 0 predicts one step into the future, -1 predicts 'in-step'

    num_test_flights, num_val_flights, num_train_flights, test_range, val_range, train_range = get_flights_and_ranges(
        all_data,
        val_split=val_split,
        test_split=test_split,
        eval_mode=eval_mode,
        random_seed=42,
        **kwargs)

    # normalize only some of the parts (payload and density never get normalized)
    if normalize:
        normalize_cols = [0, 1, 2, 3, 4, 7]
    else:
        normalize_cols = [0, 1, 2, 3, 4]

    dataset = np.concatenate([
        all_data[flight][[
            'airspeed_x', 'airspeed_y', 'vertspd', 'aoa', 'airspeed',
            'density', 'payload', 'power'
        ]] for flight in np.hstack((train_range, val_range))
    ])

    data_min = dataset[:, normalize_cols].min(axis=0)
    data_max = dataset[:, normalize_cols].max(axis=0)

    data = {}

    for flight, features in all_data.items():

        # get the data
        # removed power, added components of airspeed
        features = features[[
            'airspeed_x', 'airspeed_y', 'vertspd', 'aoa', 'airspeed',
            'density', 'payload', 'power'
        ]]
        dataset = features.values

        dataset[:, normalize_cols] = (dataset[:, normalize_cols] -
                                      data_min) / (data_max - data_min)

        # attempt to remove the zero value
        # dataset = dataset[dataset[:,-1] != 0.]

        # get x and y based on values
        start_index = 0  # where to start using data from
        end_index = None  # where to end using data from
        past_history = lookback  # lookback period
        future_target = tv  # how far in the future we want to predict
        step = 1  # rate of sampling
        single_step = True  # single prediction or sequence
        if auto_reg:
            x_in = dataset
        else:
            x_in = dataset[:, :-1]
        x, y = multivariate_data(x_in, dataset[:, -1], start_index, end_index,
                                 past_history, future_target, step,
                                 single_step)

        data[flight] = (x, y)

    return data, data_min, data_max, test_range, train_range, val_range


def create_tensors(data,
                   flight_range,
                   batch_size=32,
                   input_type='concat',
                   drop_remainder=True,
                   **kwargs):

    # model strategy is one of following
    assert input_type in ['mixed', 'concat']

    result = {}

    varying_cols = [0, 1, 2, 3, 4]
    fixed_cols = [5, 6]

    for flight in flight_range:

        if len(data[flight][0].shape) == 3:

            if input_type == 'concat':
                x = {'time_varying': data[flight][0][:, :, varying_cols]}
            if input_type == 'mixed':
                x = {
                    'time_varying':
                    data[flight][0][:, :, varying_cols + fixed_cols]
                }

            x['time_invariant'] = np.hstack([
                np.unique(data[flight][0][:, :, col], axis=1)
                for col in fixed_cols
            ])
            y = {'power_reg': data[flight][1]}

            result[flight] = tf.data.Dataset.from_tensor_slices(
                (x, y)).batch(batch_size, drop_remainder=drop_remainder)

    return result

def find_regimes(data, first=19):
    # find the break points for each flight
    change_points = {}

    # for idx, flight in enumerate(test_range):
    for idx, flight in enumerate(list(data.keys())):

        theta = data[flight]['theta'].values[first:]
        algo = rpt.Pelt(model="l2").fit(theta)
        result = algo.predict(pen=5)
        change_points[flight] = [0] + result
    
    return change_points

def load_dataset(directory):
    flight_sheet = read_flight_sheet(directory)
    all_data = load_all_data(directory, flight_sheet, 'load')
    
    return all_data

def sim_to_network_transform(flight_data, payload=0, air_density=1.1718938453052181):
    '''
    This function returns the flight states df with x and y componenets of airspeed
    It is to be used with the flight states df generated by QuadSim
    
    Parameters:
        flight_data (pd.DataFrame): Contains the states of the flight
    '''
    # need to check the 'row' in `calc_aoa` has the x, y, z, and w values it needs to work
    # flight_data["theta"] = (-1 * flight_data.apply(lambda row: calc_aoa(row, True, 0), axis=1)) % 360
    flight_data["airspeed_x"] = flight_data["airspeed"] * np.cos(np.deg2rad(flight_data["airspeed_angle"] - flight_data["heading"]))
    flight_data["airspeed_y"] = flight_data["airspeed"] * np.sin(np.deg2rad(flight_data["airspeed_angle"] - flight_data["heading"]))
    flight_data["power"] = [-1.62551003, 954.7004077] + [0]*(len(flight_data["airspeed"])-2) # add in max and min
    
    return flight_data