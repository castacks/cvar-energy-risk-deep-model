import numpy as np
import QuadSim
from QuadSim import QuadSim
from QuadSim import QuadState
import liu_model
import evaluation
from data_utils import *
from deep_energy_model import load_model
import time
from scipy.integrate import simps
from multiprocessing import Pool
import matplotlib.pyplot as plt
import Risk



def one_flight():
    # waypoints given as x, y, z, time
    waypoints = np.array([[-259.9, -260., -260., -200.,  -65.,  -64.9],
                          [  125.,  125.,   50.,  -35.,  100.,   100.],
                          [    0.,   20.,   20.,   20.,   20.,     0.],
                          [    0.,    3.,    3.,    3.,    3.,     5.]])

    groundspeed = 10
    displayFig = True
    wind_map_ang = np.load('wind_data/dataset_hawkins.npy')
    wind_map_mag = np.load('wind_data/dataset_hawkins_mag.npy')
    inlet_ang = 180 
    inlet_mag = 2


    for i in range(1):
        testsim = QuadSim(None, waypoints[:,0], wind_map_ang, wind_map_mag, inlet_ang, inlet_mag, command_airspeed=True, ghost_delta=1.0)
        
        testsim.init_path_from_waypoints(waypoints, airspeed=groundspeed)
        start_time = time.time()
        states = testsim.propagate(displayFig)
        end_time = time.time()
        print("Time to complete sim: {} seconds".format((end_time-start_time)))
        print("Flight time: {} seconds".format(states[1]/10))

        print(states[3])

    return states

def calc_energy(power, ts):
    """
    power (list of float): power value at each time step
    ts (list of float): time step value
    NOTE: make sure that indices are matching up for all the inputs
    Returns the energy for a flight
    """
    assert len(power) == len(ts)

    actual_power = simps(power, x=ts, even='avg')/1000

    return actual_power

def energy_predictions(states_list, model='s-TCN'):
    '''
        Returns the Kilojoules predicted by the model
                Parameters:
                        states_list (List): List of the states from sims
                        model (string): specified learned model to use
                Returns:
                        energy_list (List): List of the energy for each flight
    ''' 
    configs = evaluation.get_pre_def_configs()
    model, inputs = load_model(configs[model], model)

    first = inputs['lookback'] - 1

    all_data = {i: sim_to_network_transform(states) for i, states in enumerate(states_list)}
   
    data, data_min, data_max, _, _, _ = process_data(all_data, eval_mode=True, **inputs)


    data_tensors = create_tensors(data, range(len(states_list)), **inputs)

    popt = liu_model.optimum_values()

    energy_list = []

    for i, states in enumerate(states_list):
        # liu_pow_list = liu_model.power(states[['vertspd', 'airspeed','aoa','payload', 'density']].T, popt[0], popt[1], popt[2], popt[3], popt[4])
        # liu_set_energy = simps(liu_pow_list, x=states['time'], even="avg")/1000 # Kilojoules
        # print(liu_set_energy)

        model.reset_states()
        y_pred = model.predict(data_tensors[i]).reshape(-1)
        y_pred = y_pred*(data_max[-1] - data_min[-1]) + data_min[-1]

        energy_list.append(calc_energy(y_pred, states['time'].values[first:]))
        
    return energy_list


def MC_func(arg_in):
    i, waypoints, wind_map_ang, wind_map_mag, airspeed,wind_inlet_ang_dist, wind_inlet_mag_dist = arg_in
    np.random.seed()
    testsim = QuadSim(init_pos=waypoints[:,0], wind_map_ang=wind_map_ang, wind_map_mag=wind_map_mag, command_airspeed=True, ghost_delta=1.0)
    testsim.init_path_from_waypoints(waypoints, airspeed=airspeed) #Modify set of waypoints here

    inlet_ang = np.random.normal(wind_inlet_ang_dist[0], wind_inlet_ang_dist[1])
    inlet_mag = np.random.normal(wind_inlet_mag_dist[0], wind_inlet_mag_dist[1])
    testsim.update_inlet(inlet_ang, inlet_mag, 'rbf')

    states, count, path_count, success = testsim.propagate()
    
    states = sim_to_network_transform(states)
    
    return (states, success)

def mc_flights(n_times=50, threads=3):
    wind_map_ang = np.load('wind_data/dataset_hawkins.npy')
    wind_map_mag = np.load('wind_data/dataset_hawkins_mag.npy')
    wind_inlet_ang_dist = (-2.53455, 28.4662) # mean and std
    wind_inlet_mag_dist = (3.0, 1.5) # mean and std


    # waypoints given as x, y, z, time
    waypoints = np.array([[-259.9, -260., -260., -200.,  -65.,  -64.9],
                          [  125.,  125.,   50.,  -35.,  100.,   100.],
                          [    0.,   20.,   20.,   20.,   20.,     0.],
                          [    0.,    3.,    3.,    3.,    3.,     5.]])


    airspeed = 10

    state_list = []
    start_time = time.time()
    pool = Pool(processes=threads)
    count = 0
    fail_count = 0 # in situations where the wind is too strong for vehicle to reach target

    for i in pool.imap_unordered(MC_func, [(i,waypoints, wind_map_ang, wind_map_mag, airspeed,wind_inlet_ang_dist, wind_inlet_mag_dist) for i in range(n_times)]):
        state_list.append(i[0])
        count += 1
        if i[1] == False:
            fail_count+=1
        if count%10 == 0:
            print("Sim Count: {}".format(count))


    end_time = time.time()
    print("Fail count: {}".format(fail_count))
    print("Time to complete MC: {}".format(end_time - start_time))
    return state_list

def plot_hist(energy_list):
    num_bins = 20
    n, bins, patches = plt.hist(np.array(energy_list), num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel('Kilojoules')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.title('Energy consumption')
    plt.show()

# if the file is executed as the main program
if __name__ == '__main__':
    # Example of running one flight
    states = one_flight()
    states = [states[0]]
    energy_list = energy_predictions(states)

    # Example of MC runs
    state_list = mc_flights(n_times=50, threads=3)
    energy_list = energy_predictions(state_list)
    plot_hist(energy_list)

    # Calculating risk from MC runs
    risk = Risk.Risk(np.array(energy_list), limit=99., a = 64 ,b = 92.34 ,B = 369.36)
    risk_array = risk.risk
    print("CVaR is {}".format(risk.cvar()))