import numpy as np
from windfield import WindField
from windfield import DrydenSim
import pylab as pl
import pandas as pd
from IPython import display
import time


class QuadSim():
    """
    A class to simulate a quadrotor flying a set of waypoints including the    |
    effects of wind and drag.

    ...

    Attributes
    ----------
    path : np.ndarray
        Array of shape 3 x n, where n is the number of points to follow. Each 
        point is separated in relation to the desired speed of the UAV and the 
        trajectory resolution (traj_res). The rows are the x, y, z position of 
        the point.
    init_pos : np.ndarray
        The initial x, y, z position of the UAV. By default it is at the origin
        and 20 meters in the air. 
    windField : WindField
        WindField object that describes the wind flow field for a given wind map
        and inlet angle and magnitude.
    dryden : DrydenSim
        DyrdenSim object that models Dryden turbulence 
    ground_speed : float
        Value for groundspeed in m/s
    ghost_delta : float
        Value for the look-ahead time for the trajectory following
    traj_res : float
        The time in seconds between each point in the path.
    states_hist: np.ndarray
        Array of specified states of interest to be the output of the propogate
        function.
    sim_dt : float
        The time delta between loops of the sim. Should match traj_res.
    mass : float
        Value for the mass of the UAV.
    max_tilt_angle : float
        Value for the maximum tilt angle of the UAV (degrees).

    Methods
    -------
    dynamics_model(last_pred_state, u):
        Returns the next predicted state as a QuadState object
    sim_controller(self, state, des_state):
        Returns commanded roll, pitch, and thrust
    propagate(display_fig=False):
        Runs the simulation with the set path and returns the states history, 
        loop iteration count, number of path points, and bool for successfully
        reaching the goal location.
    plot_path():
        Plots the path attribute
    init_path_from_waypoints(waypoints, groundspeed):
        Sets the path and groundspeed attributes using the input waypoints 
        and groundspeed.
    update_inlet(inlet_ang, inlet_mag, method='rbf'):
        Calls the update_inlet method of the WindField object
    append_states_hist(time, state):
        Appends specified states to states_hist attribute
    """
    def __init__(self,path=None, init_pos=None, wind_map_ang=None, 
                 wind_map_mag=None, inlet_ang=None, inlet_mag=None,
                 ground_speed=10, ghost_delta=2.0, traj_res=0.1,
                 sim_dt=0.1, mass=3.71, max_tilt_angle=35, 
                 air_speed=10, command_airspeed = False):
        """
        Constructs all the necessary attributes for the risk object.

        Parameters
        ----------
            path : np.ndarray
                Array of shape 3 x n, where n is the number of points to 
                follow. Each point is separated in relation to the desired 
                speed of the UAV and the trajectory resolution (traj_res). The
                rows are the x, y, z position of the point.
            init_pos : np.ndarray
                The initial x, y, z position of the UAV. By default it is at 
                the origin and 20 meters in the air. 
            wind_map_ang : np.ndarray
                n x 4 x m matrix of wind angles, where n is the number of x,y 
                points, m is the number of inlet angles, and the columns of the 
                second dimension are the inlet angles, x position, y position,
                and the angle at the point.
            wind_map_mag : np.ndarray
                n x 4 x m matrix of wind magnitudes, where n is the number of 
                x,y points, m is the number of inlet angles, and the columns of 
                the second dimension are the inlet angles, x position, y 
                position, and the magnitude at the point.
            inlet_ang : float
                inlet angle (degrees)
            inlet_mag : float
                current inlet magnitude (m/s)
            ground_speed : float
                Value for desired groundspeed in m/s
            ghost_delta : float
                Value for the look-ahead time for the trajectory following
            traj_res : float
                The time in seconds between each point in the path.
            sim_dt : float
                The time delta between loops of the sim. Should match traj_res.
            mass : float
                Value for the mass of the UAV.
            max_tilt_angle : float
                Value for the maximum tilt angle of the UAV (degrees).
        """
        self.ground_speed = ground_speed
        self.air_speed = air_speed
        self.command_airspeed = command_airspeed
        self.traj_res = traj_res
        self.sim_dt = sim_dt
        self.mass = mass
        self.max_tilt_angle = max_tilt_angle
        self.path = path
        self.payload = 0
        self.air_density = 1.1718938453052181
        if init_pos is None:
            init_x = 0
            init_y = 0
            init_z = 20
        else:
            init_x = init_pos[0]
            init_y = init_pos[1]
            init_z = init_pos[2]
        self.init_state = QuadState(pos=np.array([init_x,init_y,init_z]))
        self.states_hist = []
        self.append_states_hist(0, self.init_state)
        self.ghost_delta = ghost_delta #secs
        self.actual_path = [self.init_state]
        self.xy_path = [(init_x,init_y)]
        dryden_upsample = 10
        if command_airspeed:
            # print("commanding airspeed")
            self.dryden = DrydenSim(self.sim_dt/dryden_upsample, self.air_speed)
        else:
            self.dryden = DrydenSim(self.sim_dt/dryden_upsample, self.ground_speed)

        if inlet_ang is not None and inlet_mag is not None:
            self.windField = WindField(wind_map_ang, wind_map_mag, 
                                       inlet_ang, inlet_mag)
        else:
            self.windField = WindField(wind_map_ang, wind_map_mag)
    
    def dynamics_model(self, last_pred_state, u, C_d=.65, dist_std=.1, A=.25, 
                       rho=1.1718938453052181):
        '''
        Returns the next predicted state of the quadrotor (acceleration-based).

                Parameters:
                        last_pred_state (np.ndarray): percentage of confidence
                        u (np.ndarray): array of commanded roll, pitch, and 
                            thrust.
                        C_d (float): coefficient of drag .65
                        dist_std (float): standard deviation of the disturbance 
                            in acceleration.
                        A (float): surface area
                        rho (float): air density

                Returns:
                        next_pred_state (QuadState): next predicted state
        '''
        
        [roll_d, pitch_d, thrust_d] = u 
        [phi, theta, psi] = last_pred_state.rpy
        tau_roll = .2 # .2
        tau_pitch = .2
        K_roll = 1.5 # 1.5
        K_pitch = 1.5 # 1.5
        g = -9.81

        vphi = (1.0/tau_roll) * (K_roll*(roll_d - phi))
        vtheta = (1.0/tau_pitch) * (K_pitch*(pitch_d - theta))
        vpsi = 0 # assumes constant yaw
        v_rpy = np.array([vphi, vtheta, vpsi])

        quad_rpy = last_pred_state.rpy + self.sim_dt * v_rpy
        [phi, theta, psi] = quad_rpy
        ax_nodist = (np.cos(psi)*np.sin(theta)*np.cos(phi) + 
                     np.sin(psi)*np.sin(phi)) * thrust_d * 1/self.mass
        ay_nodist = (np.sin(psi)*np.sin(theta)*np.cos(phi) - 
                     np.cos(psi)*np.sin(phi)) * thrust_d * 1/self.mass
        az_nodist = (np.cos(theta)*np.cos(phi)) * thrust_d * 1/self.mass + g
        quad_acc_nodist = np.array([ax_nodist, ay_nodist, az_nodist])
        
        # Raleigh drag equation
        Fd = -(.5 * C_d * rho * A * np.power(last_pred_state.airVel,2) * 
               np.sign(last_pred_state.airVel)) 
        wind_acc = Fd/self.mass 
        quad_acc_dist = quad_acc_nodist + wind_acc

        # Random disturbance from sensor noise, model mismatch, etc.
        quad_acc = quad_acc_dist + np.random.normal(loc = 0, scale = dist_std, 
                                                    size=3)# point-mass model
        quad_vel = last_pred_state.vel + (last_pred_state.acc * self.sim_dt) 
        quad_pos = last_pred_state.pos + (last_pred_state.vel * self.sim_dt)
        
        # Query wind speed
        ang, mag = self.windField.wind_at_point(quad_pos[0],quad_pos[1]) # wind is in NWU convention. ***Check***
        ang = np.deg2rad(ang)
        for i in range(9): # updating dryden model at higher rate than sim
            self.dryden.update() 
        gust = self.dryden.update()
        mag_gust = np.hypot(gust[0], gust[1])
        mag_ang = np.arctan2(gust[1],gust[0])
        
        
        #wind triangle
        quad_airVel = quad_vel - np.array(([mag*np.cos(ang), 
                                            mag*np.sin(ang), 
                                            0])) + np.array(([
                                            mag_gust*np.cos(mag_ang + psi), 
                                            mag_gust*np.sin(mag_ang + psi), 
                                            gust[2]]))

        next_pred_state = QuadState(quad_pos, quad_vel, quad_acc, quad_rpy, 
                                    quad_airVel, thrust_d)
        
        return next_pred_state
            
    def sim_controller(self, state, des_state):
        '''
        Returns the array of commanded roll, pitch, and thrust.

                Parameters:
                        state (QuadState): current state
                        des_state (QuadState): desired state

                Returns:
                        pred_u (np.ndarray): array of commanded roll, 
                            pitch, and thrust.
        '''
        # Output: Predicted Control Output
        gain_P = 1
        gain_D = 3 #3
        error_vel = gain_P * (des_state.pos - state.pos)
        if self.command_airspeed:
            error_vel_mag = np.linalg.norm(error_vel[0:2])
            error_vel_ratio = np.clip(error_vel_mag,-self.air_speed,
                                    self.air_speed)/error_vel_mag
            error_vel[0:2] *= error_vel_ratio
            # cascaded controller, no feedforward
            error_acc = gain_D * (error_vel - state.airVel) 
            error_acc[2] += 9.81
        else:
            error_vel_mag = np.linalg.norm(error_vel[0:2])
            error_vel_ratio = np.clip(error_vel_mag,-self.ground_speed,
                                    self.ground_speed)/error_vel_mag
            error_vel[0:2] *= error_vel_ratio
            # cascaded controller, no feedforward
            error_acc = gain_D * (error_vel - state.vel) 
            error_acc[2] += 9.81
        
        # Dynamic Inversion
        acc_norm = np.linalg.norm(error_acc)
        
        roll_d = -np.sin(error_acc[1]/acc_norm)
        pitch_d = np.sin(error_acc[0]/(acc_norm * np.cos(roll_d)))
        thrust_d = error_acc[2]*self.mass # thrust_boost_factor
        
        # rough tilt limit
        tilt_angle = np.rad2deg(np.arccos(np.cos(roll_d)*np.cos(pitch_d)))
        if tilt_angle > self.max_tilt_angle:
            tilt_ratio = self.max_tilt_angle/tilt_angle
            roll_d *= tilt_ratio
            pitch_d *= tilt_ratio


        pred_u = np.array([roll_d, pitch_d, thrust_d])
        return pred_u     
    
    def propagate(self, display_fig=False):
        '''
        Runs the simulation with the set path and returns the states history, 
        loop iteration count, number of path points, and bool for successfully
        reaching the goal location.

                Parameters:
                        display_fig (bool): bool to display the figure or not

                Returns:
                        self.states_hist (np.array): 
                        count (int): number of loops for the sim to run
                        self.path.shape[1] (int): number of path points in the sim
                        success (bool): if the quadrotor reached the last path point
        '''
        if display_fig:
            fig = pl.figure(1,figsize=(10,10))
            ax = fig.add_subplot(111)
            ax.plot(self.path[0,:],self.path[1,:],color='k')
            ax.axis('equal')
        
        success = True
        curr_state = self.init_state
        curr_time = 0
        i = int(self.ghost_delta/self.traj_res)
        count = 0
        # print("initial path length {}".format(self.path.shape[1]))
        while i < self.path.shape[1]:
            self.xy_path.append((curr_state.pos[0],curr_state.pos[1]))
            X_d = self.path[:,i]
            desired_state = QuadState(pos=X_d)
            pred_u = self.sim_controller(curr_state,desired_state)
            next_state = self.dynamics_model(curr_state,pred_u)
            
            if display_fig:
                plot_path = np.array(self.xy_path)
                
            self.actual_path.append(curr_state) 
            curr_state = next_state 
            curr_time += self.sim_dt
            self.append_states_hist(curr_time, next_state)
            dist = np.linalg.norm(self.path[0:2,i]-curr_state.pos[0:2])
            count += 1
            
            # make sure desired state doesn't get too far away from current state
            if self.command_airspeed:
                if dist < (self.air_speed * self.ghost_delta): 
                    i += 1
            else:  
                if dist < (self.ground_speed * self.ghost_delta): 
                    i += 1

            # break if sim is running to long and hasn't reached the goal state
            if count > 15*self.path.shape[1]:
                success = False
                break
                
        if display_fig:
            ax.plot(plot_path[:,0],plot_path[:,1],color='r')   
            pl.show()
        
        return_states = pd.DataFrame(data=np.array(self.states_hist), columns=['time', 'airspeed', 'airspeed_angle', 'east_position', 'north_position', 'up_position', 'east_velocity', 'north_velocity', 'vertspd', 'aoa', 'heading', 'payload', 'density'])
        return return_states, count, self.path.shape[1], success
          
        
    def plot_path(self):
        '''
        Plots the path attribute
        '''
        plt.figure(1)
        plt.plot(self.path[0,:],self.path[1,:])
        plt.show()
        
    def init_path_from_waypoints(self, waypoints, groundspeed=None, airspeed=None):
        '''
        Sets the path and groundspeed attributes using the input waypoints 
        and groundspeed.

                Parameters:
                        waypoints (np.array): 4 x n matrix of waypoints. 
                            Row 1 is x, row 2 is y, and row 3 is z. Row 4
                            is the desired time at the waypoint. 
                        groundspeed (float): desired groundspeed of UAV (m/s)
        '''
        # waypoints are x, y, z, time
        if groundspeed != None:
            self.ground_speed = groundspeed
            self.dryden = DrydenSim(self.sim_dt/10, self.ground_speed)
        elif airspeed != None:
            # print("Flying airspeed")
            self.air_speed = airspeed
            self.dryden = DrydenSim(self.sim_dt/10, self.air_speed)
        else:
            print("Need to input speed")
            return
        path = np.empty((3,0))
        # Add descent and accent rate here in splitting paths
        for i in range(0,np.shape(waypoints)[1]-1):
            waypoint1 = waypoints[0:3,i]
            waypoint2 = waypoints[0:3,i+1]
            way_time = waypoints[3,i+1]

            distance = np.linalg.norm(waypoint2-waypoint1)
            if groundspeed != None:
                num_points = int(distance/groundspeed/self.traj_res)
            elif airspeed != None:
                num_points = int(distance/airspeed/self.traj_res)
            else:
                print("Need to input speed")
                return
            path = np.hstack((path,np.vstack((np.linspace(waypoint1[0],
                                                          waypoint2[0],
                                                          num_points),
                                              np.linspace(waypoint1[1],
                                                          waypoint2[1],
                                                          num_points),
                                              np.linspace(waypoint1[2],
                                                          waypoint2[2],
                                                          num_points)))))
            if way_time > 0:
                num_points = int(way_time/self.traj_res)
                path = np.hstack((path,
                                  np.vstack((np.ones(num_points)*waypoint2[0],
                                             np.ones(num_points)*waypoint2[1],
                                             np.ones(num_points)*waypoint2[2]))))
                
        self.path = path
        
    def update_inlet(self, inlet_ang, inlet_mag, method='rbf'):
        '''
        Calls the update_inlet method of the WindField object

                Parameters:
                        inlet_ang (float): inlet angle (degrees)
                        inlet_mag (float): inlet magnitude (m/s)
                        method (string): method of interpolation. Options
                            are rbf, linear, or nearest
        '''
        self.windField.update_inlet(inlet_ang, inlet_mag, method)
        
    def append_states_hist(self, time, state):
        '''
        Appends specified states to states_hist attribute

                Parameters:
                        time (float): current time
                        state (QuadState): current state
        '''
        # time, airspeed, airspeed angle, position, velocity (ENU), tilt angle, flight_angle
        NWU = False
        # print(state.airVel)
        airspeed_ang_NWU = np.rad2deg(np.arctan2(state.airVel[1], state.airVel[0]))
        airspeed_mag = np.hypot(state.airVel[1], state.airVel[0])
        if NWU:
            self.states_hist.append((time, 
                                     airspeed_mag, 
                                     airspeed_ang_NWU, 
                                     state.pos[0],
                                     state.pos[1],
                                     state.pos[2],
                                     state.vel[0], 
                                     state.vel[1],
                                     state.vel[2]))
        else: # ENU and Wind clockwise from north
            # airspeed_ang_NWU = (airspeed_ang_NWU-180)%360
            # 'time', 'airspeed', 'airspeed_angle', 'east_position', 'north_position', 'up_position', 'east_velocity', 'north_velocity', 'vertspd', 'aoa', 'heading'
            self.states_hist.append((time, 
                                     airspeed_mag, 
                                     (airspeed_ang_NWU+90)%360, 
                                     -state.pos[1],
                                     state.pos[0],
                                     state.pos[2],
                                     -state.vel[1], 
                                     state.vel[0],
                                     state.vel[2], 
                                     np.arccos(np.cos(state.rpy[0])*np.cos(state.rpy[1])),#aoa radians
                                     np.rad2deg(np.arctan2(state.vel[0], -state.vel[1])), #heading degrees
                                     self.payload,
                                     self.air_density))
            

        
        
    

class QuadState():
    """
    A class for representing the state of a quadrotor

    ...

    Attributes
    ----------
    pos : np.ndarray
        array of the x, y, z position of the quadrotor
    vel : np.ndarray
        array of the x, y, z ground velocity of the quadrotor
    acc : np.ndarray
        array of the x, y, z accelaration of the quadrotor
    rpy : np.ndarray
        array of the roll, pitch, and yaw of the quadrotor
    airVel : np.ndarray
        array of the x, y, z air velocity of the quadrotor
    thrust : float
        the current thrust of the quadrotor
    """
    def __init__(self, pos=np.zeros((3,)), vel=np.zeros((3,)),
                 acc=np.zeros((3,)), rpy=np.zeros((3,)),
                 airVel=np.zeros((3,)), thrust=0):
        """
        Constructs all the necessary attributes for the QuadState object.

        Parameters
        ----------
            pos : np.ndarray
                array of the x, y, z position of the quadrotor
            vel : np.ndarray
                array of the x, y, z ground velocity of the quadrotor
            acc : np.ndarray
                array of the x, y, z accelaration of the quadrotor
            rpy : np.ndarray
                array of the roll, pitch, and yaw of the quadrotor
            airVel : np.ndarray
                array of the x, y, z air velocity of the quadrotor
            thrust : float
                the current thrust of the quadrotor
        """
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.rpy = rpy
        self.airVel = airVel
        self.thrust = thrust
