import numpy as np
import pandas as pd
"""Module to calculate the power consumption and estimate the parameters
Note: The arguments in all calculation functions can be of type float or NumPy Array(float) to accomodate different steps in calculation

Bastian Wagner
Carnegie Mellon University - July 2019

Revised:
Arnav Choudhry
Carnegie Mellon University - May 2021
"""

####################################
# Constants for the Calculation
####################################
M = 3.71  # Empty Mass of the drone (kg)
R = 0.175  # Length of a blade (m)
C = 0.03  # Blade Chord width (m)
G = 9.81  # Gravitational constant (m/s²)
N = 4  # Number of props


def thrust(payload, c4, c5, c6, v_air, alpha):
    """Calculate the thrust based on the payload
    T=(m+payload)*g
    Arguments:
        payload {float} -- The payload (g)

    Returns:
        float  -- The thrust (kg*m/s²)
    """
    mg = (M + payload) * G
    first = np.power(mg - c5 * np.power(v_air * np.cos(alpha), 2), 2)
    second = np.power(c4 * np.power(v_air, 2), 2)
    return np.sqrt(first + second)


def induced(k1, k2, v_vert, thrust):
    """Calculate the induced power

    Arguments:
        v_i {float} -- The induced velocity (m/s)
        v_air {float} -- The airspeed (m/s)
        alpha {float} -- The AOA (rad)
        thrust {float} -- The thrust (kg*m/s²)

    Returns:
        float -- The induced power (W)
    """

    return k1 * thrust * ((v_vert / 2) + np.sqrt((v_vert / 2)**2 +
                                                 (thrust / (np.power(k2, 2)))))


def angular(thrust, k):
    """Calculate the angular speed

    Arguments:
        thrust {float} -- The Thrust (kg*m/s²)
        k {float} -- The scaling factor k

    Returns:
        float -- The angular speed
    """

    return np.sqrt(thrust / k)


def profile(v_air, thrust, alpha, c2, c3):
    """Calculate the profile power

    Arguments:
        v_air {float} -- Airspeed (m/s)
        thrust {float} -- Thrust (kg*m/s²)
        alpha {float} -- AOA (rad)
        rho {floar} -- Air Density (kg/m³)
        k {float} -- Scaling factor k
        c_d {float} -- Drag coefficient of the blades

    Returns:
        float -- The profile power (W)
    """

    return c2 * np.power(thrust, 1.5) + c3 * np.power(
        (v_air * np.cos(alpha)), 2) * np.sqrt(thrust)


def parasitic(v_air, c4):
    """Calculate the parasitic power

    Arguments:
        v_air {float} -- Airspeed (m/s)
        rho {floar} -- Air Density (kg/m³)
        c_d {float} -- Drag coefficient of the drone
        A {float} -- Facing area of the drone (m²)

    Returns:
        float -- The parasitic power (W)
    """

    return c4 * np.power(v_air, 3)


def solve_quart(v_air, thrust, rho, A):
    """Solve the formula for v_i in regard to v_i

    Arguments:
        v_air {float} -- The airspeed (m/s)
        thrust {float} -- The thrust (kg*m/s²)
        rho {floar} -- Air Density (kg/m³)
        A {float} -- The facing area (m²)

    Returns:
        float -- The induced speed (m/s)
    """

    # Calculate the four solutions for v_i
    # Only solution 2 is needed because the others are either negative or unreal
    one = -np.sqrt(
        np.sqrt((16 * thrust**2) /
                (A**2 * rho**2) + 16 * v_air**4) / 8 - v_air**2 / 2)
    two = np.sqrt(
        np.sqrt((16 * thrust**2) / (A**2 * rho**2) + 16 * v_air**4) / 8 -
        v_air**2 / 2)
    three = -np.sqrt(-np.sqrt((A**2 * rho**2 * v_air**4 + thrust**2) /
                              (A**2 * rho**2)) - v_air**2) / np.sqrt(2)
    four = np.sqrt(-np.sqrt((A**2 * rho**2 * v_air**4 + thrust**2) /
                            (A**2 * rho**2)) - v_air**2) / np.sqrt(2)
    return two


def power(data, k1, k2, c2, c4, c5):
    """Calculate the combined power consumption

    Arguments:
        data  -- Either an array in scipy x-data format or a tuple of x-data
        k {float} -- The scaling factor
        c_d {float} -- Drag coefficient of the blade
        c_d2 {float} -- Drag coefficient of the dron
        A {float} -- Facing area of the drone (m²)

    Returns:
        float -- The combined power consumption
    """

    c1 = k1 / k2
    c3 = 0
    c6 = 0

    # Split the data tuple
    vertspd = data.loc["vertspd", :].to_numpy()
    airspeed = data.loc["airspeed", :].to_numpy()
    aoa = data.loc["aoa", :].to_numpy()
    payload = data.loc["payload", :].to_numpy()
    density = data.loc["density", :].to_numpy()

    results = []
    # Calculate the induced power based on the current flight status
    # Currently only forward flight data is being read from the flights
    # for v_vert, v_air, alpha, pld, rho in zip(vertspd, airspeed, aoa, payload, density):
    # Calculate the single powers and needed values
    t = thrust(payload, c4, c5, c6, airspeed, aoa)
    # When the airspeed is over 1 m/s, the drone is in forward flight
    # if np.round(np.abs(v_air), decimals=2) > 1:
    # v_i = solve_quart(airspeed, t, density, A)
    p_i = induced(k1, k2, vertspd, t)
    # # Else, when the vertical speed is between -0.5 and 0.5 m/s, the drone is in hover
    # elif np.round(v_vert, decimals=2) < 0.5 and np.round(v_vert, decimals=2) > -0.5:
    #     v_i = np.sqrt(t/(2*rho*A))
    #     p_i = t*v_i
    # # # Else, when the vertical speed is greater than 0.5 m/s, the drone is in climb
    # elif np.round(v_vert, decimals=2) >= 0.5:
    #     v_i = (-v_vert/2)+np.sqrt(np.power(v_vert/2,2)+t/(2*rho*A))
    #     p_i = t*(v_i+v_vert)
    # # Else, when the vetical speed is less than 0.5 m/s, the drone is in descend
    # elif np.round(v_vert, decimals=2) <= -0.5:
    #     v_i = (-v_vert/2)+np.sqrt(np.power(v_vert/2,2)-t/(2*rho*A))
    #     p_i = t*(v_i+v_vert)
    p_p = profile(airspeed, t, aoa, c2, c3)
    p_pa = parasitic(airspeed, c4)

    # Combine all powers
    result = p_i + p_p + p_pa
    return result


def fwd_power(data, c4, c5):
    """Calculate the combined power consumption

    Arguments:
        data  -- Either an array in scipy x-data format or a tuple of x-data
        k {float} -- The scaling factor
        c_d {float} -- Drag coefficient of the blade
        c_d2 {float} -- Drag coefficient of the dron
        A {float} -- Facing area of the drone (m²)

    Returns:
        float -- The combined power consumption
    """
    k1 = 0.9999
    k2 = 100
    c2 = 2.06187671

    c1 = k1 / k2
    c3 = 0
    c6 = 0

    # Split the data tuple
    vertspd = data.loc["vertspd", :].to_numpy()
    airspeed = data.loc["airspeed", :].to_numpy()
    aoa = data.loc["aoa", :].to_numpy()
    payload = data.loc["payload", :].to_numpy()
    density = data.loc["density", :].to_numpy()

    results = []
    # Calculate the induced power based on the current flight status
    # Currently only forward flight data is being read from the flights
    # for v_vert, v_air, alpha, pld, rho in zip(vertspd, airspeed, aoa, payload, density):
    # Calculate the single powers and needed values
    t = thrust(payload, c4, c5, c6, airspeed, aoa)
    # When the airspeed is over 1 m/s, the drone is in forward flight
    # if np.round(np.abs(v_air), decimals=2) > 1:
    # v_i = solve_quart(airspeed, t, density, A)
    p_i = induced(k1, k2, 0, t)
    # # Else, when the vertical speed is between -0.5 and 0.5 m/s, the drone is in hover
    # elif np.round(v_vert, decimals=2) < 0.5 and np.round(v_vert, decimals=2) > -0.5:
    #     v_i = np.sqrt(t/(2*rho*A))
    #     p_i = t*v_i
    # # # Else, when the vertical speed is greater than 0.5 m/s, the drone is in climb
    # elif np.round(v_vert, decimals=2) >= 0.5:
    #     v_i = (-v_vert/2)+np.sqrt(np.power(v_vert/2,2)+t/(2*rho*A))
    #     p_i = t*(v_i+v_vert)
    # # Else, when the vetical speed is less than 0.5 m/s, the drone is in descend
    # elif np.round(v_vert, decimals=2) <= -0.5:
    #     v_i = (-v_vert/2)+np.sqrt(np.power(v_vert/2,2)-t/(2*rho*A))
    #     p_i = t*(v_i+v_vert)
    p_p = profile(airspeed, t, aoa, c2, c3)
    p_pa = parasitic(airspeed, c4)

    # Combine all powers
    result = p_i + p_p + p_pa
    return result


def ascend_power(data, k1, k2, c2):
    """Calculate the combined power consumption

    Arguments:
        data  -- Either an array in scipy x-data format or a tuple of x-data
        k {float} -- The scaling factor
        c_d {float} -- Drag coefficient of the blade
        c_d2 {float} -- Drag coefficient of the dron
        A {float} -- Facing area of the drone (m²)

    Returns:
        float -- The combined power consumption
    """

    c1 = k1 / k2
    c3 = 0
    c6 = 0
    # Split the data tuple
    vertspd = data.loc["vertspd", :].to_numpy()
    airspeed = data.loc["airspeed", :].to_numpy()
    aoa = data.loc["aoa", :].to_numpy()
    payload = data.loc["payload", :].to_numpy()
    density = data.loc["density", :].to_numpy()

    results = []
    # Calculate the induced power based on the current flight status
    # Currently only forward flight data is being read from the flights
    # for v_vert, v_air, alpha, pld, rho in zip(vertspd, airspeed, aoa, payload, density):
    # Calculate the single powers and needed values
    t = (M + payload) * G

    # When the airspeed is over 1 m/s, the drone is in forward flight
    # if np.round(np.abs(v_air), decimals=2) > 1:
    # v_i = solve_quart(airspeed, t, density, A)
    p_i = induced(k1, k2, vertspd, t)
    # # Else, when the vertical speed is between -0.5 and 0.5 m/s, the drone is in hover
    # elif np.round(v_vert, decimals=2) < 0.5 and np.round(v_vert, decimals=2) > -0.5:
    #     v_i = np.sqrt(t/(2*rho*A))
    #     p_i = t*v_i
    # # # Else, when the vertical speed is greater than 0.5 m/s, the drone is in climb
    # elif np.round(v_vert, decimals=2) >= 0.5:
    #     v_i = (-v_vert/2)+np.sqrt(np.power(v_vert/2,2)+t/(2*rho*A))
    #     p_i = t*(v_i+v_vert)
    # # Else, when the vetical speed is less than 0.5 m/s, the drone is in descend
    # elif np.round(v_vert, decimals=2) <= -0.5:
    #     v_i = (-v_vert/2)+np.sqrt(np.power(v_vert/2,2)-t/(2*rho*A))
    #     p_i = t*(v_i+v_vert)
    p_p = profile(0, t, 0, c2, 0)

    # Combine all powers
    result = p_i + p_p
    return result

def optimum_values():
    
    # default values for liu model
    popt = [0.99545857,100.,2.06852797,0.02745631,0.01473469]
    return popt

