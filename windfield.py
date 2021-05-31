import numpy as np
from scipy import interpolate

class WindField:
    """
    A class to find wind at a point given a wind field model and inlet 
    conditions.

    ...

    Attributes
    ----------
    wind_map_ang: n x 4 x m matrix of wind angles, where n is the number of x,y 
                points, m is the number of inlet angles, and the columns of the 
                second dimension are the inlet angles, x position, y position,
                and the angle at the point.
    wind_map_mag: n x 4 x m matrix of wind magnitudes, where n is the number of 
                x,y points, m is the number of inlet angles, and the columns of 
                the second dimension are the inlet angles, x position, y 
                position, and the magnitude at the point.
    wind_map_degrees: m dimmensional array of the inlet angles for the wind map, where m is the number of different
        inlet angles.
    wind_map_probes: n x 2 matrix of the x,y position of the points for measurements. x = North, and y = West.
    current_mag_map: n dimensional array of the wind magnitudes at each x,y point for the given inlet conditions
    current_ang_map: n dimensional array of the wind angles at each x,y point for the given inlet conditions
    wind_inlet_ang: current inlet angle (degrees)
    wind_inlet_mag: current inlet magnitude (m/s)

    Methods
    -------
    update_inlet(wind_inlet):
        Updates the inlet conditions and the current magnitude and angle map for the inlet conditions
    wind_at_point(x,y):
        returns the wind angle and magnitude at the x,y positions
    """

    def __init__(self, wind_map_ang, wind_map_mag, wind_inlet_ang=None, wind_inlet_mag=None, method='rbf'):
        """
        Constructs all the necessary attributes for the wind field object.

        Parameters
        ----------
            wind_map_ang: n x 4 x m matrix of wind angles, where n is the number of x,y points, m is the number of
                inlet angles, and the columns of the second dimension are the inlet angles, x position, y position,
                and the angle at the point.
            wind_map_mag: n x 4 x m matrix of wind magnitudes, where n is the number of x,y points, m is the number of
                inlet angles, and the columns of the second dimension are the inlet angles, x position, y position,
                and the magnitude at the point.
            wind_inlet_ang: current inlet angle (degrees)
            wind_inlet_mag: current inlet magnitude (m/s)
        """
        self.wind_map_ang = wind_map_ang[:, 3, :]
        self.wind_map_mag = wind_map_mag[:, 3, :]
        self.wind_map_degrees = wind_map_mag[0, 0, :]
        self.wind_map_probes = wind_map_mag[:, 1:3, 0]
        self.current_mag_map = None
        self.current_ang_map = None 
        self.interp_ang_real = None
        self.interp_ang_imag = None
        self.interp_mag = None
        self.wind_inlet_ang = wind_inlet_ang
        self.wind_inlet_mag = wind_inlet_mag
        self.method = method
        self.update_inlet(wind_inlet_ang, wind_inlet_mag, method)

    def update_inlet(self, inlet_ang, inlet_mag, method='rbf'):
        '''
        Updates the inlet conditions and the current magnitude and angle map for the inlet conditions

        For large wind fields with many data points, nearest is the recommended method for the sake 
        of computational efficiency. For smaller wind fields, the default rbf or linear works well.

                Parameters:
                        wind_inlet: tuple of inlet angle and magnitude
        '''
        self.wind_inlet_ang = inlet_ang
        self.wind_inlet_mag = inlet_mag
        self.method = method
        CFD_inlet_mag = 10
        if inlet_ang:
            inlet_ang = inlet_ang % 360
            self.wind_inlet_ang = inlet_ang
            below_index_list = np.argwhere(self.wind_map_degrees <= inlet_ang)
            below_index = below_index_list[np.size(below_index_list) - 1]
            if below_index == np.size(self.wind_map_degrees) - 1:
                above_index = [0]
            else:
                above_index = below_index + 1

            # Find angles at probes from linear interpolation of data
            below_ang_map = self.wind_map_ang[:, below_index]
            above_ang_map = self.wind_map_ang[:, above_index]
            diff_ang_map = above_ang_map - below_ang_map
            above_wrap = np.argwhere(diff_ang_map > 180)
            below_wrap = np.argwhere(diff_ang_map < -180)
            diff_ang_map[above_wrap] -= 360
            diff_ang_map[below_wrap] += 360
            if above_index[0]:
                fractional_diff = (inlet_ang - self.wind_map_degrees[below_index]) \
                                  / (self.wind_map_degrees[above_index] - self.wind_map_degrees[below_index])
            else:
                fractional_diff = (inlet_ang - self.wind_map_degrees[below_index]) \
                                  / (360 - self.wind_map_degrees[below_index])      
            self.current_ang_map = (below_ang_map + fractional_diff * diff_ang_map) % 360

            complex_ang_map = np.exp(1j * np.deg2rad(self.current_ang_map))

            # Find mag at probes from linear interpolation of data
            below_mag_map = self.wind_map_mag[:, below_index]
            above_mag_map = self.wind_map_mag[:, above_index]
            diff_mag_map = above_mag_map - below_mag_map
            self.current_mag_map = (below_mag_map + fractional_diff * diff_mag_map) * inlet_mag / CFD_inlet_mag


            # Set interpolator
            if method == 'nearest':
                self.interp_ang_real = interpolate.NearestNDInterpolator(self.wind_map_probes[:,0:2] ,np.real(complex_ang_map))
                self.interp_ang_imag = interpolate.NearestNDInterpolator(self.wind_map_probes[:,0:2] ,np.imag(complex_ang_map))
                self.interp_mag = interpolate.NearestNDInterpolator(self.wind_map_probes[:,0:2],self.current_mag_map)
            elif method == 'linear':
                self.interp_ang_real = interpolate.LinearNDInterpolator(self.wind_map_probes[:,0:2] ,np.real(complex_ang_map))
                self.interp_ang_imag = interpolate.LinearNDInterpolator(self.wind_map_probes[:,0:2] ,np.imag(complex_ang_map))
                self.interp_mag = interpolate.LinearNDInterpolator(self.wind_map_probes[:,0:2],self.current_mag_map)
            else: 
                self.interp_ang_real = interpolate.Rbf(self.wind_map_probes[:,0], self.wind_map_probes[:,1],np.real(complex_ang_map))
                self.interp_ang_imag = interpolate.Rbf(self.wind_map_probes[:,0], self.wind_map_probes[:,1],np.imag(complex_ang_map))
                self.interp_mag = interpolate.Rbf(self.wind_map_probes[:,0], self.wind_map_probes[:,1],self.current_mag_map)



    def wind_at_point(self, x, y):
        '''
        Returns the magnitude of the wind velocity vector at the queried point.

                Parameters:
                        x: North position of the queried point
                        y: West position of the queried point

                Returns:
                        ang: wind angle at the x,y points. Angle is in the direction the wind is going. Degrees
                        mag: wind magnitude at the x,y point
        # '''

        ang_r = self.interp_ang_real(x,y)
        ang_i = self.interp_ang_imag(x,y)
        ang = np.rad2deg(np.angle(ang_r + ang_i*1j))
        
        mag = self.interp_mag(x,y)
        if self.method == 'nearest':
            ang = ang[0]
            mag = mag[0]

        return ang, mag

class DrydenSim:
    def __init__(self, Ts, Va):
        #   Dryden gust model parameters (pg 56 of Beard and McLain UAV book)
        # HACK:  Setting Va to a constant value is a hack.  We set a nominal airspeed for the gust model.
        Va = 8
        height = 20
        airspeed = Va * 3.28084
        turbulence_level = 15
        
        
        Lu = height / ((0.177 + 0.00823*height)**(0.2))
        Lv = Lu
        Lw = height
        sigma_w = 0.1 * turbulence_level 
        sigma_u = sigma_w / ((0.177 + 0.000823*height) ** (0.4))
        sigma_v = sigma_u
        
        
        coeff_u = sigma_u*np.sqrt(2*Va/Lu)
        coeff_v = sigma_v*np.sqrt(2*Va/Lv)
        coeff_w = sigma_w*np.sqrt(2*Va/Lw)
        ua = coeff_u
        ub = Va/Lu
        va = coeff_v
        vb = coeff_v * Va/(Lv*np.sqrt(3))
        vc = 2*Va/Lv
        vd = (Va/Lv)**2
        wa = coeff_w
        wb = coeff_w * Va/(Lw*np.sqrt(3))
        wc = 2*Va/Lw
        wd = (Va/Lw)**2

        self._A = np.array([[(-ub), 0., 0., 0., 0.], \
                            [0., (-vc), (-vd), 0., 0.], \
                            [0., 1., 0., 0., 0.], \
                            [0., 0., 0., (-wc), (-wd)], \
                            [0., 0., 0., 1., 0.]])
        self._B = np.array([[1.],[1.],[0.],[1.],[0.]])
        self._C = np.array([[ua, 0., 0., 0., 0.],[0., va, vb, 0., 0.],[0., 0., 0., wa, wb]])
        self._gust_state = np.array([[0., 0., 0., 0., 0.]]).T
        self._Ts = Ts

    def update(self):
        #   The three elements are the gust in the body frame
        return np.concatenate((self._gust()))

    def _gust(self):
        # calculate wind gust using Dryden model.  Gust is defined in the body frame
        w = np.random.randn()  # zero mean unit variance Gaussian (white noise)
        w1 = np.random.randn()  # zero mean unit variance Gaussian (white noise)
        w2 = np.random.randn()  # zero mean unit variance Gaussian (white noise)
        self._gust_state += self._Ts * (self._A @ self._gust_state + self._B * np.array([[w, w1, 0., w2, 0.]]).T)
        return self._C @ self._gust_state