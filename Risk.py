import numpy as np
 
class Risk:
    """
    A class to find the risk.

    ...

    Attributes
    ----------
    powers : np.ndarray
        array of power on a path across several monte carlo runs
    limit : float
        value for confidence
    a : float
        coeff a  of risk profile
    b : float
        coeff b  of risk profile
    B : float
        battery capacity

    Methods
    -------
    cvar(new_limit=None):
        Returns the value of risky power value for a path using CVaR
    var(new_limit=None):
        Returns the value of risky power value for a path using VaR
    """
    
    def __init__(self, powers, limit=95.,a = None ,b = 300 ,B = 1200):
        """
        Constructs all the necessary attributes for the risk object.

        Parameters
        ----------
            powers : np.ndarray
                array of power on a path across several monte carlo runs
            limit : float
                value for confidence
        """
        self.powers = powers
        self.limit = limit
        if a == None:
            self.a = np.log(2)*b
        else:
            self.a = a
            
        self.b = b
        self.B = B
        
        self.risk_profile()
        
    def risk_profile(self):
        '''
        Returns the risk value for a path using CVaR.

                Parameters:
                       None

                Returns:
                        risk (np.ndarray): Array of risk values
        '''        

        self.risk = np.exp(np.divide(self.a,np.maximum(self.B-self.powers,self.b))) - 1
        
        
        
    def cvar(self, new_limit=None):
        '''
        Returns the value of risky power value for a path using CVaR.

                Parameters:
                        new_limit (float): percentage of confidence

                Returns:
                        cvar (float): Value for risk power based on cvar
        '''
        
        if new_limit != None:
            self.limit = new_limit
        
        assert self.limit < 100

        var = np.percentile(self.risk, self.limit)
        cvar = self.risk[self.risk >= var].mean()
        
        return cvar
    
    def var(self, new_limit=None):
        '''
        Returns the value of risky power value for a path using VaR.

                Parameters:
                        new_limit (float): percentage of confidence

                Returns:
                        var (float): Value for risk power based on cvar
        '''
        
        if new_limit != None:
            self.limit = new_limit
        
        assert self.limit < 100

        var = np.percentile(self.risk, self.limit)
        
        return var