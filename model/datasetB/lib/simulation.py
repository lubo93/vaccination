from numpy import linspace, heaviside, asarray, diff
from scipy.integrate import solve_ivp

# epidemic model class
class epidemic_model:   
    """
    A class to model SEIRD dynamics with prime and prime-boost vaccination.

    """
    def __init__(self,
                 params,
                 initial_conditions,
                 time_step,
                 duration,
                 Euler = True):   
        
        # rate parameters
        self.beta = params[0]
        self.betap = params[1]    
        self.betapp = params[2]    
        self.beta_1 = params[3]   
        self.beta_1p = params[4]    
        self.beta_1pp = params[5]    
        self.beta_2 = params[6] 
        self.beta_2p = params[7]    
        self.beta_2pp = params[8]
        self.nu_1 = params[9]
        self.nu_2 = params[10]
        self.eta_1 = params[11] 
        self.eta_2 = params[12]
        self.gamma = params[13]
        self.gammap = params[14]
        self.gammapp = params[15] 
        self.sigma = params[16]
        self.sigma_1 = params[17]
        self.sigma_2 = params[18]

        # reproduction number
        self.reproduction_number = self.beta/self.gamma
        
        print("(R0=)beta/gamma =",self.reproduction_number)        
        
        # infection fatality ratios for different vaccination stages
        self.IFR = params[19] 
        self.IFRp = params[20] 
        self.IFRpp = params[21] 
        
        self.td = params[22]
        
        # initial conditions
        self.S0 = initial_conditions[0]
        self.S0p = initial_conditions[1]
        self.S0pp = initial_conditions[2]
        self.E0 = initial_conditions[3]
        self.E0p = initial_conditions[4]
        self.E0pp = initial_conditions[5]
        self.I0 = initial_conditions[6]
        self.I0p = initial_conditions[7]
        self.I0pp = initial_conditions[8]
        self.R0 = initial_conditions[9]
        self.D0 = initial_conditions[10]

        # simulation arrays to store the evolution of all compartments
        self.S_arr = []
        self.Sp_arr = []
        self.Spp_arr = []
        self.E_arr = []
        self.Ep_arr = []
        self.Epp_arr = []
        self.I_arr = []
        self.Ip_arr = []
        self.Ipp_arr = []
        self.R_arr = []
        self.D_arr = []
        self.np_arr = []
        self.npp_arr = []

        # set ininitial conditions
        self.S = self.S0
        self.Sp = self.S0p
        self.Spp = self.S0pp
        self.E = self.E0
        self.Ep = self.E0p
        self.Epp = self.E0pp
        self.I = self.I0
        self.Ip = self.I0p
        self.Ipp = self.I0pp
        self.R = self.R0
        self.D = self.D0
        self.np = 0
        self.npp = 0

        self.S_arr.append(self.S)
        self.Sp_arr.append(self.Sp)
        self.Spp_arr.append(self.Spp)
        self.E_arr.append(self.E)
        self.Ep_arr.append(self.Ep)
        self.Epp_arr.append(self.Epp)
        self.I_arr.append(self.I)
        self.Ip_arr.append(self.Ip)
        self.Ipp_arr.append(self.Ipp)
        self.R_arr.append(self.R)
        self.D_arr.append(self.D)
        self.np_arr.append(self.np)
        self.npp_arr.append(self.npp)

        # simulation parameters
        self.dt = time_step
        self.t = 0
        self.T = duration
        self.N = int(self.T/self.dt)
        
        self.t_arr = linspace(0, self.T, self.N+1)
        
        self.Euler = Euler
        
        # maximum change of deaths and total deaths
        self.delta_d = 0
        self.D_tot = 0          
        
        # total vaccination count
        self.vaccine_total = 0
    
    def step_EULER(self):
        """
        Euler forward integration step of SEIRD dynamics with prime and
        prime-boost vaccination.
        """
        
        self.vaccine_total += self.dt*((self.nu_1+self.nu_2)*heaviside(self.S,0) * \
        heaviside(self.td-self.t,0) + self.nu_1*heaviside(self.S,0)*heaviside(self.t-self.td,0) + \
        self.nu_2*heaviside(self.Sp,0)*heaviside(self.t-self.td,0)+\
        self.nu_1*heaviside(self.Sp,0)*heaviside(self.t-self.td,0)*(1-heaviside(self.S,0)))        
                
        delta_S = -self.beta*self.S*self.I -  \
        self.betap*self.S*self.Ip - self.betapp*self.S*self.Ipp - \
        (self.nu_1+self.nu_2)*heaviside(self.S,0)*heaviside(self.td-self.t,0) - \
        self.nu_1*heaviside(self.S,0)*heaviside(self.t-self.td,0) + self.eta_1*self.Sp + \
        self.eta_2*self.Spp
        
        delta_Sp = (self.nu_1+self.nu_2)*heaviside(self.S,0)*heaviside(self.td-self.t,0) + \
        self.nu_1*heaviside(self.S,0)*heaviside(self.t-self.td,0) - self.beta_1*self.Sp*self.I -  \
        self.beta_1p*self.Sp*self.Ip - self.beta_1pp*self.Sp*self.Ipp -\
        self.nu_2*heaviside(self.Sp,0)*heaviside(self.t-self.td,0) - self.eta_1*self.Sp-\
        self.nu_1*heaviside(self.Sp,0)*heaviside(self.t-self.td,0)*(1-heaviside(self.S,0))
        
        delta_Spp = self.nu_2*heaviside(self.Sp,0)*heaviside(self.t-self.td,0) - \
        self.beta_2*self.Spp*self.I - self.beta_2p*self.Spp*self.Ip - \
        self.beta_2pp*self.Spp*self.Ipp - self.eta_2*self.Spp + \
        self.nu_1*heaviside(self.Sp,0)*heaviside(self.t-self.td,0)*(1-heaviside(self.S,0))
        
        delta_E = self.beta*self.S*self.I + self.betap*self.S*self.Ip + \
        self.betapp*self.S*self.Ipp - self.sigma*self.E
        
        delta_Ep = self.beta_1*self.Sp*self.I + self.beta_1p*self.Sp*self.Ip + \
        self.beta_1pp*self.Sp*self.Ipp - self.sigma_1*self.Ep
        
        delta_Epp = self.beta_2*self.Spp*self.I + \
        self.beta_2p*self.Spp*self.Ip + self.beta_2pp*self.Spp*self.Ipp - \
        self.sigma_2*self.Epp
        
        delta_I = self.sigma*self.E - self.gamma*self.I
        
        delta_Ip = self.sigma_1*self.Ep - self.gammap*self.Ip
        
        delta_Ipp = self.sigma_2*self.Epp - self.gammapp*self.Ipp
        
        delta_R = self.gamma*(1-self.IFR)*self.I + \
        self.gammap*(1-self.IFRp)*self.Ip + \
        self.gammapp*(1-self.IFRpp)*self.Ipp
        
        delta_D = self.gamma*self.IFR*self.I + \
        self.gammap*self.IFRp*self.Ip + \
        self.gammapp*self.IFRpp*self.Ipp
        
        # Euler forward integration steps
        S_new = self.S + self.dt*delta_S
        
        Sp_new = self.Sp + self.dt*delta_Sp
        
        Spp_new = self.Spp + self.dt*delta_Spp
        
        E_new = self.E + self.dt*delta_E
        
        Ep_new = self.Ep + self.dt*delta_Ep
        
        Epp_new = self.Epp + self.dt*delta_Epp

        I_new = self.I + self.dt*delta_I

        Ip_new = self.Ip + self.dt*delta_Ip

        Ipp_new = self.Ipp + self.dt*delta_Ipp

        R_new = self.R + self.dt*delta_R
        
        D_new = self.D + self.dt*delta_D
        
        # determine maximum rate of change in the number of deaths
        if D_new-self.D > self.delta_d:
            self.delta_d = D_new-self.D
        
        # update current compartment values
        self.S = S_new
        self.Sp = Sp_new
        self.Spp = Spp_new
        self.E = E_new
        self.Ep = Ep_new
        self.Epp = Epp_new
        self.I = I_new
        self.Ip = Ip_new
        self.Ipp = Ipp_new
        self.R = R_new
        self.D = D_new
        
    def DOPRI(self):
        """
        Dormand--Prince integration of SEIRD dynamics with prime and
        prime-boost vaccination.
        """
        
        delta_S = lambda t, y: -self.beta*y[0]*y[6] -  \
        self.betap*y[0]*y[7] - self.betapp*y[0]*y[8] - \
        (self.nu_1+self.nu_2)*heaviside(y[0],0)*heaviside(self.td-t,0) - \
        self.nu_1*heaviside(y[0],0)*heaviside(t-self.td,0) \
        + self.eta_1*y[1] + self.eta_2*y[2]
        
        delta_Sp = lambda t, y: (self.nu_1+self.nu_2)*heaviside(y[0],0)*heaviside(self.td-t,0) + \
        self.nu_1*heaviside(y[0],0)*heaviside(t-self.td,0) - self.beta_1*y[1]*y[6] -  \
        self.beta_1p*y[1]*y[7] - self.beta_1pp*y[1]*y[8] - \
        self.nu_2*heaviside(y[1],0)*heaviside(t-self.td,0) - \
        self.eta_1*y[1] - self.nu_1*heaviside(y[1],0)*heaviside(t-self.td,0)*(1-heaviside(y[0],0))
        
        delta_Spp = lambda t, y: self.nu_2*heaviside(y[1],0)*heaviside(t-self.td,0) - \
        self.beta_2*y[2]*y[6] - self.beta_2p*y[2]*y[7] - \
        self.beta_2pp*y[2]*y[8] - self.eta_2*y[2] + \
        self.nu_1*heaviside(y[1],0)*heaviside(t-self.td,0)*(1-heaviside(y[0],0))
        
        delta_E = lambda t, y: self.beta*y[0]*y[6] + self.betap*y[0]*y[7] + \
        self.betapp*y[0]*y[8] - self.sigma*y[3]
        
        delta_Ep = lambda t, y: self.beta_1*y[1]*y[6] + self.beta_1p*y[1]*y[7] + \
        self.beta_1pp*y[1]*y[8] - self.sigma_1*y[4]
        
        delta_Epp = lambda t, y: self.beta_2*y[2]*y[6] + \
        self.beta_2p*y[2]*y[7] + self.beta_2pp*y[2]*y[8] - \
        self.sigma_2*y[5]
        
        delta_I = lambda t, y: self.sigma*y[3] - self.gamma*y[6]
        
        delta_Ip = lambda t, y: self.sigma_1*y[4] - self.gammap*y[7]
        
        delta_Ipp = lambda t, y: self.sigma_2*y[5] - self.gammapp*y[8]
        
        delta_R = lambda t, y: self.gamma*(1-self.IFR)*y[6] + \
        self.gammap*(1-self.IFRp)*y[7] + \
        self.gammapp*(1-self.IFRpp)*y[8]
        
        delta_D = lambda t, y: self.gamma*self.IFR*y[6] + \
        self.gammap*self.IFRp*y[7] + \
        self.gammapp*self.IFRpp*y[8]
        
        delta_vaccine = lambda t, y: (self.nu_1+self.nu_2)*heaviside(y[0],0) * \
        heaviside(self.td-t,0) + self.nu_1*heaviside(y[0],0)*heaviside(t-self.td,0) + \
        self.nu_2*heaviside(y[1],0)*heaviside(t-self.td,0) + \
        self.nu_1*heaviside(y[1],0)*heaviside(t-self.td,0)*(1-heaviside(y[0],0))
        
        delta_np = lambda t, y: (self.nu_1+self.nu_2)*heaviside(y[0],0)*heaviside(self.td-t,0) + \
        self.nu_1*heaviside(y[0],0)*heaviside(t-self.td,0) - \
        self.nu_2*heaviside(y[1],0)*heaviside(t-self.td,0) - \
        self.nu_1*heaviside(y[1],0)*heaviside(t-self.td,0)*(1-heaviside(y[0],0))
        
        delta_npp = lambda t, y: self.nu_2*heaviside(y[1],0)*heaviside(t-self.td,0) + \
        self.nu_1*heaviside(y[1],0)*heaviside(t-self.td,0)*(1-heaviside(y[0],0))

        
        f = lambda t, y: [delta_S(t, y), delta_Sp(t, y), delta_Spp(t, y), 
                          delta_E(t, y), delta_Ep(t, y), delta_Epp(t, y),
                          delta_I(t, y), delta_Ip(t, y), delta_Ipp(t, y),
                          delta_R(t, y), delta_D(t, y), delta_vaccine(t, y), 
                          delta_np(t, y), delta_npp(t, y)]
        
        y0 = [self.S, self.Sp, self.Spp, self.E, self.Ep, self.Epp, \
              self.I, self.Ip, self.Ipp, self.R, self.D, self.vaccine_total,
              self.np, self.npp]
                
        sol = solve_ivp(f, 
                        [0,self.T], 
                        y0,
                        method='RK45', 
                        t_eval=self.t_arr,
                        max_step=self.dt)
             
        # determine maximum rate of change in the number of deaths
        self.delta_d = diff(sol.y[10]).max()
        
        # update current compartment values
        self.S = sol.y[0][-1]
        self.Sp = sol.y[1][-1]
        self.Spp = sol.y[2][-1]
        self.E = sol.y[3][-1]
        self.Ep = sol.y[4][-1]
        self.Epp = sol.y[5][-1]
        self.I = sol.y[6][-1]
        self.Ip = sol.y[7][-1]
        self.Ipp = sol.y[8][-1]
        self.R = sol.y[9][-1]
        self.D = sol.y[10][-1]
        self.vaccine_total = sol.y[11][-1]
        self.np = sol.y[12][-1]
        self.npp = sol.y[13][-1]
        
        self.S_arr = sol.y[0]
        self.Sp_arr = sol.y[1]
        self.Spp_arr = sol.y[2]
        self.E_arr = sol.y[3]
        self.Ep_arr = sol.y[4]
        self.Epp_arr = sol.y[5]
        self.I_arr = sol.y[6]
        self.Ip_arr = sol.y[7]
        self.Ipp_arr = sol.y[8]
        self.R_arr = sol.y[9]
        self.D_arr = sol.y[10]
        self.np_arr = sol.y[12]
        self.npp_arr = sol.y[13]

    def simulate(self):
        """
        Simulate SEIRD dynamics with prime and prime-boost 
        vaccination over N time steps.
        """
        
        if self.Euler == True:
            
            for i in range(self.N):
                
                    self.step_EULER()
                    self.t += self.dt
                    self.S_arr.append(self.S)
                    self.Sp_arr.append(self.Sp)
                    self.Spp_arr.append(self.Spp)
                    self.E_arr.append(self.E)
                    self.Ep_arr.append(self.Ep)
                    self.Epp_arr.append(self.Epp)
                    self.I_arr.append(self.I)
                    self.Ip_arr.append(self.Ip)
                    self.Ipp_arr.append(self.Ipp)
                    self.R_arr.append(self.R)
                    self.D_arr.append(self.D)
            
            self.S_arr = asarray(self.S_arr)
            self.Sp_arr = asarray(self.Sp_arr)
            self.Spp_arr = asarray(self.Spp_arr)
            self.E_arr = asarray(self.E_arr)
            self.Ep_arr = asarray(self.Ep_arr)
            self.Epp_arr = asarray(self.Epp_arr)
            self.I_arr = asarray(self.I_arr)
            self.Ip_arr = asarray(self.Ip_arr)
            self.Ipp_arr = asarray(self.Ipp_arr)
            self.R_arr = asarray(self.R_arr)
            self.D_arr = asarray(self.D_arr)
        
            self.D_tot = self.D_arr[-1]
        
        else:
            
            self.DOPRI()
                
            self.D_tot = self.D_arr[-1]

