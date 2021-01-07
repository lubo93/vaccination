import numpy as np

# epidemic model class
class epidemic_model:   
      
    def __init__(self, params, initial_conditions):   
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
        
        self.reproduction_number = self.beta/self.gamma
        
        print("(R0=)beta/gamma =",self.reproduction_number)        
        
        self.IFR = params[16] 
        self.IFRp = params[17] 
        self.IFRpp = params[18] 
        
        self.S0 = initial_conditions[0]
        self.S0p = initial_conditions[1]
        self.S0pp = initial_conditions[2]
        self.I0 = initial_conditions[3]
        self.I0p = initial_conditions[4]
        self.I0pp = initial_conditions[5]
        self.R0 = initial_conditions[6]
        self.D0 = initial_conditions[7]

        self.S_arr = []
        self.Sp_arr = []
        self.Spp_arr = []
        self.I_arr = []
        self.Ip_arr = []
        self.Ipp_arr = []
        self.R_arr = []
        self.D_arr = []
        
        self.S = self.S0
        self.Sp = self.S0p
        self.Spp = self.S0pp
        self.I = self.I0
        self.Ip = self.I0p
        self.Ipp = self.I0pp
        self.R = self.R0
        self.D = self.D0

        self.S_arr.append(self.S)
        self.Sp_arr.append(self.Sp)
        self.Spp_arr.append(self.Spp)
        self.I_arr.append(self.I)
        self.Ip_arr.append(self.Ip)
        self.Ipp_arr.append(self.Ipp)
        self.R_arr.append(self.R)
        self.D_arr.append(self.D)

        self.dt = 1e-1
        self.T = 300
        self.N = int(self.T/self.dt)
        
        self.t_arr = np.linspace(0, self.T, self.N+1)
        
        # maximum change of deaths and total deaths
        self.delta_d = 0
        self.D_tot = 0          
        
        self.vaccine_total = 0
        
    def step(self):
        
        self.vaccine_total += self.dt*(self.nu_1*self.S+self.nu_2*self.Sp)        
                
        delta_S = (-self.beta*self.S*self.I -  \
        self.betap*self.S*self.Ip - self.betapp*self.S*self.Ipp - \
        self.nu_1 + self.eta_1*self.Sp + \
        self.eta_2*self.Spp)
        
        delta_Sp = (self.nu_1 - self.beta_1*self.Sp*self.I -  \
        self.beta_1p*self.Sp*self.Ip - self.beta_1pp*self.Sp*self.Ipp - \
        self.nu_2 -self.eta_1*self.Sp)
        
        delta_Spp = (self.nu_2 - self.beta_2*self.Spp*self.I -  \
        self.beta_2p*self.Spp*self.Ip - self.beta_2pp*self.Spp*self.Ipp - \
        self.eta_2*self.Spp)
        
        delta_I = (self.beta*self.S*self.I + \
        self.betap*self.S*self.Ip + self.betapp*self.S*self.Ipp - \
        self.gamma*self.I)
        
        delta_Ip = (self.beta_1*self.Sp*self.I + \
        self.beta_1p*self.Sp*self.Ip + self.beta_1pp*self.Sp*self.Ipp - \
        self.gammap*self.Ip)
        
        delta_Ipp = (self.beta_2*self.Spp*self.I + \
        self.beta_2p*self.Spp*self.Ip + self.beta_2pp*self.Spp*self.Ipp - \
        self.gammapp*self.Ipp)
        
        delta_R = (self.gamma*(1-self.IFR)*self.I + \
        self.gammap*(1-self.IFRp)*self.Ip + \
        self.gammapp*(1-self.IFRpp)*self.Ipp)
        
        delta_D = (self.gamma*self.IFR*self.I + \
        self.gammap*self.IFRp*self.Ip + \
        self.gammapp*self.IFRpp*self.Ipp)
                
        S_new = self.S + self.dt*delta_S
        
        Sp_new = self.Sp + self.dt*delta_Sp
        
        Spp_new = self.Spp + self.dt*delta_Spp

        I_new = self.I + self.dt*delta_I

        Ip_new = self.Ip + self.dt*delta_Ip

        Ipp_new = self.Ipp + self.dt*delta_Ipp

        R_new = self.R + self.dt*delta_R
        
        D_new = self.D + self.dt*delta_D
        
        # determine maximum rate of change in the number of deaths
        if D_new-self.D > self.delta_d:
            self.delta_d = D_new-self.D
        
        self.S = S_new
        self.Sp = Sp_new
        self.Spp = Spp_new
        self.I = I_new
        self.Ip = Ip_new
        self.Ipp = Ipp_new
        self.R = R_new
        self.D = D_new

    def simulate(self):
        
        for i in range(self.N):
            self.step()
            self.S_arr.append(self.S)
            self.Sp_arr.append(self.Sp)
            self.Spp_arr.append(self.Spp)
            self.I_arr.append(self.I)
            self.Ip_arr.append(self.Ip)
            self.Ipp_arr.append(self.Ipp)
            self.R_arr.append(self.R)
            self.D_arr.append(self.D)

        self.S_arr = np.asarray(self.S_arr)
        self.Sp_arr = np.asarray(self.Sp_arr)
        self.Spp_arr = np.asarray(self.Spp_arr)
        self.I_arr = np.asarray(self.I_arr)
        self.Ip_arr = np.asarray(self.Ip_arr)
        self.Ipp_arr = np.asarray(self.Ipp_arr)
        self.R_arr = np.asarray(self.R_arr)
        self.D_arr = np.asarray(self.D_arr)
        
        self.D_tot = self.D_arr[-1]