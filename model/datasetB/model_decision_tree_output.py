import numpy as np
import random
import sys
from scipy.stats import loguniform
from lib.simulation import epidemic_model

# fix seed
seed = random.randrange(sys.maxsize)
random.seed(seed)

### prime/boost protocols
# simulation parameters/initial conditions
gamma = 1/14
beta_min = gamma
beta_max = 4*gamma

beta_distr = lambda beta_min, beta_max: random.uniform(beta_min, beta_max)
betap_distr = lambda beta: random.uniform(0, beta)
betapp_distr = lambda betap: random.uniform(0, betap)

beta_1_distr = lambda beta: random.uniform(0, beta)
beta_1p_distr = lambda beta_1: random.uniform(0, beta_1)
beta_1pp_distr = lambda beta_1p: random.uniform(0, beta_1p)

beta_2_distr = lambda beta_1: random.uniform(0, beta_1)
beta_2p_distr = lambda beta_2: random.uniform(0, beta_2)
beta_2pp_distr = lambda beta_2p: random.uniform(0, beta_2p)

nu_max_distr = lambda: random.uniform(0, 1.3e-2)

eta_1_distr = lambda: random.uniform(0, 0.1)
eta_2_distr = lambda eta_1: random.uniform(0, eta_1)
delta_eta_distr = lambda eta_1: random.uniform(0, min(eta_1,1.7e-2))

gammap_distr = lambda gamma: random.uniform(gamma, 2*gamma)
gammapp_distr = lambda gammap: random.uniform(gammap, 2*gammap)

sigma_distr = lambda: random.uniform(1/5, 1/2)
sigma_1_distr = lambda: random.uniform(1/5, 1/2)
sigma_2_distr = lambda: random.uniform(1/5, 1/2)

IFR_distr = lambda: random.uniform(1e-3, 1e-1)
IFR1_distr = lambda IFR: random.uniform(1e-3, IFR)

I0_distr = lambda: loguniform.rvs(1e-7, 3e-1)
S0p_distr = lambda: random.uniform(1e-4, 0.1)
S0pp_distr = lambda: random.uniform(1e-4, 0.1)

td_distr = lambda: random.choice(np.arange(7,35))

samples = int(1000)

file = open("decision_tree_output.dat", "w")
file.write("#seed: %d \n"%seed)
file.write("beta,betap,betapp,beta_1,beta_1p,beta_1pp,\
beta_2,beta_2p,beta_2pp,nu_max,eta_1,eta_2,\
gamma,gammap,gammapp,sigma,sigma_1,sigma_2,IFR,IFR1,IFR2,td,\
S0,S0p,S0pp,E0,E0p,E0pp,I0,I0p,I0pp,R0,D0,d1,d2,D1,D2,\
delta < 0,Delta < 0\n")

for i in range(samples):
    
    beta = beta_distr(beta_min, beta_max)
    betap = betap_distr(beta)
    betapp = betapp_distr(betap)

    beta_1 = beta_1_distr(beta)
    beta_1p = beta_1p_distr(beta_1)
    beta_1pp = beta_1pp_distr(beta_1p)
    
    beta_2 = beta_2_distr(beta_1)
    beta_2p = beta_2p_distr(beta_1p)
    beta_2pp = beta_2pp_distr(beta_1pp)
    
    nu_max = nu_max_distr()

    eta_1 = eta_1_distr()
    delta_eta = delta_eta_distr(eta_1)
    eta_2 = eta_1 - delta_eta
    
    gammap = gammap_distr(gamma)
    gammapp = gammapp_distr(gammap)
    
    sigma = sigma_distr()
    sigma_1 = sigma_1_distr()
    sigma_2 = sigma_2_distr()
    
    IFR = IFR_distr()
    IFR1 = IFR1_distr(IFR)
    IFR2 = IFR1
    
    td = td_distr()
    
    # simulation parameters/initial conditions
    # [beta, betap, betapp, beta_1, beta_1p, beta_1pp, \
    # beta_2, beta_2p, beta_2pp, nu_1, nu_2, eta_1, eta_2, \
    # gamma, gammap, gammapp, sigma, sigma_1, sigma_2, IFR, IFR1, IFR2, td]
    params1 = [beta, betap, betapp, beta_1, beta_1p, beta_1pp, \
    beta_2, beta_2p, beta_2pp, nu_max, 0, eta_1, eta_2, \
    gamma, gammap, gammapp, sigma, sigma_1, sigma_2, IFR, IFR1, IFR2, td]
    
    params2 = [beta, betap, betapp, beta_1, beta_1p, beta_1pp, \
    beta_2, beta_2p, beta_2pp, nu_max/2, nu_max/2, eta_1, eta_2, \
    gamma, gammap, gammapp, sigma, sigma_1, sigma_2, IFR, IFR1, IFR2, td]
    
    # [S0, S0p, S0pp, E0, E0p, E0pp, I0, I0p, I0pp, R0, D0]
    I0 = I0_distr()
    S0p = S0p_distr()
    S0pp = S0pp_distr()

    S0 = 1-I0-S0p-S0pp
    E0 = 0
    E0p = 0
    E0pp = 0
    I0p = 0
    I0pp = 0
    R0 = 0
    D0 = 0
    
    initial_conditions = [S0, S0p, S0pp, E0, E0p, E0pp, I0, I0p, I0pp, R0, D0]
        
    model1 = epidemic_model(params1, 
                            initial_conditions,
                            time_step = 1e-1,
                            duration = 300,
                            Euler = False)
    model1.simulate()
    
    model2 = epidemic_model(params2, 
                            initial_conditions,
                            time_step = 1e-1,
                            duration = 300,
                            Euler = False)
    model2.simulate()
        
    f = (model2.delta_d-model1.delta_d)/max(model1.delta_d,model2.delta_d)
    F = (model2.D_tot-model1.D_tot)/max(model1.D_tot,model2.D_tot)
    
    print("f", f)
    
    print("F", F)
    
    file.write("%1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, \
%1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, \
%1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, \
%1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %1.9f, %d, %d \n" %
    (beta, betap, betapp, beta_1, beta_1p, beta_1pp, \
    beta_2, beta_2p, beta_2pp, nu_max, eta_1, eta_2, \
    gamma, gammap, gammapp, sigma, sigma_1, sigma_2, IFR, IFR1, IFR2, td, \
    S0, S0p, S0pp, E0, E0p, E0pp, I0, I0p, I0pp, R0, D0, \
    model1.delta_d, model2.delta_d, model1.D_tot, model2.D_tot, \
    int(f < 0), int(F < 0)))
    
file.close()
