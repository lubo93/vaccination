import numpy as np
import matplotlib.pyplot as plt
from lib.simulation import epidemic_model
from matplotlib import rcParams, colors

# customized settings
params = {  # 'backend': 'ps',
    'font.family': 'serif',
    'font.serif': 'Latin Modern Roman',
    'font.size': 10,
    'axes.labelsize': 'medium',
    'axes.titlesize': 'medium',
    'legend.fontsize': 'medium',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'savefig.dpi': 150,
    'text.usetex': True}
# tell matplotlib about your params
rcParams.update(params)

# set nice figure sizes
fig_width_pt = 2*245    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width*ratio  # height in inches
fig_size = [fig_width, 0.5*fig_width]
rcParams.update({'figure.figsize': fig_size})

### prime/boost protocols
# simulation parameters/initial conditions

beta1dbeta_arr = np.linspace(1e-3, 1, 10)
nu1dnu2_arr = np.linspace(1e-4, 0.999, 10)

f_arr = []
F_arr = []

BETA1dBETA, NU1dNU2 = np.meshgrid(beta1dbeta_arr,nu1dnu2_arr)

beta = 3/14
nu = 5*1e-3

for (pref,nu_pref) in zip(np.ravel(BETA1dBETA), np.ravel(NU1dNU2)):
    
    # simulation parameters/initial conditions
    # [beta, betap, betapp, beta_1, beta_1p, beta_1pp, \
    # beta_2, beta_2p, beta_2pp, nu_1, nu_2, eta_1, eta_2, \
    # gamma, gammap, gammapp, sigma, sigma_1, sigma_2, IFR, IFR1, IFR2]
    params1 = [beta, beta/10, beta/20, pref*beta, pref*beta/10, pref*beta/20, \
    0.05*pref*beta, 0.05*pref*beta/10, 0.05*pref*beta/20, nu, 0, \
    1e-3, 1e-3, 1/6, 1/6, 1/6, 1/5, 1/5, 1/5, 1e-2, 1e-2, 1e-2]
    
    params2 = [beta, beta/10, beta/20, pref*beta, pref*beta/10, pref*beta/20, \
    0.05*pref*beta, 0.05*pref*beta/10, 0.05*pref*beta/20, nu*nu_pref, nu*(1-nu_pref), \
    1e-3, 1e-3, 1/6, 1/6, 1/6, 1/5, 1/5, 1/5, 1e-2, 1e-2, 1e-2]
    
    # [S0, S0p, S0pp, E0, E0p, E0pp, I0, I0p, I0pp, R0, D0]
    initial_conditions = [0.99, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0]
    
    model1 = epidemic_model(params1, initial_conditions)
    model1.simulate()
    
    model2 = epidemic_model(params2, initial_conditions)
    model2.simulate()
    
    if pref >= 1e2 and nu_pref >= 1:
        
        print(model2.delta_d, model1.delta_d)
        print(model2.D_tot, model1.D_tot)
        print(model2.vaccine_total, model1.vaccine_total)
        print(model1.S+model1.Sp+model1.Spp+model1.I+model1.Ip+model1.Ipp+model1.R+model1.D)
        print(model2.S+model2.Sp+model2.Spp+model2.I+model2.Ip+model2.Ipp+model2.R+model2.D)

        fig, ax = plt.subplots()

        plt.plot(model1.t_arr, model1.S_arr, label = r"$S(t)$")
        plt.plot(model2.t_arr, model2.S_arr, linestyle = '--', color = 'grey')
        
        plt.plot(model1.t_arr, model1.Sp_arr, label = r"$S^{\star}(t)$")
        plt.plot(model2.t_arr, model2.Sp_arr, linestyle = '--', color = 'grey')

        plt.plot(model1.t_arr, model1.Spp_arr, label = r"$S^{\star \star}(t)$")
        plt.plot(model2.t_arr, model2.Spp_arr, linestyle = '--', color = 'grey')

        plt.plot(model1.t_arr, model1.I_arr, label = r"$I(t)$")
        plt.plot(model2.t_arr, model2.I_arr, linestyle = '--', color = 'grey')

        plt.plot(model1.t_arr, model1.Ip_arr, label = r"$I^{\star}(t)$")
        plt.plot(model2.t_arr, model2.Ip_arr, linestyle = '--', color = 'grey')

        plt.plot(model1.t_arr, model1.Ipp_arr, label = r"$I^{\star \star}(t)$")
        plt.plot(model2.t_arr, model2.Ipp_arr, linestyle = '--', color = 'grey')

        plt.plot(model1.t_arr, model1.R_arr, label = r"$R(t)$")
        plt.plot(model2.t_arr, model2.R_arr, linestyle = '--', color = 'grey')

        plt.plot(model1.t_arr, model1.D_arr, label = r"$D(t)$")
        plt.plot(model2.t_arr, model2.D_arr, linestyle = '--', color = 'grey')

        plt.legend(frameon = False, fontsize = 8, ncol = 2)
        
        plt.xlabel(r"$t$")
        plt.ylabel(r"proportion")
        plt.ylim(-0.1,1)
        plt.tight_layout()
        plt.margins(0,0)
        plt.savefig('SIR.png', dpi=480, bbox_inches = 'tight',
            pad_inches = 0)
        plt.show()
    
    f_arr.append((model2.delta_d-model1.delta_d)/max(model1.delta_d,model2.delta_d))
    
    F_arr.append((model2.D_tot-model1.D_tot)/max(model1.D_tot,model2.D_tot))
    
f_arr = np.asarray(f_arr)
F_arr = np.asarray(F_arr)

f = f_arr.reshape(BETA1dBETA.shape)    

F = F_arr.reshape(BETA1dBETA.shape)    

print("f", f)

print("F", F)

f = f < 0
F = F < 0

f = np.ma.masked_where(f == False, f)
F = np.ma.masked_where(F == False, F)

# set color for which f,F < 0 is True
cmap = colors.ListedColormap(['#b7241b'])
# set color for which f,F > 0 is False
cmap.set_bad(color='#265500')

fig, ax = plt.subplots(ncols = 2)

ax[0].set_title(r"$f=(d_2-d_1)/\mathrm{max}(d_1,d_2)$")
cm1 = ax[0].pcolormesh(BETA1dBETA, NU1dNU2/(1-NU1dNU2), f, cmap=cmap, alpha = 0.8, linewidth=0, antialiased=True)
ax[0].set_xlabel(r"$\beta_1/\beta$")
ax[0].set_ylabel(r"$\nu_{1}/\nu_{2}$")

ax[1].set_title(r"$F=(D_2-D_1)/\mathrm{max}(D_1,D_2)$")
cm2 = ax[1].pcolormesh(BETA1dBETA, NU1dNU2/(1-NU1dNU2), F, cmap=cmap, alpha = 0.8, linewidth=0, antialiased=True)
ax[1].set_xlabel(r"$\beta_1/\beta$")

#fig.colorbar(cm1, ax=ax[0])
#fig.colorbar(cm2, ax=ax[1])

ax[0].set_xlim([0,1])
ax[1].set_xlim([0,1])
ax[0].set_ylim([1e-1,1e1])
ax[1].set_ylim([1e-1,1e1])
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[1].set_yticks([])

plt.tight_layout()
plt.savefig("nu1nu2_betabeta1_1.png", dpi = 300)