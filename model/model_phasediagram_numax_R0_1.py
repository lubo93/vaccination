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

beta_arr = np.linspace(1/14, 0.3, 30)
nu_max_arr = np.linspace(0, 1e-3, 30)

f_arr = []
F_arr = []

R_0_arr = []

BETA, NU_MAX = np.meshgrid(beta_arr,nu_max_arr)

for (beta,nu_max) in zip(np.ravel(BETA),np.ravel(NU_MAX)):
    
    params1 = [beta, beta/10, beta/20, 0.8*beta, 0.8*beta/10, 0.8*beta/20, \
    0.1*beta, 0.1*beta/10, 0.1*beta/20, nu_max, 0, \
    1e-3, 1e-3, 1/14, 1/14, 1/14, 1e-2, 1e-2, 1e-2]
    
    params2 = [beta, beta/10, beta/20, 0.8*beta, 0.8*beta/10, 0.8*beta/20, \
    0.1*beta, 0.1*beta/10, 0.1*beta/20, nu_max/2, nu_max/2, \
    1e-3, 1e-3, 1/14, 1/14, 1/14, 1e-2, 1e-2, 1e-2]
    
    initial_conditions = [0.99, 0, 0, 0.01, 0, 0, 0, 0]
    
    model1 = epidemic_model(params1, initial_conditions)
    model1.simulate()
    
    model2 = epidemic_model(params2, initial_conditions)
    model2.simulate()
    
    if model1.reproduction_number > 1e2 and nu_max >= 0.0002:
        
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
        plt.plot(model2.t_arr, model2.Spp_arr, '.', linestyle = '--', color = 'grey')

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
    
    R_0_arr.append(model1.reproduction_number)

f_arr = np.asarray(f_arr)
F_arr = np.asarray(F_arr)
R_0_arr = np.asarray(R_0_arr)

R_0 = R_0_arr.reshape(BETA.shape)   

f = f_arr.reshape(BETA.shape)    

F = F_arr.reshape(BETA.shape)    

print("f", f)

print("F", F)

f = f < 0
F = F < 0

cmap = colors.ListedColormap(['#b7241b','#265500'])

fig, ax = plt.subplots(ncols = 2)

ax[0].set_title(r"$f=(d_2-d_1)/\mathrm{max}(d_1,d_2)$")
cm1 = ax[0].pcolormesh(R_0, NU_MAX, f, cmap=cmap, alpha = 0.8, linewidth=0, antialiased=True)

ax[0].set_xlabel(r"$R_0$")
ax[0].set_ylabel(r"$\nu_{\mathrm{max}}$")

ax[1].set_title(r"$F=(D_2-D_1)/\mathrm{max}(D_1,D_2)$")
cm2 = ax[1].pcolormesh(R_0, NU_MAX, F, cmap=cmap, alpha = 0.8, linewidth=0, antialiased=True)
ax[1].set_xlabel(r"$R_0$")

#fig.colorbar(cm1, ax=ax[0])
#fig.colorbar(cm2, ax=ax[1])

ax[0].set_xlim([1,4])
ax[1].set_xlim([1,4])
ax[0].set_xticks([1,2,3,4])
ax[1].set_xticks([1,2,3,4])
ax[1].set_yticks([])

plt.tight_layout()
plt.savefig("numax_R0_1.png", dpi = 300)