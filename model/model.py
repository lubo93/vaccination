import numpy as np
import matplotlib.pyplot as plt
from lib.simulation import epidemic_model
from matplotlib import rcParams

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
fig_width_pt = 245    # Get this from LaTeX using \showthe\columnwidth
golden_mean = (np.sqrt(5.) - 1.) / 2.  # Aesthetic ratio
ratio = golden_mean
inches_per_pt = 1. / 72.27  # Convert pt to inches
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width*ratio  # height in inches
fig_size = [fig_width, fig_height]
rcParams.update({'figure.figsize': fig_size})

# simulation parameters/initial conditions
params = [0.1, 0.01, 0.001, 0.005, 0.001, 0.0001, 0.01, 0.001, 0.0001, \
0.01, 0.01, 0.001, 0.001, 0.05, 0.1, 0.15, 1e-2, 1e-3, 1e-4]

initial_conditions = [0.99, 0, 0, 0.01, 0, 0, 0, 0]

model = epidemic_model(params, initial_conditions)
model.simulate()

fig, ax = plt.subplots()

plt.plot(model.t_arr, model.S_arr, label = r"$S(t)$")
plt.plot(model.t_arr, model.Sp_arr, label = r"$S^{\star}(t)$")
plt.plot(model.t_arr, model.Spp_arr, label = r"$S^{\star \star}(t)$")
plt.plot(model.t_arr, model.I_arr, label = r"$I(t)$")
plt.plot(model.t_arr, model.Ip_arr, label = r"$I^{\star}(t)$")
plt.plot(model.t_arr, model.Ipp_arr, label = r"$I^{\star \star}(t)$")
plt.plot(model.t_arr, model.R_arr, label = r"$R(t)$")
plt.plot(model.t_arr, model.D_arr, label = r"$D(t)$")

plt.legend(frameon = False, fontsize = 8, ncol = 2)

plt.xlabel(r"$t$")
plt.ylabel(r"proportion")
plt.ylim(0,1)
plt.tight_layout()
plt.margins(0,0)
plt.savefig('SIR.png', dpi=480, bbox_inches = 'tight',
    pad_inches = 0)
plt.show()