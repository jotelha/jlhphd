import numpy as np
import ase
import matplotlib.pyplot as plt

# default simulation parameters
std_netcdf_output_interval = 1000

# conversion units, only for better readability
fs = 1e-15 # s
ps = 1e-12 # s
AA = 1e-10 # m

dt = 2e-15 # s, 2 fs timestep

# type dict, manually for SDS
sds_t2n = {1: 1, 2: 1, 3: 6, 4: 6, 5: 8, 6: 8, 7: 16, 8: 1, 9: 8, 10: 11, 12: 79}
sds_t2n_array = np.array([0,*list(sds_t2n.values())],dtype='uint64')
sds_t2e_array = np.array(ase.data.chemical_symbols)[sds_t2n_array] # double-check against LAMMPS data file

# type dict, manually for CTAB
ctab_t2n = {1: 1, 2: 1, 3: 1, 4: 6, 5: 6, 6: 6, 7: 7, 8: 1, 9: 8, 10: 35, 11: 79}
ctab_t2n_array = np.array([0,*list(ctab_t2n.values())],dtype='uint64')
ctab_t2e_array = np.array(ase.data.chemical_symbols)[ctab_t2n_array] # double-check against LAMMPS data file

# matplotlib settings

# expecially for presentation, larger font settings for plotting are recommendable
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlex

plt.rcParams["figure.figsize"] = (11,7) # the standard figure size

# numpy truncates the output of large array above the treshold length
np.set_printoptions(threshold=100)
