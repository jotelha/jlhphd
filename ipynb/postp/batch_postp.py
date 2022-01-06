#!/usr/bin/env python
# coding: utf-8
# %%

# # Collect results

# ## Init

# ### Imports and options

# %%


get_ipython().run_line_magic('load_ext', 'autoreload')


# %%


get_ipython().run_line_magic('autoreload', '2')


# %%


get_ipython().run_line_magic('config', 'Application.log_level="WARN"')


# %%


get_ipython().run_line_magic('matplotlib', 'inline')


# %%


import os
import sys
from glob import glob
import io
import tarfile


# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%


import ase
from asap3.analysis.rdf import RadialDistributionFunction

# file formats, input - output
import ase.io
from ase.io import read
from ase.io import NetCDFTrajectory

import scipy.constants as C


# %%


import postprocessing


# %%


from fireworks.utilities.filepad import FilePad 


# %%


fp = FilePad(
    host='localhost',
    port=27018,
    database='fireworks-jhoermann',
    username='fireworks',
    password='fireworks')


# %%


content, doc = fp.get_file(identifier='surfactant_on_AU_111_df_json')


# %%


sim_df = pd.read_json(content, orient='index')


# ### Poster plotting

# %%


# matplotlib settings

# expecially for presentation, larger font settings for plotting are recommendable
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 26

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlex

plt.rcParams["figure.figsize"] = (16,10) # the standard figure size

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 14
plt.rcParams["lines.markeredgewidth"]=1 


# ### Constants and factors

# ### Forces
# 
# $ [F_{LMP}] = \frac{ \mathrm{kcal}}{ \mathrm{mol} \cdot \mathrm{\mathring{A}}} $
# 
# $ [F_{PLT}] = \mathrm{nN}$
# 
# $ \mathrm{kcal} = 4.184 \mathrm{kJ} = 4.184 \cdot 10^{3} \mathrm{J}$
# 
# $ J = N \cdot m$
# 
# $ N = J m^{-1} = \frac{ \mathrm{kcal}}{ 4.184 \cdot 10^3 \mathrm{m} }
#     = \frac{ \mathrm{kcal}}{ 4.184 \cdot 10^{13} \mathrm{\mathring{A}} }$
#     
# $ [F_{PLT}] = \mathrm{nN} 
#     = \frac{ 10^9 \cdot N_A^{-1}}{ 4.184 \cdot 10^{13} } 
#         \frac{\mathrm{kcal}}{\mathrm{mol} \cdot \mathrm{\mathring{A}}}
#     = \frac{ 10^{-4} \cdot N_A^{-1}}{ 4.184 } [F_{LMP}]
#     = 0.239 \cdot 10^{-4} N_A^{-1} [F_{LMP}] $
#     
#     
# $ \frac{\mathrm{ kcal }}{ {\mathrm{mol} \mathrm{\mathring{A}}}} = 1.66053892103219 \cdot 10^{-11} \frac{\mathrm{J}}{\mathrm{m}}$

# %%


C.calorie # kCal -> kJ


# %%


C.calorie * 1e3 # kCal -> J


# %%


C.calorie * 1e3 / C.angstrom # kCal / Ang -> J / m


# %%


C.calorie * 1e3 / C.angstrom *1e9 # kCal / Ang -> n J / m = n N


# %%


force_conversion_factor = C.calorie * 1e3 / C.angstrom *1e9 / C.Avogadro# kCal / (mol * Ang ) -> n N


# %%


force_conversion_factor


# %%


# force_conversion_factor_per_mole = 1*C.calorie


# %%


# force_conversion_factor_per_mole


# %%


# force_conversion_factor_absolute = force_conversion_factor_per_mole/C.Avogadro


# %%


# force_conversion_factor_absolute


# ### Constant velocity force-distance curves

# %%


def constant_velocity_force_distance_curve_from_thermo_ave(
    thermo_ave_out_file = None,
    thermo_out_file = None,
    initial_distance = 7.5, # nm
    total_steps = 375000,
    averaging_steps = 1000,
    dt = 2*1e-6, # ns,
    force_label         = r'$F \ [ \mathrm{nN} ]$',
    distance_label      = r'$d \ [ \mathrm{nm} ]$',
    window = 1,
    legend_prefix = None,
    interval = slice(None) ):

    # assume LAMMPS "real" units:
    # [ force ] = Kcal/mole-Angstrom
    # conversion factor to nN (nano-Newton):
    force_conversion_factor = C.calorie * 1e3 / C.angstrom *1e9 / C.Avogadro# kCal / (mol * Ang ) -> n N
    total_time = total_steps*dt
    velocity_SI = initial_distance / total_time
    
    averaging_time = averaging_steps * dt * 1e3 # ps
    
    production_thermo_pd = None
    production_thermo_ave_pd = None
    
    if thermo_out_file:
        production_thermo_pd = pd.read_csv(thermo_out_file,delim_whitespace=True)

        production_thermo_pd["distance"] =             initial_distance - production_thermo_pd["Step"] * velocity_SI * dt

        production_thermo_pd.set_index('distance',inplace=True)

        production_thermo_pd["indenter_non_indenter_interaction[3]"] =             production_thermo_pd["c_indenter_substrate_interaction[3]"] +             production_thermo_pd["c_indenter_surfactant_interaction[3]"] +             production_thermo_pd["c_indenter_solvent_interaction[3]"] +             production_thermo_pd["c_indenter_ion_interaction[3]"]
        plt.plot(
        force_conversion_factor*production_thermo_pd["indenter_non_indenter_interaction[3]"],
        label = "instantaneous force")
            
    if thermo_ave_out_file:
        header = pd.read_csv(thermo_ave_out_file,delim_whitespace=True,nrows=0,skiprows=1)
        columns = header.columns[1:]
        production_thermo_ave_pd = pd.read_csv( thermo_ave_out_file, delim_whitespace=True, header=None, comment='#',
            names=columns)
        production_thermo_ave_pd["distance"] =             initial_distance - production_thermo_ave_pd["TimeStep"] * velocity_SI * dt

        production_thermo_ave_pd.set_index('distance',inplace=True)

        production_thermo_ave_pd["indenter_non_indenter_interaction[3]"] =             production_thermo_ave_pd["c_indenter_substrate_interaction[3]"] +             production_thermo_ave_pd["c_indenter_surfactant_interaction[3]"] +             production_thermo_ave_pd["c_indenter_solvent_interaction[3]"] +             production_thermo_ave_pd["c_indenter_ion_interaction[3]"]
        plt.plot(
        force_conversion_factor*production_thermo_ave_pd["indenter_non_indenter_interaction[3]"],
        label = "{:.1f} ps average".format(averaging_time))    
   
    plt.xlabel(distance_label)
    plt.ylabel(force_label)
    plt.legend()
    return production_thermo_pd, production_thermo_ave_pd


# %%


def constant_velocity_force_distance_curve_from_force_file(
    force_file,
    initial_distance = 7.5, # nm
    total_steps = 375000,
    averaging_steps = 1000,
    dt = 2*1e-6, # ns,
    force_label         = r'$F \ [ \mathrm{nN} ]$',
    distance_label      = r'$d \ [ \mathrm{nm} ]$',
    window = 1,
    legend_prefix = None,
    interval = slice(None) ):

    # assume LAMMPS "real" units:
    # [ force ] = Kcal/mole-Angstrom
    # conversion factor to nN (nano-Newton):
    force_conversion_factor = C.calorie * 1e3 / C.angstrom *1e9 / C.Avogadro# kCal / (mol * Ang ) -> n N
    total_time = total_steps*dt
    velocity_SI = initial_distance / total_time
    
    averaging_time = averaging_steps * dt * 1e3 # ps
    
    indenter_forces_df = pd.read_csv(force_file,index_col=0, delim_whitespace=True) 

    indenter_forces_df["distance"] =         initial_distance - indenter_forces_df.index * velocity_SI * dt

    indenter_forces_df.set_index('distance',inplace=True)


    plt.plot(
        force_conversion_factor*indenter_forces_df["f_storeUnconstrainedForces"],
        label = "instantaneous force")

    plt.plot(
        force_conversion_factor*indenter_forces_df["f_storeUnconstrainedForcesAve"],
        label = "{:.1f} ps average".format(averaging_time))    
   
    plt.xlabel(distance_label)
    plt.ylabel(force_label)
    plt.legend()
    return indenter_forces_df 


# %%


averaging_time = 2.0
force_label         = r'$F \ [ \mathrm{nN} ]$'
distance_label      = r'$d \ [ \mathrm{nm} ]$'


# %%


# force distanc
thermo_df_10_m_per_s, thermo_ave_df_10_m_per_s = constant_velocity_force_distance_curve_from_thermo_ave(
    thermo_out_file = 'sandbox/10_m_per_s/joint.thermo.out',
    thermo_ave_out_file = 'sandbox/10_m_per_s/joint.thermo_ave.out',
    initial_distance = 7.5, # nm
    total_steps = 375000,
    averaging_steps = 1000 )


# %%


query = {
    'identifier': { '$regex': '.*indenter_forces\.txt$'},
    'metadata.surfactant':     'SDS',
    'metadata.sf_nmolecules':  646,
    'metadata.sf_preassembly': 'hemicylinders',
    'metadata.indenter_dist':  7.5}


# %%


files = fp.get_file_by_query(query)


# %%


# correction for missing entry
files[1][1]['metadata']['total_steps'] = 3750000


# %%


# force distanc) in files
plotData = []
for (cont,doc) in files:
    contStream = io.StringIO(cont.decode())
    plotData.append(
        constant_velocity_force_distance_curve_from_force_file(
            force_file = contStream,
            initial_distance = doc['metadata']['indenter_dist'], # nm
            total_steps = doc['metadata']['total_steps'],
            averaging_steps = 1000 ) )


# %%


query = {
    'identifier': { '$regex': '.*indenter_forces\.txt$'},
    'metadata.surfactant':     'SDS',
    'metadata.sf_nmolecules':  646,
    'metadata.indenter_dist':  3}
files = fp.get_file_by_query(query)


# %%


len(files)


# %%


for 
files[0][1]['metadata']['total_steps'] = 15000000


# %%


# force distanc) in files
plotData = []
for (cont,doc) in files:
    contStream = io.StringIO(cont.decode())
    plotData.append(
        constant_velocity_force_distance_curve_from_force_file(
            force_file = contStream,
            initial_distance = doc['metadata']['indenter_dist'], # nm
            total_steps = doc['metadata']['total_steps'],
            averaging_steps = 1000 ) )


# %%


plotData[0]


# %%


query = {
    'identifier': { '$regex': '.*indenter_forces\.txt$'},
    'metadata.surfactant':     'SDS',
    'metadata.sf_nmolecules':  646}
files = fp.get_file_by_query(query)


# %%


len(files)


# %%


for i,f in enumerate(files):
    print('{:3d}:'.format(i), f[1]['identifier'],':', 'total_steps' in f[1]['metadata'])


# %%


# wrong step size in input files
files[0][1]['metadata']['total_steps'] = 15000000 # wrong step size
files[1][1]['metadata']['total_steps'] = 37500000
files[2][1]['metadata']['total_steps'] = 1500000 # wrong step size
files[3][1]['metadata']['total_steps'] = 375000
files[4][1]['metadata']['total_steps'] = 3750000
files[5][1]['metadata']['total_steps'] = 37500000
files[6][1]['metadata']['total_steps'] = 15000000 # wrong step size
files[7][1]['metadata']['total_steps'] = 37500000


# %%


for f in files:
    print(f[1]['identifier'],':', f[1]['metadata']['total_steps'])


# %%


# force distanc) in files
plotData = []
for (cont,doc) in files:
    contStream = io.StringIO(cont.decode())
    plotData.append(
        constant_velocity_force_distance_curve_from_force_file(
            force_file = contStream,
            initial_distance = doc['metadata']['indenter_dist'], # nm
            total_steps = doc['metadata']['total_steps'],
            averaging_steps = 1000 ) )


# %%


for i,f in enumerate(files):
    print('{:3d}:'.format(i), f[1]['identifier'],':', 'total_steps' in f[1]['metadata'])


# %%


files_hemicylinders = files[2:6]


# %%


files_1m_per_s = [ files[6], files[0], files[2] ] 


# %%


files_10cm_per_s = [ files[7], files[1], files[5] ] 


# %%


plotData_hemicylinders = [ plotData[5], plotData[4], plotData[2], plotData[3] ]


# %%


plotData_1m_per_s = [ plotData[6], plotData[0], plotData[2] ] 


# %%


plotData_10cm_per_s = [ plotData[7], plotData[1], plotData[5] ] 


# %%


labels =  [ 
    r'$1 \mathrm{\ m \ s^{-1}}$, bilayer',
    r'$10 \mathrm{\ m \ s^{-1}}$, hemicylinder',    
    r'$1 \mathrm{\ m \ s^{-1}}$, hemicylinder',
    r'$0.1 \mathrm{\ m \ s^{-1}}$, hemicylinder',
    r'$1 \mathrm{\ m \ s^{-1}}$, bilayer']   


# %%


labels_hemicylinders =  [ 
    r'$0.1 \mathrm{\ m \ s^{-1}}$',
    r'$1 \mathrm{\ m \ s^{-1}}$',
    r'$1 \mathrm{\ m \ s^{-1}}$',
    r'$10 \mathrm{\ m \ s^{-1}}$']


# %%


labels_1m_per_s = [ 
    'monolayer',    
    'bilayer',
    'hemicylinder' ]


# %%


plt.rcParams["figure.figsize"] = (8,6) # the standard figure size


# %%


plt.plot(
    force_conversion_factor*plotData_hemicylinders[0]["f_storeUnconstrainedForcesAve"][
            (0.0 < plotData_hemicylinders[0].index) & (plotData_hemicylinders[0].index < 7.0)].rolling(window=1,center=True).mean(),
    label = lab) 


# %%


figure = plt.figure()
windows = np.array([1,1,10,1]) # do average on the same scale, i.e. 2ps
for (dat,lab,win) in zip(plotData_hemicylinders,labels_hemicylinders,windows):
    plt.plot(
        force_conversion_factor*dat["f_storeUnconstrainedForcesAve"][
                (0.0 < dat.index) & (dat.index < 7.0)].rolling(window=win,center=True).mean(),
        label = lab, ls=':') 
plt.xlabel(distance_label)
plt.ylabel(force_label)
#plt.legend(frameon=False,ncol=2)
#plt.tight_layout(pad=1)
legend = plt.legend(frameon=False,loc='center right')
renderer = figure.canvas.get_renderer()
# get the width of your widest label, since every label will need 
# to shift by this amount after we align to the right
shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
for t in legend.get_texts():
    t.set_ha('right') # ha is alias for horizontalalignment
    t.set_position((shift,0))


# %%


# get the width of your widest label, since every label will need 
# to shift by this amount after we align to the right

shift = max([t.get_window_extent().width for t in legend.get_texts()])
for t in legend.get_texts():
    t.set_ha('right') # ha is alias for horizontalalignment
    t.set_position((shift,0))


# %%


legend.get_texts()[0].get_window_extent().width


# %%


# get the width of your widest label, since every label will need 
# to shift by this amount after we align to the right
shift = max([t.get_window_extent().width for t in legend.get_texts()])
for t in legend.get_texts():
    t.set_ha('right') # ha is alias for horizontalalignment
    t.set_position((shift,0))


# # 1m per s approach

# %%


windows = np.array([10,10,10,]) # do average on the same scale, i.e. 2ps
for (dat,lab,win) in zip(plotData_1m_per_s,labels_1m_per_s,windows):
    plt.plot(
        force_conversion_factor*dat["f_storeUnconstrainedForcesAve"].rolling(window=win,center=True).mean(),
        label = lab) 

plt.xlabel(distance_label)
plt.ylabel(force_label)
plt.legend()


# ## 10 cm per s approach

# %%


windows = np.array([10,10,10,]) # do average on the same scale, i.e. 2ps
for (dat,lab,win) in zip(plotData_10cm_per_s,labels_1m_per_s,windows):
    plt.plot(
        force_conversion_factor*dat["f_storeUnconstrainedForcesAve"].rolling(window=win,center=True).mean(),
        label = lab) 

plt.xlabel(distance_label)
plt.ylabel(force_label)
plt.legend()


# %%


test = None


# %%


type(test) is not list


# %%


test = [None]


# %%


test


# %%


type(test) is not list


# %%


windows = np.array([10,10,10]) # do average on the same scale, i.e. 2*2ps
for (dat,lab,win) in zip(plotData_1m_per_s,labels_1m_per_s,windows):
    plt.plot(
        force_conversion_factor*dat[
            "f_storeUnconstrainedForcesAve"][
                (0.0 < dat.index) & (dat.index < 4.0)].rolling(window=win,center=True).mean(),
        label = lab) 
plt.xlabel(distance_label)
plt.ylabel(force_label)
plt.legend()


# %%


windows = np.array([10,10,1,]) # do average on the same scale, i.e. 2*2ps
for (dat,lab,win) in zip(plotData_1m_per_s,labels_1m_per_s,windows):
    plt.plot(
        force_conversion_factor*dat[
            "f_storeUnconstrainedForcesAve"][
                (0.0 < dat.index) & (dat.index < 3.0)].rolling(window=win,center=True).mean(),
        label = lab) 
plt.xlabel(distance_label)
plt.ylabel(force_label)
plt.legend()


# %%


# force distanc
indenter_forces_10_m_per_s = constant_velocity_force_distance_curve_from_force_file(
    force_file = contStream[0],
    initial_distance = doc[0]['metadata']['indenter_dist'], # nm
    total_steps = doc[0]['metadata']['total_steps'],
    averaging_steps = 1000 )


# %%


# force distanc
indenter_forces_10_m_per_s = constant_velocity_force_distance_curve_from_force_file(
    force_file = './sandbox/10_m_per_s/indenter_z_forces.txt',
    initial_distance = 7.5, # nm
    total_steps = 375000,
    averaging_steps = 1000 )


# %%


# force distanc
indenter_forces_1_m_per_s = constant_velocity_force_distance_curve_from_force_file(
    force_file = './sandbox/1_m_per_s/indenter_z_forces.txt',
    initial_distance = 7.5, # nm
    total_steps = 3750000,
    averaging_steps = 10000 )


# %%


indenter_forces_1_m_per_s.head()


# %%


plt.plot(force_conversion_factor*indenter_forces_1_m_per_s["f_storeUnconstrainedForcesAve"],
        label = r'$1 \mathrm{\ m \ s^{-1}}$') 
plt.plot(force_conversion_factor*indenter_forces_10_m_per_s["f_storeUnconstrainedForcesAve"],
        label = r'$10 \mathrm{\ m \ s^{-1}}$')       
plt.xlabel(distance_label)
plt.ylabel(force_label)
plt.legend()


# %%


# force distanc
nonindenter_forces_10_m_per_s = constant_velocity_force_distance_curve_from_force_file(
    force_file = './sandbox/10_m_per_s/nonindenter_z_forces.txt',
    initial_distance = 7.5, # nm
    total_steps = 375000,
    averaging_steps = 1000 )


# %%


thermo_ave_df_1_m_per_s = [None]*2


# %%


thermo_df_1_m_per_s, thermo_ave_df_1_m_per_s = constant_velocity_force_distance_curve_from_thermo_ave(
   thermo_ave_out_file = 'sandbox/1_m_per_s/thermo_ave.out',
   thermo_out_file = 'sandbox/1_m_per_s/thermo.out',
   initial_distance = 7.5, # nm
   total_steps = 3750000,
   averaging_steps = 1000 )


# %%


_, thermo_ave_df_1_m_per_s[1] = constant_velocity_force_distance_curve(
    thermo_ave_out_file = 'sandbox/1_m_per_s/thermo_ave.out.3',
    initial_distance = 7.5, # nm
    total_steps = 3750000,
    averaging_steps = 1000 )


# %%


plt.plot(force_conversion_factor*thermo_ave_df_1_m_per_s[0]["indenter_non_indenter_interaction[3]"].loc[ thermo_ave_df_1_m_per_s[0].index < 7],
        label='instantaneous force')
plt.plot(force_conversion_factor*thermo_ave_df_1_m_per_s[1]["indenter_non_indenter_interaction[3]"],
        label = "{:.1f} ps average".format(averaging_time))    
   
plt.xlabel(distance_label)
plt.ylabel(force_label)
plt.legend()


# %%


plt.plot(force_conversion_factor*thermo_ave_df_1_m_per_s[0]["c_indenter_surfactant_interaction[3]"])
plt.plot(force_conversion_factor*thermo_ave_df_10_m_per_s["c_indenter_surfactant_interaction[3]"] )


# %%


plt.plot(force_conversion_factor*thermo_ave_df_1_m_per_s[0]["indenter_non_indenter_interaction[3]"].loc[ thermo_ave_df_1_m_per_s[0].index < 7 ],
        label = r'$1 \mathrm{\ m \ s^{-1}}$') 
plt.plot(force_conversion_factor*thermo_ave_df_1_m_per_s[1]["indenter_non_indenter_interaction[3]"].loc[ thermo_ave_df_1_m_per_s[1].index < 7 ])
plt.plot(force_conversion_factor*thermo_ave_df_10_m_per_s["indenter_non_indenter_interaction[3]"].loc[ (7 > thermo_ave_df_10_m_per_s.index) & (thermo_ave_df_10_m_per_s.index > 3) ],
                label = r'$10 \mathrm{\ m \ s^{-1}}$')    
   
plt.xlabel(distance_label)
plt.ylabel(force_label)
plt.legend()


# ### Drag force away from surface

# in nN

# #### 1 m / s

# %%


(force_conversion_factor*thermo_ave_df_1_m_per_s[0]["indenter_non_indenter_interaction[3]"].loc[ 
        (thermo_ave_df_1_m_per_s[0].index > 4) & (thermo_ave_df_1_m_per_s[0].index < 7) ]).mean()


# #### 10 m / s

# %%


(force_conversion_factor*thermo_ave_df_10_m_per_s["indenter_non_indenter_interaction[3]"].loc[
        (thermo_ave_df_10_m_per_s.index > 4) & (thermo_ave_df_10_m_per_s.index < 7) ]).mean()


# ### File system

# %%


lmplab_prefix = os.getcwd()


# %%


lmplab_prefix


# %%


sds_sys_subdir = os.sep.join(('sds','201810','sys'))


# %%


sds_sys_subdir


# %%


ctab_sys_subdir = os.sep.join(('ctab','201809','sys'))


# %%


ctab_sys_subdir


# %%


ctab_absolute_prefix = os.sep.join((lmplab_prefix,ctab_sys_subdir))


# %%


sds_absolute_prefix = os.sep.join((lmplab_prefix,sds_sys_subdir))


# %%


sds_absolute_prefix


# %%


indenter_system_suffix = '_50Ang_stepped'


# %%


ctab_system_glob_pattern = ''.join((
    ctab_absolute_prefix,os.sep,'*',indenter_system_suffix))


# %%


sds_system_glob_pattern = ''.join((
    sds_absolute_prefix,os.sep,'*',indenter_system_suffix))


# %%


ctab_system_glob_pattern = ''.join((
    ctab_absolute_prefix,os.sep,'*',indenter_system_suffix))


# %%


sds_system_glob_pattern


# %%


ctab_system_glob_pattern


# %%


# os.chdir(absolute_prefix)


# %%


sds_indenter_system_absolute_prefix_lst = sorted(glob(sds_system_glob_pattern))


# %%


ctab_indenter_system_absolute_prefix_lst = sorted(glob(ctab_system_glob_pattern))


# %%


sds_indenter_system_lst = [ 
    os.path.basename(
        indenter_system_absolute_prefix) for indenter_system_absolute_prefix
            in sds_indenter_system_absolute_prefix_lst ]


# %%


ctab_indenter_system_lst = [ 
    os.path.basename(
        indenter_system_absolute_prefix) for indenter_system_absolute_prefix
            in ctab_indenter_system_absolute_prefix_lst ]


# %%


ctab_indenter_system_lst


# %%


sds_indenter_system_lst


# %%


system_lst = [*sds_indenter_system_lst, *ctab_indenter_system_lst]


# %%


system_absolute_prefix_lst = [*sds_indenter_system_absolute_prefix_lst, 
                              *ctab_indenter_system_absolute_prefix_lst]


# %%


system_absolute_prefix_dict = dict(zip( system_lst, system_absolute_prefix_lst))


# %%


sorted(system_absolute_prefix_dict)


# %%


production_steps = [500000, 1000000]


# %%


production_steps


# %%


production_subdir_lst = [ 'production_{:d}'.format(steps) for steps in production_steps]


# %%


production_subdir_lst


# %%


production_dict = dict( zip(production_steps, production_subdir_lst) )


# %%


production_dict


# %%


system_names = system_lst


# %%


system_names


# %%


os.getcwd()


# %%


indenter_dict = {system: [] for system in system_names}


# ### Extract colvars tar files

# %%


# unpack all tar files
for system_name in system_names:
    system_absolute_prefix = system_absolute_prefix_dict[system_name]
    print("{:s}: {:s}".format(system_name, system_absolute_prefix))
    
    for steps, production_subdir in production_dict.items():
        production_absolute_prefix = os.sep.join((
            system_absolute_prefix, production_subdir))
        if os.path.isdir(production_absolute_prefix):
            print("  {:s}/{:s} exists.".format(system_name, production_subdir))
            tarfile_glob_pattern = os.sep.join((
                production_absolute_prefix,'*.tar.gz'))
            tarfiles = glob(tarfile_glob_pattern)
            if len(tarfiles) > 0:
                print("    {:s}/{:s} has {:s}.".format(
                    system_name, production_subdir, tarfiles[0]))
                colvars_tar = tarfile.open(tarfiles[0], 'r')
                print("    Content: {}".format(colvars_tar.getnames()))
                colvars_tar.extractall(path=production_absolute_prefix)
                print("    Extracted.")
                indenter_dict[system_name].append(steps)


# %%


indenter_dict


# %%


indenter_dict


# ## Batch

# %%


system_names = [ system_name for system_name, run_list                  in indenter_dict.items() if len(run_list) > 0 ]


# %%


set(system_names) & set(sds_indenter_system_lst)


# %%


set(system_names) & set(ctab_indenter_system_lst)


# ### Read data files

# %%


# read all systems
i = 0
#df_dict = {}
for system_name in system_names:
    system_absolute_prefix = system_absolute_prefix_dict[system_name]
    print("{:s}: {:s}".format(system_name, system_absolute_prefix))

    #df_dict[system_name] = {}
    if system_name not in df_dict:
        df_dict[system_name] = {}
        
    for total_steps in indenter_dict[system_name]:
        if exclude_dict and system_name in exclude_dict and total_steps in exclude_dict[system_name]:
            continue # skip system
            
        production_absolute_prefix = os.sep.join((
            system_absolute_prefix, production_dict[total_steps]))
        #os.chdir( production_dict[total_steps] )
        print("{:3d}: {:s}".format(i,production_absolute_prefix))
        
        df_dict[system_name][total_steps] = {}
        df_dict[system_name][total_steps]['colvars'] =             postprocessing.read_colvars_traj(
                prefix = production_absolute_prefix )
            
        ( df_dict[system_name][total_steps]['ti.pmf'],
          df_dict[system_name][total_steps]['ti.grad'],
          df_dict[system_name][total_steps]['ti.count'] ) = \
            postprocessing.read_colvars_ti( 
                prefix = production_absolute_prefix )
            
        df_dict[system_name][total_steps]['thermo'] =             postprocessing.read_production_thermo(
                prefix = production_absolute_prefix )

        i+=1


# ### Read forces from netcdf

# %%


force_keys = [
    'forces', 
    'f_storeAnteSHAKEForces', 
    'f_storeAnteStatForces', 
    'f_storeUnconstrainedForces', 
    'f_storeAnteSHAKEForcesAve', 
    'f_storeAnteStatForcesAve', 
    'f_storeUnconstrainedForcesAve' ]


# %%


t2n_array_dict = { system_name: postprocessing.sds_t2n_array for system_name
                      in sds_indenter_system_lst }


# %%


t2n_array_dict.update( 
    { system_name: postprocessing.ctab_t2n_array for system_name
                      in ctab_indenter_system_lst } )


# %%


# 
exclude_dict = {'1010_CTAB_on_AU_111_63x36x2_bilayer_with_counterion_50Ang_stepped': [1000000],
                '1010_CTAB_on_AU_111_63x36x2_monolayer_with_counterion_50Ang_stepped': [1000000],
                '653_CTAB_on_AU_111_63x36x2_monolayer_with_counterion_50Ang_stepped': [1000000],
                '653_CTAB_on_AU_111_63x36x2_hemicylinders_with_counterion_50Ang_stepped': [1000000],}
               #'653_CTAB_on_AU_111_63x36x2_bilayer_with_counterion_50Ang_stepped':[1000000]} # done, but not fully concatenated yet
# 1010_CTAB_on_AU_111_63x36x2_bilayer_with_counterion_50Ang_stepped/production_1000000


# %%


indenter_dict['653_CTAB_on_AU_111_63x36x2_bilayer_with_counterion_50Ang_stepped']


# %%


constant_offset = 7.0637 # Ang, half thickness of substrate


# %%


# RUN AGAIN
# read all systems
i = 0
# update df_dict
for system_name in system_names:
    system_absolute_prefix = system_absolute_prefix_dict[system_name]
    print("{:s}: {:s}".format(system_name, system_absolute_prefix))

    if system_name not in df_dict: # this should never be the case if data files have been read precedingly
        df_dict[system_name] = {}
        
    t2n_array = t2n_array_dict[system_name]
    
    for total_steps in indenter_dict[system_name]:
        if exclude_dict and system_name in exclude_dict and total_steps in exclude_dict[system_name]:
            continue # skip system
            
        production_absolute_prefix = os.sep.join((
            system_absolute_prefix, production_dict[total_steps]))
        #os.chdir( production_dict[total_steps] )
        print("{:3d}: {:s}".format(i,production_absolute_prefix))
        
        # check whether has been processed before:
        system_prefix = system_name + '_' + production_dict[total_steps]
        indenter_z_forces_json_name = system_prefix + '_indenter_z_forces.json'
        nonindenter_z_forces_json_name = system_prefix + '_nonindenter_z_forces.json'
        indenter_z_forces_json_absolute_file_name = os.sep.join((
            production_absolute_prefix,indenter_z_forces_json_name))
        nonindenter_z_forces_json_absolute_file_name = os.sep.join((
            production_absolute_prefix,nonindenter_z_forces_json_name))
    
        
        netcdf_glob_pattern = os.sep.join((
            production_absolute_prefix, '*.nc'))

        netcdf = glob(netcdf_glob_pattern)[0]

        tmp_traj = NetCDFTrajectory(netcdf, 'r', 
            types_to_numbers = list( t2n_array ),
            keep_open=True )
        
        # use first frame to crudely identify indenter
        solid_selection = (
            tmp_traj[0].get_atomic_numbers() == ase.data.atomic_numbers['Au'])
        indenter_selection = (
            solid_selection & (tmp_traj[0].get_positions()[:,2] > 20 ) ) 
        # 20 some arbitrary xy-plane seperating substrate & indenter
        
        print("solid: {: 9d} atoms, therof indenter: {: 9d} atoms.".format(
            np.count_nonzero(solid_selection),
            np.count_nonzero(indenter_selection) ) )
        
        # derive z dist offset
        # (COM-COM distance is stored, we want surface-apex distance)
        #pmf_df = df_dict[system_name][total_steps]['ti.pmf']
        colvars_traj_df = df_dict[system_name][total_steps]['colvars']
        #thermo_df = df_dict[system_name][total_steps]['thermo']
        
        # initial apex position at frame 0:
        apex_z0 = tmp_traj[0][indenter_selection].get_positions()[:,2].min()

        # initial substrate surface position at frame 0:
        surface_z0 = tmp_traj[0][solid_selection # in order to have a standardized width
            & ~indenter_selection].get_positions()[:,2].mean() + constant_offset

        extents_dist_0 = apex_z0 - surface_z0
        com_com_dist_0 = colvars_traj_df.loc[0,'com_com_dist_z']
        dist_offset = com_com_dist_0 - extents_dist_0
        
        print("initial com-com dist:        {:8.3f}".format(com_com_dist_0) )
        print("initial apex-surface dist:   {:8.3f}".format(extents_dist_0) )
        print("resulting positional offset: {:8.3f}".format(dist_offset) )
        
        
        # sum forces on all indenter atoms for all frames
        if os.path.isfile(indenter_z_forces_json_absolute_file_name) and             os.path.isfile(nonindenter_z_forces_json_absolute_file_name):
            print("Has been processed before, reading json files {:s} and {:s}.".format(
                indenter_z_forces_json_absolute_file_name,
                nonindenter_z_forces_json_absolute_file_name))
            indenter_force_z_sum_df = pd.read_json(
                indenter_z_forces_json_absolute_file_name, orient='index')
            nonindenter_force_z_sum_df = pd.read_json(
                nonindenter_z_forces_json_absolute_file_name, orient='index')
        else:
            print("Reading NetCDF.")
            indenter_force_sum_dict = { key: [] for key in force_keys }
            nonindenter_force_sum_dict = { key: [] for key in force_keys }
            for key in force_keys:    
                if key in tmp_traj[0].arrays:
                    indenter_force_sum_dict[key] = np.array(
                        [ f[indenter_selection].arrays[key].sum(axis=0) 
                             for f in tmp_traj ] )
                    nonindenter_force_sum_dict[key] = np.array(
                        [ f[~indenter_selection].arrays[key].sum(axis=0) 
                             for f in tmp_traj ] )
                else:
                    print("Warning: key '{:s}' not in NetCDF".format(key))

            # only keep z forces and create data frames
            indenter_force_z_sum_dict = { key: value[:,2] for key, value 
                            in indenter_force_sum_dict.items() }

            nonindenter_force_z_sum_dict = { key: value[:,2] for key, value 
                                    in nonindenter_force_sum_dict.items() }

            indenter_force_z_sum_df = pd.DataFrame.from_dict(
                indenter_force_z_sum_dict, dtype=float)

            nonindenter_force_z_sum_df = pd.DataFrame.from_dict(
                nonindenter_force_z_sum_dict, dtype=float)

            # make indices agree
            netcdf_output_interval = colvars_traj_df.index[-1]/(len(tmp_traj)-1)
            print("netcdf stores every {:d}th of {:d} frames in total.".format(
                int(netcdf_output_interval), int(colvars_traj_df.index[-1]) ) )

            indenter_force_z_sum_df.set_index(
                (indenter_force_z_sum_df.index*netcdf_output_interval).astype(int),
                inplace=True )
            nonindenter_force_z_sum_df.set_index(
                (nonindenter_force_z_sum_df.index*netcdf_output_interval).astype(int),
                inplace=True )
        
            # store z forces in json files
            indenter_force_z_sum_df.to_json(
                indenter_z_forces_json_absolute_file_name,  orient='index')

            nonindenter_force_z_sum_df.to_json(
                nonindenter_z_forces_json_absolute_file_name,  orient='index')

        # keep forces in data frame dict
        if total_steps not in df_dict[system_name]: 
            # this should never be the case if data files have been read precedingly
            df_dict[system_name][total_steps] = {}
        
        df_dict[system_name][total_steps]['indenter_forces_z'] =             indenter_force_z_sum_df
            
        df_dict[system_name][total_steps]['nonindenter_forces_z'] =             nonindenter_force_z_sum_df
            
        df_dict[system_name][total_steps]['dist_offset'] = dist_offset

        i+=1


# %%


tmp_traj[1]


# %%


df_dict['1107_SDS_on_AU_111_51x30x2_bilayer_with_counterion_50Ang_stepped'][500000].keys()


# ### Make plots

# %%


system_names


# %%


import re


# %%


bilayer_regex = re.compile('.*bilayer.*')


# %%


monolayer_regex = re.compile('.*monolayer.*')


# %%


cylinders_regex = re.compile('(?!hemi)*cylinders.*')


# %%


hemicylinders_regex = re.compile('.*hemicylinders.*')


# %%


sds_regex = re.compile('.*SDS.*')


# %%


ctab_regex = re.compile('.*CTAB.*')


# %%


ctab_653_regex = re.compile('653_CTAB.*')


# %%


sds_646_regex = re.compile('646_SDS.*')


# %%


sds_377_regex = re.compile('377_SDS.*')


# %%


sds_220_regex = re.compile('220_SDS.*')


# %%


sds_75_regex = re.compile('75_SDS.*')


# %%


ctab_bilayer_systems = list(
    set( filter(bilayer_regex.match, system_names) ) \
    & set( filter( ctab_regex.match, system_names ) ) )


# %%


ctab_monolayer_systems = list(
    set( filter(monolayer_regex.match, system_names) ) \
    & set( filter( ctab_regex.match, system_names ) ) )


# %%


ctab_hemicylindrical_systems = list( # attention cylinders & hemicylinders swicthed accidentially
    set( filter(cylinders_regex.match, system_names) ) \
    & set( filter( ctab_regex.match, system_names ) ) )


# %%


ctab_cylindrical_systems = list( # attention cylinders & hemicylinders swicthed accidentially
    set( filter(hemicylinders_regex.match, system_names) ) \
    & set( filter( ctab_regex.match, system_names ) ) )


# %%


ctab_653_systems = list( set( filter( ctab_653_regex.match, system_names ) ) )


# %%


ctab_monolayer_systems


# %%


ctab_bilayer_systems


# %%


ctab_hemicylindrical_systems


# %%


ctab_cylindrical_systems


# %%


ctab_653_systems


# %%


system_names


# %%


sds_bilayer_systems = list(
    set( filter(bilayer_regex.match, system_names) ) \
    & set( filter( sds_regex.match, system_names ) ) )


# %%


sds_monolayer_systems = list(
    set( filter(monolayer_regex.match, system_names) ) \
    & set( filter( sds_regex.match, system_names ) ) )


# %%


sds_hemicylindrical_systems = list(
    set( filter(hemicylinders_regex.match, system_names) ) \
    & set( filter( sds_regex.match, system_names ) ) )


# %%


sds_cylindrical_systems = list( 
    set( filter(cylinders_regex.match, system_names) ) \
    & set( filter( sds_regex.match, system_names ) ) )


# %%


sds_646_systems = list( set( filter( sds_646_regex.match, system_names ) ) )


# %%


sds_377_systems = list( set( filter( sds_377_regex.match, system_names ) ) )


# %%


sds_220_systems = list( set( filter( sds_220_regex.match, system_names ) ) )


# %%


sds_75_systems = list( set( filter( sds_75_regex.match, system_names ) ) )


# %%


system_names


# %%


sds_monolayer_systems


# %%


sds_bilayer_systems


# %%


shape_labels = ['monolayer','bilayer', 'cylinders']


# %%


shape_labels


# %%


shape_label_dict = dict(zip(system_names, shape_labels))


# %%


def shape_label_assigner(system_name):
    if monolayer_regex.match(system_name):
        return 'monolayer'
    elif bilayer_regex.match(system_name):
        return 'bilayer'
    elif hemicylinders_regex.match(system_name) and sds_regex.match(system_name):
        return 'hemicylinder'
    elif cylinders_regex.match(system_name) and ctab_regex.match(system_name):
        return 'hemicylinder'
    else:
        return 'cylinder'


# %%


shape_label_dict = {
    system_name: shape_label_assigner(system_name) for system_name in system_names
}


# %%


shape_label_dict


# %%


np.array(production_steps) * postprocessing.fs


# %%


distance_covered = 100 # rough estimate


# %%


rate_labels = distance_covered * postprocessing.AA / ( 
    2* np.array(production_steps) * postprocessing.fs ) # labels in m/s or nm/ns


# %%


rate_labels


# %%


production_dict


# %%


rate_labels_str = [ 
    r'$' + '{:>2d}'.format(int(vel)) + r' \mathrm{m} \mathrm{s}^{-1}$' \
                   for vel in rate_labels ]


# %%


rate_labels_str


# %%


rate_label_dict = dict(zip(production_steps,rate_labels_str))


# %%


data_tags = ['thermo', 'colvars', 'ti.pmf', 'ti.grad', 'ti.count']
fig_tags =  ['thermo', 'colvars', 'groupgroup', 'pmf' ]


# %%


system_selection = ctab_653_systems


# %%


dist_interval = slice(20,80)


# %%


df_dict.keys()


# %%


df_dict['377_SDS_on_AU_111_51x30x2_bilayer_with_counterion_50Ang_stepped'].keys()


# #### Thermo- & group interaction plots

# %%


system_lst


# %%


indenter_system_suffix = '_50Ang_stepped'


# %%


original_system_lst = [ s[:-len(indenter_system_suffix)] for s in system_lst ]


# %%


molecules_per_area = 1e-18/sim_df.loc[original_system_lst,'sb_area_per_sf_molecule'] # in nm**2


# %%


shape_label_dict = dict( zip( system_lst, [ '${: 4.2f} \mathrm{{nm}}^{{-2}}$'.format(gamma) for gamma in molecules_per_area]))


# %%


n_surfactant_regexp = re.compile('^([0-9]+)_.*')


# %%


n_surfactant = [ int(n_surfactant_regexp.match(s).group(1)) for s in system_lst ]


# %%


system_selection = ctab_hemicylindrical_systems


# %%


system_selection = sorted(system_selection,key=lambda s: dict(zip(system_lst,n_surfactant))[s])


# %%


system_selection = sorted(system_selection)


# %%


system_selection


# %%


shape_label_dict = dict( zip( system_selection, ['bilayer', 'monolayer', 'cylinders']))


# %%


shape_label_dict = dict( zip( system_selection, ['bilayer', 'hemicylinders', 'monolayer']))


# %%


shape_label_dict = dict( zip( system_selection, ['bilayer','monolayer']))


# %%


window = 3


# %%


# plot all
i = 0
rate_fig_dict = {} # same shape, different rates
shape_fig_dict = {} # same rate, different shapes

# plots_of_interest = ''
# for system_name, steps_df_dict in df_dict.items():
for system_name in system_selection:
    steps_df_dict = df_dict[system_name]          
    for total_steps, data_df_dict in steps_df_dict.items():
        if system_name in exclude_dict and total_steps in exclude_dict[system_name]:
            continue
            
        print("{:<72s} {:>12d}".format(system_name, total_steps))
        
        if total_steps not in shape_fig_dict:
            shape_fig_dict[total_steps] = {}
            for tag in fig_tags:
                shape_fig_dict[total_steps][tag] = {}
                shape_fig_dict[total_steps][tag]['fig']  = None
                shape_fig_dict[total_steps][tag]['axes'] = None
    
        if system_name not in rate_fig_dict:
            rate_fig_dict[system_name] = {}
            for tag in fig_tags:
                rate_fig_dict[system_name][tag] = {}
                rate_fig_dict[system_name][tag]['fig']  = None
                rate_fig_dict[system_name][tag]['axes'] = None
        #for df_name, df in data_df_dict.items():
        #    if df_name 
        #    print(df_name)
        ( shape_fig_dict[total_steps]['groupgroup']['fig'],
          shape_fig_dict[total_steps]['groupgroup']['axes'] ) = \
            postprocessing.makeGroupGroupInteractionsByDistPlot( 
                data_df_dict['thermo'], 
                data_df_dict['colvars'], 
                data_df_dict['ti.pmf'],
                fig  = shape_fig_dict[total_steps]['groupgroup']['fig'],
                axes = shape_fig_dict[total_steps]['groupgroup']['axes'],
                legend_prefix = shape_label_dict[system_name], 
                interval = dist_interval, window = window,
                x_offset = - data_df_dict['dist_offset'],
                force_factor = force_conversion_factor,
                distance_label=r'Distance $ d \ \left( \mathrm{\AA} \right)$',
                force_label=r'Normal force $ F \ \left( \mathrm{nN} \right)$')

        ( rate_fig_dict[system_name]['groupgroup']['fig'],
          rate_fig_dict[system_name]['groupgroup']['axes'] ) = \
            postprocessing.makeGroupGroupInteractionsByDistPlot( 
                data_df_dict['thermo'], 
                data_df_dict['colvars'], 
                data_df_dict['ti.pmf'],
                fig  = rate_fig_dict[system_name]['groupgroup']['fig'],
                axes = rate_fig_dict[system_name]['groupgroup']['axes'],
                legend_prefix = rate_label_dict[total_steps], 
                interval = dist_interval, window = window,
                x_offset = - data_df_dict['dist_offset'],
                force_factor = force_conversion_factor,
                distance_label=r'Distance $ d \ \left( \mathrm{\AA} \right)$',
                force_label=r'Normal force $ F \ \left( \mathrm{nN} \right)$')
        i+=1
    
print("Finished after {:d} loops".format(i))


# %%





# %%


shape_fig_dict[500000]['groupgroup']['fig']


# %%


rate_fig_dict.keys()


# %%





# %%


system_selection


# %%


rate_fig_dict['75_SDS_on_AU_111_51x30x2_monolayer_with_counterion_50Ang_stepped']['groupgroup']['fig']


# %%


for ax in shape_fig_dict[500000]['groupgroup']['axes']:
    ax.legend().set_visible(False)


# %%





# %%


rate_fig_dict['653_CTAB_on_AU_111_63x36x2_hemicylinders_with_counterion_50Ang_stepped']['groupgroup']['fig']


# %%


shape_fig_dict[500000]['groupgroup']['fig']


# %%


fig_tags


# %%


fig_tags = ['indenter_forces_z', 'nonindenter_forces_z']


# %%


dist_interval = slice(25,80)


# #### Netcdf force plots

# %%


data_df_dict.keys()


# %%


# plot all
i = 0
rate_fig_dict = {} # same shape, different rates
shape_fig_dict = {} # same rate, different shapes

# plots_of_interest = ''
for system_name, steps_df_dict in df_dict.items():
    # rate_fig_dict[system_name] = {}
    # shape_fig_dict[system_name] = {}
    
    if system_name not in system_selection:
        continue
          
    for total_steps, data_df_dict in steps_df_dict.items():
        if system_name in exclude_dict and total_steps in exclude_dict[system_name]:
            continue
            
        print("{:<72s} {:>12d}".format(system_name, total_steps))
        
        if total_steps not in shape_fig_dict:
            shape_fig_dict[total_steps] = {}
            for tag in fig_tags:
                shape_fig_dict[total_steps][tag] = {}
                shape_fig_dict[total_steps][tag]['fig']  = None
                shape_fig_dict[total_steps][tag]['axes'] = None
    
#         if system_name not in rate_fig_dict:
#             rate_fig_dict[system_name] = {}
#             for tag in fig_tags:
#                 rate_fig_dict[system_name][tag] = {}
#                 rate_fig_dict[system_name][tag]['fig']  = None
#                 rate_fig_dict[system_name][tag]['axes'] = None
        
        
        ( shape_fig_dict[total_steps]['indenter_forces_z']['fig'],
          shape_fig_dict[total_steps]['indenter_forces_z']['axes'] ) = \
            postprocessing.makeVariableByDistPlot( 
                data_df_dict['indenter_forces_z'][["f_storeUnconstrainedForcesAve"]]*force_conversion_factor,
                data_df_dict['colvars'], 
                data_df_dict['ti.pmf'],
                fig  = shape_fig_dict[total_steps]['indenter_forces_z']['fig'],
                axes = shape_fig_dict[total_steps]['indenter_forces_z']['axes'],
                #legend_prefix = shape_label_dict[system_name], 
                legend = shape_label_dict[system_name],
                interval = dist_interval, window = 5,
                x_offset = data_df_dict['dist_offset'],
                x_label=r'Distance $ d \ \left( \mathrm{\AA} \right)$',
                y_label=r'Normal force $ F \ \left( \mathrm{nN} \right)$')

#         ( shape_fig_dict[total_steps]['nonindenter_forces_z']['fig'],
#           shape_fig_dict[total_steps]['nonindenter_forces_z']['axes'] ) = \
#             postprocessing.makeVariableByDistPlot( 
#                 data_df_dict['nonindenter_forces_z'][["f_storeUnconstrainedForcesAve"]], 
#                 data_df_dict['colvars'], 
#                 data_df_dict['ti.pmf'],
#                 fig  = shape_fig_dict[total_steps]['nonindenter_forces_z']['fig'],
#                 axes = shape_fig_dict[total_steps]['nonindenter_forces_z']['axes'],
#                 #legend_prefix = shape_label_dict[system_name], 
#                 legend = shape_label_dict[system_name],
#                 interval = dist_interval, window = 5,
#                 x_offset = data_df_dict['dist_offset'],
#                 x_label=r'Distance $ d \ \left( \mathrm{\AA} \right)$',
#                 y_label=r'Normal force $ F \ \left( \mathrm{nN} \ \mathrm{mol}^{-1} \right)$')
            
        #for tag in fig_tags:
        #    shape_fig_dict[total_steps][tag]['fig'].legend()
        i+=1
    
print("Finished after {:d} loops".format(i))


# %%


label_dict = dict( zip( system_lst, system_lst))


# %%


ctab_653_systems


# %%


system_selection = ['653_CTAB_on_AU_111_63x36x2_bilayer_with_counterion_50Ang_stepped']


# %%


sds_cylindrical_systems


# %%


nsds_lst = [646,75,129,377,220]


# %%


system_selection = sds_monolayer_systems


# %%


system_selection


# %%


system_selection


# %%


system_selection = sorted(system_selection,key=lambda s: dict(zip(sds_bilayer_systems,nsds_lst))[s])


# %%


sys


# %%


rate_label_dict


# %%


window = 1


# %%


dist_interval = slice(25,80)


# %%


system_selection = sds_377_systems


# %%


# plot all
i = 0
rate_fig_dict = {} # same shape, different rates
shape_fig_dict = {} # same rate, different shapes

# plots_of_interest = ''
# for system_name, steps_df_dict in df_dict.items():
for system_name in system_selection:
    steps_df_dict = df_dict[system_name]
    # rate_fig_dict[system_name] = {}
    # shape_fig_dict[system_name] = {}
    
    if system_name not in system_selection:
        continue
          
    for total_steps, data_df_dict in steps_df_dict.items():
        if system_name in exclude_dict and total_steps in exclude_dict[system_name]:
            continue
            
        print("{:<72s} {:>12d}".format(system_name, total_steps))
        
        if total_steps not in shape_fig_dict:
            shape_fig_dict[total_steps] = {}
            for tag in fig_tags:
                shape_fig_dict[total_steps][tag] = {}
                shape_fig_dict[total_steps][tag]['fig']  = None
                shape_fig_dict[total_steps][tag]['axes'] = None
    
        if system_name not in rate_fig_dict:
            rate_fig_dict[system_name] = {}
            for tag in fig_tags:
                rate_fig_dict[system_name][tag] = {}
                rate_fig_dict[system_name][tag]['fig']  = None
                rate_fig_dict[system_name][tag]['axes'] = None
        
        
        ( shape_fig_dict[total_steps]['indenter_forces_z']['fig'],
          shape_fig_dict[total_steps]['indenter_forces_z']['axes'] ) = \
            postprocessing.makeVariableByDistPlot( 
                data_df_dict['indenter_forces_z'][["f_storeUnconstrainedForcesAve"]]*force_conversion_factor,
                data_df_dict['colvars'], 
                data_df_dict['ti.pmf'],
                fig  = shape_fig_dict[total_steps]['indenter_forces_z']['fig'],
                axes = shape_fig_dict[total_steps]['indenter_forces_z']['axes'],
                #legend_prefix = shape_label_dict[system_name], 
                legend = label_dict[system_name],
                interval = dist_interval, window = window,
                x_offset = data_df_dict['dist_offset'],
                x_label=r'Distance $ d \ \left( \mathrm{\AA} \right)$',
                y_label=r'Normal force $ F \ \left( \mathrm{nN} \right)$')

#         ( shape_fig_dict[total_steps]['nonindenter_forces_z']['fig'],
#           shape_fig_dict[total_steps]['nonindenter_forces_z']['axes'] ) = \
#             postprocessing.makeVariableByDistPlot( 
#                 data_df_dict['nonindenter_forces_z'][["f_storeUnconstrainedForcesAve"]], 
#                 data_df_dict['colvars'], 
#                 data_df_dict['ti.pmf'],
#                 fig  = shape_fig_dict[total_steps]['nonindenter_forces_z']['fig'],
#                 axes = shape_fig_dict[total_steps]['nonindenter_forces_z']['axes'],
#                 #legend_prefix = shape_label_dict[system_name], 
#                 legend = shape_label_dict[system_name],
#                 interval = dist_interval, window = 5,
#                 x_offset = data_df_dict['dist_offset'],
#                 x_label=r'Distance $ d \ \left( \mathrm{\AA} \right)$',
#                 y_label=r'Normal force $ F \ \left( \mathrm{nN} \ \mathrm{mol}^{-1} \right)$')

        ( rate_fig_dict[system_name]['indenter_forces_z']['fig'],
          rate_fig_dict[system_name]['indenter_forces_z']['axes'] ) = \
            postprocessing.makeVariableByDistPlot( 
                data_df_dict['indenter_forces_z'][["f_storeUnconstrainedForcesAve"]]*force_conversion_factor,
                data_df_dict['colvars'], 
                data_df_dict['ti.pmf'],
                fig  = rate_fig_dict[system_name]['indenter_forces_z']['fig'],
                axes = rate_fig_dict[system_name]['indenter_forces_z']['axes'],
                #legend_prefix = shape_label_dict[system_name], 
                legend = rate_label_dict[total_steps], #label_dict[system_name],
                interval = dist_interval, window = window,
                x_offset = data_df_dict['dist_offset'],
                x_label=r'Distance $ d \ \left( \mathrm{\AA} \right)$',
                y_label=r'Normal force $ F \ \left( \mathrm{nN} \right)$')
            
        #for tag in fig_tags:
        #    shape_fig_dict[total_steps][tag]['fig'].legend()
        i+=1
    
print("Finished after {:d} loops".format(i))


# %%


ctab_653_systems


# %%


df_dict[ctab_653_systems[0]][500000]['dist_offset']


# %%


df_dict[ctab_653_systems[1]][500000]['dist_offset']


# %%


df_dict[ctab_653_systems[2]][500000]['dist_offset']


# %%


ave_offset = np.sum( [df_dict[ctab_653_systems[i]][500000]['dist_offset'] for i in range(3) ] ) / 3


# %%


ave_offset


# %%


nonindenter_force_z_sum_df.index = indenter_force_z_sum_df.index


# %%





# %%


postprocessing.makeVariableByDistPlot(
    nonindenter_force_z_sum_df[['f_storeUnconstrainedForcesAve']]*force_conversion_factor, 
    colvars_traj_df, pmf_df,
    x_offset = dist_offset,
    interval=d_interval, window=1,
    x_label=r'Distance $ d \ \left( \mathrm{\AA} \right)$',
    y_label=r'Normal force $ F \ \left( \mathrm{nN} \mathrm{mol}^{-1} \right)$')


# %%


data_df_dict.keys()


# %%


data_df_dict.keys()


# ## Colvars

# %%


df_dict['646_SDS_on_AU_111_51x30x2_bilayer_with_counterion_50Ang_stepped'][500000].keys()


# %%


colvars_traj_df = df_dict['646_SDS_on_AU_111_51x30x2_bilayer_with_counterion_50Ang_stepped'][500000]['colvars']


# %%


pmf_df = df_dict['646_SDS_on_AU_111_51x30x2_bilayer_with_counterion_50Ang_stepped'][500000]['ti.pmf']


# %%


grad_df = df_dict['646_SDS_on_AU_111_51x30x2_bilayer_with_counterion_50Ang_stepped'][500000]['ti.grad']


# %%


count_df = df_dict['646_SDS_on_AU_111_51x30x2_bilayer_with_counterion_50Ang_stepped'][500000]['ti.count']


# %%


fig, axes = postprocessing.makeColvarsPlots(
    colvars_traj_df, #interval=slice(5000,None), 
    legend_prefix = 'raw');


# %%


fig, axes = postprocessing.makeColvarsPlots(
    colvars_traj_df, #interval=slice(5000,None), 
    window = 100,
    fig = fig, axes = axes, legend_prefix = 'ave' );


# %%


fig


# %%


dist_interval=slice(20,60)


# %%


fig, axes = postprocessing.makeColvarsPlotsByDist(
    colvars_traj_df, pmf_df, legend_prefix = 'raw',
    interval=dist_interval);


# %%


fig, axes = postprocessing.makeColvarsPlotsByDist(
    colvars_traj_df, pmf_df, legend_prefix = 'ave', window = 5,
    fig = fig, axes = axes, interval=dist_interval );


# %%


fig


# %%


fig, axes = postprocessing.makePMEPlots(
    pmf_df, grad_df, count_df, 
    interval = dist_interval,
    legend_prefix ='raw');


# %%


fig, axes = postprocessing.makePMEPlots(
    pmf_df, grad_df, count_df, 
    interval = slice(40,100),
    legend_prefix ='ave', window = 5,
    fig = fig, axes = axes);


# %%


fig


# ## Thermo output

# %%


postprocessing.read_production_thermo()


# %%


thermo_df = postprocessing.evaluate_production();


# %%


fig, axes = postprocessing.makeThermoPlotsFromDataFrame(
    thermo_df, legend_prefix = 'raw');


# %%


fig, axes = postprocessing.makeThermoPlotsFromDataFrame(
    thermo_df, fig = fig, axes = axes, window=50, 
    legend_prefix='ave');


# %%


fig


# ## Group - Group interactions

# %%


fig, axes = postprocessing.makeGroupGroupInteractionsPlot(thermo_df,legend_prefix='raw')


# %%


fig, axes = postprocessing.makeGroupGroupInteractionsPlot(
    thermo_df,legend_prefix='ave', window=50,
    fig = fig, axes = axes)


# %%


fig


# %%


fig, axes = postprocessing.makeGroupGroupInteractionsByDistPlot(
    thermo_df, colvars_traj_df, pmf_df, legend_prefix = 'raw');


# %%


fig, axes = postprocessing.makeGroupGroupInteractionsByDistPlot(
    thermo_df, colvars_traj_df, pmf_df, legend_prefix = 'ave', window=5,
    fig = fig, axes = axes );


# %%


fig


# %%


thermo_df = postprocessing.evaluate_production();


# %%


postprocessing.evaluate_group_group_interactions(thermo_df);


# %%


for dirname, dirnames, filenames in os.walk('.'):
    # print path to all subdirectories first.
    for subdirname in dirnames:
        print(os.path.join(dirname, subdirname))

    # print path to all filenames.
    for filename in filenames:
        print(os.path.join(dirname, filename))

    # Advanced usage:
    # editing the 'dirnames' list will stop os.walk() from recursing into there.
    if '.git' in dirnames:
        # don't go into any .git directories.
        dirnames.remove('.git')


# %%


get_ipython().system('cat  653_CTAB_on_AU_111_63x36x2_hemicylinders_with_counterion_50Ang_stepped.indenter_pulled.ti.pmf')


# ## netcdf evaluation

# ### Forces
# 
# $ [F_{LMP}] = \frac{ \mathrm{kcal}}{ \mathrm{mol} \cdot \mathrm{\mathring{A}}} $
# 
# $ [F_{PLT}] = \mathrm{nm}$
# 
# $ \mathrm{kcal} = 4.184 \mathrm{kJ} = 4.184 \cdot 10^{3} \mathrm{J}$
# 
# $ J = N \cdot m$
# 
# $ N = J m^{-1} = \frac{ \mathrm{kcal}}{ 4.184 \cdot 10^3 \mathrm{m} }
#     = \frac{ \mathrm{kcal}}{ 4.184 \cdot 10^{13} \mathrm{\mathring{A}} }$
#     
# $ [F_{PLT}] = \mathrm{nN} 
#     = \frac{ 10^9 \cdot N_A^{-1}}{ 4.184 \cdot 10^{13} } 
#         \frac{\mathrm{kcal}}{\mathrm{mol} \cdot \mathrm{\mathring{A}}}
#     = \frac{ 10^{-4} \cdot N_A^{-1}}{ 4.184 } [F_{LMP}]
#     = 0.239 \cdot 10^{-4} N_A^{-1} [F_{LMP}] $

# %%


system_names


# %%


system_name


# %%


production_absolute_prefix = os.sep.join((
    system_absolute_prefix_dict[system_name],
    production_dict[500000]))


# %%


production_absolute_prefix


# %%


netcdf_glob_pattern = os.sep.join((
    production_absolute_prefix, '*.nc'))


# %%


netcdf_glob_pattern


# %%


netcdf = glob(netcdf_glob_pattern)[0]


# %%


netcdf


# %%


postprocessing.sds_t2n_array


# %%


tmp_traj = NetCDFTrajectory(
    netcdf, 'r', 
    types_to_numbers = list( postprocessing.sds_t2n_array ),
                        keep_open=True )


# %%


len(tmp_traj)


# %%


f1 = tmp_traj[0]


# %%


f2 = tmp_traj[-1]


# ### select groups

# %%


solid_selection = (
    f.get_atomic_numbers() == ase.data.atomic_numbers['Au'])


# %%


indenter_selection = (
    solid_selection & (f.get_positions()[:,2] > 20) )


# %%


np.count_nonzero(indenter_selection)


# %%


tmp_traj[0][ indenter_selection ]


# %%


tmp_traj[-1][ indenter_selection]


# %%


len(tmp_traj[0][ solid_selection ]) # 21901


# %%


len(tmp_traj[0][ indenter_selection ]) # 3541


# %%


tmp_traj[0][ indenter_selection ].get_positions().max(axis=0)


# %%


for key in force_keys:
    force_sum_dict[key] = f_indenter.arrays[key].sum(axis=0)


# %%


force_keys = [
    'forces', 
    'f_storeAnteSHAKEForces', 
    'f_storeAnteStatForces', 
    'f_storeUnconstrainedForces', 
    'f_storeAnteSHAKEForcesAve', 
    'f_storeAnteStatForcesAve', 
    'f_storeUnconstrainedForcesAve' ]


# %%


force_sum_dict = { key: [] for key in force_keys }


# %%


df = pd.DataFrame()


# %%


tmp_traj[100][indenter_selection].arrays.keys()


# %%


# sum forces on all indenter atoms for all frames
indenter_force_sum_dict = { key: [] for key in force_keys }
nonindenter_force_sum_dict = { key: [] for key in force_keys }
for key in force_keys:    
    indenter_force_sum_dict[key] = np.array(
        [ f[indenter_selection].arrays[key].sum(axis=0) 
             for f in tmp_traj ] )
    nonindenter_force_sum_dict[key] = np.array(
        [ f[~indenter_selection].arrays[key].sum(axis=0) 
             for f in tmp_traj ] )


# %%


data_prefix = '/work/ws/nemo/fr_jh1130-201708-0/jobs/doc/md/surfactants/data/forces'


# %%


indenter_force_z_sum_dict = { key: value[:,2] for key, value 
                        in indenter_force_sum_dict.items() }


# %%


nonindenter_force_z_sum_dict = { key: value[:,2] for key, value 
                        in nonindenter_force_sum_dict.items() }


# %%


indenter_force_z_sum_df = pd.DataFrame.from_dict(indenter_force_z_sum_dict, dtype=float)


# %%


nonindenter_force_z_sum_df = pd.DataFrame.from_dict(nonindenter_force_z_sum_dict, dtype=float)


# %%


production_steps = 500000


# %%


production_dict[production_steps]


# %%


system_prefix = system_name + '_' + production_dict[production_steps]


# %%


indenter_z_forces_json_name = system_prefix + '_indenter_z_forces.json'


# %%


nonindenter_z_forces_json_name = system_prefix + '_nonindenter_z_forces.json'


# %%


# indenter_z_forces_csv_name = system_name + '_' + production_dict[500000] + '_indenter_z_forces.csv'


# %%


indenter_force_z_sum_df.to_json(
    os.sep.join((data_prefix, indenter_z_forces_json_name)),  orient='index')


# %%


nonindenter_force_z_sum_df.to_json(
    os.sep.join((data_prefix, nonindenter_z_forces_json_name)),  orient='index')


# %%





# %%





# %%


pmf_df = df_dict[system_name][production_steps]['ti.pmf']


# %%


colvars_traj_df = df_dict[system_name][production_steps]['colvars']


# %%


thermo_df = df_dict[system_name][production_steps]['thermo']


# %%


# initial apex position at frame 0:
apex_z0 = tmp_traj[0][indenter_selection].get_positions()[:,2].min()


# %%


# initial substrate surface position at frame 0:
surface_z0 = tmp_traj[0][solid_selection & ~indenter_selection].get_positions()[:,2].max()


# %%


extents_dist_0 = apex_z0 - surface_z0


# %%


extents_dist_0


# %%


com_com_dist_0 = colvars_traj_df.loc[0,'com_com_dist_z']


# %%


com_com_dist_0


# %%


dist_offset = com_com_dist_0 - extents_dist_0


# %%


dist_offset


# %%


df_dict[system_name][production_steps].keys()


# %%


len(thermo_df.index)


# %%


thermo_df.index


# %%


len(colvars_traj_df)


# %%


colvar_freq = 1 # every step


# %%


netcdf_freq = 1e-3 # every 1000 steps


# %%


indenter_force_z_sum_df.set_index(
    (indenter_force_z_sum_df.index/netcdf_freq*colvar_freq).astype(int),
    inplace=True )


# %%


colvars_traj_df_dist_column = "com_com_dist_z"


# %%


#colvars_traj_df[colvars_traj_df_dist_column]


# %%


d_interval = slice(25,80)


# %%


postprocessing.makeVariableByDistPlot(
    indenter_force_z_sum_df[['f_storeUnconstrainedForcesAve']]*force_conversion_factor, 
    colvars_traj_df, pmf_df,
    x_offset = dist_offset,
    interval=d_interval, window=1,
    x_label=r'Distance $ d \ \left( \mathrm{\AA} \right)$',
    y_label=r'Normal force $ F \ \left( \mathrm{nN} \mathrm{mol}^{-1} \right)$')


# %%


indenter_force_z_sum_df


# %%


interval = slice(100,400)


# %%


plt.plot(indenter_force_z_sum_df['f_storeUnconstrainedForcesAve']*force_conversion_factor)
plt.plot(nonindenter_force_z_sum_df['f_storeUnconstrainedForcesAve']*force_conversion_factor)


# %%


plt.plot(indenter_force_z_sum_df['f_storeUnconstrainedForcesAve'][interval]*force_conversion_factor_per_mole)
plt.plot(nonindenter_force_z_sum_df['f_storeUnconstrainedForcesAve'][interval]*force_conversion_factor_per_mole)


# %%


'f_storeAnteSHAKEForcesAve', 
'f_storeAnteStatForcesAve', 
'f_storeUnconstrainedForcesAve' 


# %%


plt.plot(indenter_force_z_sum_df['f_storeAnteStatForcesAve']*force_conversion_factor_per_mole)
plt.plot(nonindenter_force_z_sum_df['f_storeAnteStatForcesAve']*force_conversion_factor_per_mole)


# %%


plt.plot(indenter_force_z_sum_df['f_storeAnteStatForcesAve'][interval]*force_conversion_factor_per_mole)
plt.plot(nonindenter_force_z_sum_df['f_storeAnteStatForcesAve'][interval]*force_conversion_factor_per_mole)


# %%


plt.plot(indenter_force_z_sum_df['f_storeAnteSHAKEForcesAve']*force_conversion_factor_per_mole)
plt.plot(nonindenter_force_z_sum_df['f_storeAnteSHAKEForcesAve']*force_conversion_factor_per_mole)


# %%


plt.plot(indenter_force_z_sum_df['f_storeAnteSHAKEForcesAve'][interval]*force_conversion_factor_per_mole)
plt.plot(nonindenter_force_z_sum_df['f_storeAnteSHAKEForcesAve'][interval]*force_conversion_factor_per_mole)


# %%


plt.plot(indenter_force_z_sum_df['forces'][interval]*force_conversion_factor_per_mole)
plt.plot(nonindenter_force_z_sum_df['forces'][interval]*force_conversion_factor_per_mole)


# %%


force_sum_dict['f_storeUnconstrainedForcesAve'][:,2]


# %%


# 3541 in indenter, 18360 in substrate


# %%


f_indenter.arrays.keys()


# %%


force_sum_dict = {}


# %%


for key in force_keys:
    force_sum_dict[key] = f_indenter.arrays[key].sum(axis=0)


# %%





# %%


force_sum_dict


# %%


get_ipython().run_line_magic('pinfo', 'NetCDFTrajectory')


# %%


traj = ase.io.read(netcdf,index=0,format='netcdftrajectory')


# %%


ase.__version__


# %%


data_df


# %%




