# coding: utf-8

# # CTAB on AU 111 LAMMPS

# This notebook contains a set of code snippets and functions, supporting analysis of LAMMPS output.
# Exemplarily, the minimization, equilibration and production trajectories of one CTA+ ion in
# the vicinity of a 111 gold layer are investigated.
#
# With the help of numpy, pandas and matplotlib, LAMMPS output energy conbtributions are evaluated and plotted.
# With the help of ase, parmed, nglview and ipywidgets, trajectories are visulized.
#
# With the help of ase and asap, radial distribution functions, distances, displacements and diffusivities are evaluated.

# ## Header

## preferred installation method for netcdf on NEMO locally:
# module load mpi/openmpi/2.1-gnu-5.2
# module load {...}
# export CPPFLAGS="${CPPFLAGS} -I${MPI_INC_DIR}"
# export LDFLAGS="${LDFLAGS} -L${MPI_LIB_DIR}"
# pip install --user netCDF4

## alternatively:
# pip install --user --global-option=build_ext --global-option="-L${MPI_INC_DIR}" netCDF4

# for some reason, nglview sometimes changes into some temporary directory
# therefore ALWAY use absolute filenames and paths
# %cd /work/ws/nemo/fr_jh1130-201708-0/jobs/lmplab/sds/201806/1_SDS_on_AU_100_1x4x4/

# ### Imports

# system basics
import os
import subprocess
from glob import glob

# data analysis
import pandas as pd
import numpy as np

import ase
from asap3.analysis.rdf import RadialDistributionFunction

# file formats, input - output
import ase.io
from ase.io import read
from ase.io import NetCDFTrajectory
import parmed as pmd

# visualization
from ase.visualize import view
# import nglview as nv
import matplotlib.pyplot as plt
# import ipywidgets # just for jupyter notebooks

# import ipyparallel as ipp

# ### Global options

absolute_prefix = os.getcwd() # might be handy to get back to the initial working directory at any point

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


# ### Definition of helper functions

# conversion units, only for better readability
fs = 1e-15 # s
ps = 1e-12 # s
AA = 1e-10 # m

dt = 2e-15 # s, 2 fs timestep


def fullprint(*args, **kwargs):
    """prints a long numpy array without altering numpy treshold options permanently"""
    from pprint import pprint
    opt = np.get_printoptions()
    np.set_printoptions(edgeitems=3,infstr='inf',
        linewidth=75, nanstr='nan', precision=8,
        suppress=False, threshold=100000, formatter=None)
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)

def runningMeanFast(x, N):
    """a quick way to compute the running or rolling mean on a numpy array"""
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def running_mean(x, N):
    """another quick way to compute the running or rolling mean on a numpy array"""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def subplotPosition(rows,cols):
    """generator for subplot positions"""
    for p in range(0,rows*cols):
        yield rows*100+cols*10+p+1

def addSubplot(x, y,
               title=None, xlabel=None, ylabel=None, legend=None,
               fig=None, ax=None, pos=None, figsize=(8,5)):
    """facilitate matplotlib figure & subplot creation. only one data series per call."""

    if not pos:
        pos = 111
    if not fig and not ax:
        fig = plt.figure(figsize=figsize)
    elif not fig and ax:
        fig = ax.get_figure()
    if not ax:
        ax = fig.add_subplot(pos)

    if legend:
        ax.plot(x,y,label=legend)
    else:
        ax.plot(x,y)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if legend:
        ax.legend()

    return fig, ax


def makeThermoPlotsFromDataFrame(df, fig=None, axes=None,
        time_label          = r'$\frac{\mathrm{Steps}}{2 \mathrm{fs}}$',
        temperature_label   = r'$\frac{T}{\mathrm{K}}$',
        pressure_label      = r'$\frac{P}{\mathrm{atm}}$',
        energy_label        = r'$\frac{E}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1}}$',
        window = 1,
        interval=slice(None),
        legend_prefix = None):
    """Automizes the plotting of thermo output."""

    rows = 3
    cols = 2
    if fig == None:
        fig = plt.figure(figsize=(cols*8,rows*5))

    # def subplotPosition(rows,cols):
    #    for p in range(0,rows*cols):
    #        yield rows*100+cols*10+p+1

    # def addSubplot(df,fig,pos,title,xlabel,ylabel):
    #     ax = fig.add_subplot(pos)
    #     df.plot(ax=ax) # taimed temperature
    #     ax.set_title(title)
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     return ax

    pos = subplotPosition(rows,cols)

    if axes is None:
        axes = [None for _ in range(rows*cols)]

    # sum up intramolecular contributions
    df["E_intramolecular"] = df[["E_bond","E_angle","E_dihed"]].sum(axis=1)


    df = df.loc[interval,:].rolling(window=window,center=True).mean()
    x = df.index.get_values()

    i = 0
    curpos = next(pos)
    _, axes[i] = addSubplot(x, df[["Temp"]], legend=legend_prefix,
            fig = fig , ax=axes[i], pos=curpos, title="Temperature",
            xlabel=time_label, ylabel=temperature_label)
    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot(x, df[["Press"]], legend=legend_prefix,
            fig = fig , ax=axes[i], pos=curpos, title="Pressure",
            xlabel=time_label, ylabel=pressure_label)

    # intramolecular contributions (without angle)
    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot(x, df[["E_intramolecular"]],
        legend='{} total intramolecular'.format(legend_prefix),
        title="Intramolecular energies",
        xlabel=time_label, ylabel=energy_label,
        fig = fig , ax=axes[i], pos=curpos )
    _, axes[i] = addSubplot(x, df[["E_bond"]], legend='{} bonds'.format(legend_prefix),
        fig = fig , ax=axes[i], pos=curpos )
    _, axes[i] = addSubplot(x, df[["E_angle"]], legend='{} angles'.format(legend_prefix),
        fig = fig , ax=axes[i], pos=curpos )
    _, axes[i] = addSubplot(x, df[["E_dihed"]], legend='{} dihedrals'.format(legend_prefix),
        fig = fig , ax=axes[i], pos=curpos )

    # intermolecular ("non-bonded") energy contribtutions
    # E_pair is the sum of the three latter, just as E_intramolecular in the plot above
    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot(x, df[["E_pair"]],
        legend='{} total pair-wise'.format(legend_prefix),
        title="Intermolecular (non-bonded) energies",
        xlabel=time_label, ylabel=energy_label,
        fig = fig , ax=axes[i], pos=curpos )
    _, axes[i] = addSubplot(x, df[["E_vdwl"]],
        legend='{} VDWL'.format(legend_prefix),
        fig = fig , ax=axes[i], pos=curpos )
    _, axes[i] = addSubplot(x, df[["E_coul"]],
        legend='{} short range coulombic'.format(legend_prefix),
        fig = fig , ax=axes[i], pos=curpos )
    _, axes[i] = addSubplot(x, df[["E_long"]],
        legend='{} long range coulombic'.format(legend_prefix),
        fig = fig , ax=axes[i], pos=curpos )

    # visualize the difference between total and non-bonded potential:
    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot(x, df[["PotEng"]],
        legend='{} total'.format(legend_prefix),
        title="Total potential and non-bonded potential",
        xlabel=time_label, ylabel=energy_label,
        fig = fig , ax=axes[i], pos=curpos )
    _, axes[i] = addSubplot(x, df[["E_pair"]],
        legend='{} non-bonded'.format(legend_prefix),
        fig = fig , ax=axes[i], pos=curpos )

    i +=1
    curpos = next(pos)
    _, axes[i] = addSubplot(x, df[["TotEng"]],
        legend='{} total'.format(legend_prefix),
        title="Total, kinetic and potential energies",
        xlabel=time_label, ylabel=energy_label,
        fig = fig , ax=axes[i], pos=curpos )
    _, axes[i] = addSubplot(x, df[["KinEng"]],
        legend='{} kinetic'.format(legend_prefix),
        fig = fig , ax=axes[i], pos=curpos )
    _, axes[i] = addSubplot(x, df[["PotEng"]],
        legend='{} potential'.format(legend_prefix),
        fig = fig , ax=axes[i], pos=curpos )

    fig.tight_layout()
    return fig, axes

def makeRollingAverageThermoPlotsFromDataFrame(df, fig=None,
        time_label          = r'$\frac{\mathrm{Steps}}{2 \mathrm{fs}}$',
        temperature_label   = r'$\frac{T}{\mathrm{K}}$',
        pressure_label      = r'$\frac{P}{\mathrm{atm}}$',
        energy_label        = r'$\frac{E}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1}}$',
        window = 1000):
    """Automizes the plotting of thermo output. Displays the rolling average with default window = 1000."""

    rows = 3
    cols = 2
    if fig == None:
        fig = plt.figure(figsize=(cols*8,rows*5))

    def subplotPosition(rows,cols):
        for p in range(0,rows*cols):
            yield rows*100+cols*10+p+1

    def addSubplot(df,fig,pos,title,xlabel,ylabel):
        ax = fig.add_subplot(pos)
        df.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    pos = subplotPosition(rows,cols)

    # sum up intramolecular contributions
    df["E_intramolecular"] = df[["E_bond","E_angle","E_dihed"]].sum(axis=1)

    addSubplot(df[["Temp"]].rolling(window=window,center=True).mean(),
               fig, next(pos), "Temperature", time_label, temperature_label)
    addSubplot(df[["Press"]].rolling(window=window,center=True).mean(),
               fig, next(pos), "Pressure", time_label, pressure_label)

    # intramolecular contributions (without angle)
    addSubplot(
      df[["E_intramolecular","E_bond","E_angle","E_dihed"]].rolling(window=window,center=True).mean(),
      fig, next(pos), "Intramolecular energies", time_label, energy_label)
    # intermolecular ("non-bonded") energy contribtutions
    # E_pair is the sum of the three latter, just as E_intramolecular in the plot above
    addSubplot(df[["E_pair","E_vdwl","E_coul","E_long"]].rolling(window=window,center=True).mean(),
                fig, next(pos), "Intermolecular (non-bonded) energies", time_label, energy_label)

    # visualize the difference between total and non-bonded potential:
    addSubplot(df[["PotEng","E_pair"]].rolling(window=window,center=True).mean(),
               fig, next(pos), "Total potential and non-bonded potential", time_label, energy_label)

    addSubplot(df[["TotEng","KinEng","PotEng"]].rolling(window=window,center=True).mean(),
               fig, next(pos), "Total, kinetic and potential energies", time_label, energy_label)

    fig.tight_layout() # tigh_layout avoids label overlap
    return fig

def makeGroupGroupInteractionsPlot(df, fig=None, axes=None,
        time_label          = r'$\frac{\mathrm{Steps}}{2 \mathrm{fs}}$',
        temperature_label   = r'$\frac{T}{\mathrm{K}}$',
        pressure_label      = r'$\frac{P}{\mathrm{atm}}$',
        energy_label        = r'$\frac{E}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1}}$',
        force_label         = r'$\frac{F}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1} \AA^{-1} }$',
        distance_label      = r'$\frac{d}{\AA}$',
        window        = 1,
        interval      = slice(None),
        legend_prefix = None ):
    """Automizes the plotting of group group interactions."""

    # plot group interactions in z direction
    # interactions_of_interest = [
    #     'c_substrate_solvent_interaction[3]',
    #     'c_substrate_surfactant_interaction[3]',
    #     'c_indenter_substrate_interaction[3]',
    #     'c_indenter_surfactant_interaction[3]',
    #     'c_indenter_solvent_interaction[3]',
    #     'c_indenter_ion_interaction[3]' ]

    rows = 3
    cols = 2

    if fig is None:
        # fig = plt.figure(figsize=(cols*8,rows*5))
        fig, ax_array = plt.subplots(rows, cols,squeeze=False,
            figsize=(cols*8,rows*5))

    if axes is None:
        # axes = [None for _ in range(rows*cols)]
        axes = [ax for ax_list in ax_array for ax in ax_list]

    df = df.loc[interval,:].rolling(window=window,center=True).mean()
    x = df.index.get_values() * dt

    pos = subplotPosition(rows,cols)

    i = 0
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_substrate_solvent_interaction"]],
        legend = "{} absolute value".format(legend_prefix),
        title = "substrate - solvent interaction",
        xlabel = time_label, ylabel = energy_label,
        fig = fig, ax=axes[i], pos = curpos )
    _, axes[i] = addSubplot( x, df[["c_substrate_solvent_interaction[3]"]],
        legend = "{} substrate-normal component".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_substrate_surfactant_interaction"]],
        legend = "{} absolute value".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos, title = "substrate - surfactant interaction",
        xlabel = time_label, ylabel = energy_label )
    _, axes[i] = addSubplot( x, df[["c_substrate_surfactant_interaction[3]"]],
        legend = "{} substrate-normal component".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_substrate_interaction"]],
        legend = "{} absolute value".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos, title = "indenter - substrate interaction",
        xlabel = time_label, ylabel = energy_label )
    _, axes[i] = addSubplot( x, df[["c_indenter_substrate_interaction[3]"]],
        legend = "{} substrate-normal component".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_surfactant_interaction"]],
        legend = "{} absolute value".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos, title = "indenter - surfactant interaction",
        xlabel = time_label, ylabel = energy_label )
    _, axes[i] = addSubplot( x, df[["c_indenter_surfactant_interaction[3]"]],
        legend = "{} substrate-normal component".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_solvent_interaction"]],
        legend = "{} absolute value".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos, title = "indenter - solvent interaction",
        xlabel = time_label, ylabel = energy_label )
    _, axes[i] = addSubplot( x, df[["c_indenter_solvent_interaction[3]"]],
        legend = "{} substrate-normal component".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_ion_interaction"]],
        legend = "{} absolute value".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos, title = "indenter - ion interaction",
        xlabel = time_label, ylabel = energy_label )
    _, axes[i] = addSubplot( x, df[["c_indenter_ion_interaction[3]"]],
        legend = "{} substrate-normal component".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos )

    fig.tight_layout()
    return fig, axes

def makeGroupGroupInteractionsByDistPlot( thermo_df, colvars_traj_df, pmf_df,
        fig=None, axes=None,
        time_label          = r'$\frac{\mathrm{Steps}}{2 \mathrm{fs}}$',
        temperature_label   = r'$\frac{T}{\mathrm{K}}$',
        pressure_label      = r'$\frac{P}{\mathrm{atm}}$',
        energy_label        = r'$\frac{E}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1}}$',
        force_label         = r'$\frac{F}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1} \AA^{-1} }$',
        distance_label      = r'$\frac{d}{\AA}$',
        time_factor         = 1,
        temperature_factor  = 1,
        pressure_factor     = 1,
        energy_factor       = 1,
        force_factor        = 1,
        distance_factor     = 1,
        window        = 1,
        interval      = slice(40,100),
        legend_prefix = None,
        x_offset      = 0,
        colvars_traj_df_dist_column = "com_com_dist_z" ):
    """Automizes the plotting of group group interactions."""

    # plot group interactions in z direction
    # interactions_of_interest = [
    #     'c_substrate_solvent_interaction[3]',
    #     'c_substrate_surfactant_interaction[3]',
    #     'c_indenter_substrate_interaction[3]',
    #     'c_indenter_surfactant_interaction[3]',
    #     'c_indenter_solvent_interaction[3]',
    #     'c_indenter_ion_interaction[3]' ]

    # use PMF xi grid as distance bins:
    z_bins = np.array(pmf_df.index)
    # bin colvars time steps by distance:
    binned_steps = pd.cut( colvars_traj_df[colvars_traj_df_dist_column], z_bins )
    # >>> type(binned_steps[0])
    #     pandas._libs.interval.Interval

    # assign each thermo output step its substrate - indenter distance
    # use the bin interval's midpoint
    thermo_df.loc[:,"com_com_dist"] = \
        binned_steps[thermo_df.index].apply( lambda r: r.mid )

    # group thermo output by distance
    grouped_thermo_df = thermo_df.groupby( thermo_df["com_com_dist"] )
    # ATTENTION:
    # >>> type( type(grouped_thermo_df.mean().index) )
    #     pandas.core.indexes.category.CategoricalIndex
    # still contains all bins, even empty ones

    # select only groups with entries
    thermo_df_by_dist = grouped_thermo_df.mean()[ grouped_thermo_df.size() > 0 ]

    # removes empty indices, but not inplace:
    # >>> thermo_df_by_dist.index.remove_unused_categories()

    # add number of values in bin as df column
    thermo_df_by_dist.loc[:,"count"] = \
        grouped_thermo_df.size()[ grouped_thermo_df.size() > 0 ]

    # replaces categorical indices by values
    thermo_df_by_dist.index = thermo_df_by_dist.index.get_values()

    # standard plotting routine

    rows = 4
    cols = 2

    if fig is None:
        # fig = plt.figure(figsize=(cols*8,rows*5))
        fig, ax_array = plt.subplots(rows, cols,squeeze=False,
            figsize=(cols*8,rows*5))

    if axes is None:
        # axes = [None for _ in range(rows*cols)]
        axes = [ax for ax_list in ax_array for ax in ax_list]

    df = thermo_df_by_dist.loc[interval,:].rolling(window=window,center=True).mean()
    x = df.index.get_values() + x_offset

    df.loc[:,"c_indenter_nonindenter_interaction"] = \
        df["c_indenter_surfactant_interaction"] + \
        df["c_indenter_substrate_interaction"] + \
        df["c_indenter_solvent_interaction"] + \
        df["c_indenter_ion_interaction"]

    df.loc[:,"c_indenter_nonindenter_interaction[3]"] = \
        df["c_indenter_surfactant_interaction[3]"] + \
        df["c_indenter_substrate_interaction[3]"] + \
        df["c_indenter_solvent_interaction[3]"] + \
        df["c_indenter_ion_interaction[3]"]

    pos = subplotPosition(rows,cols)

    # fig.tight_layout()
    #
    # ret = [fig]
    # rows = 4
    # cols = 2
    #
    # #if fig == None:
    # fig = plt.figure(figsize=(cols*8,rows*5))
    # pos = subplotPosition(rows,cols)

    # i += 1
    i = 0
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_substrate_solvent_interaction[3]"]]*force_factor,
        legend = legend_prefix,
        fig = fig, ax=axes[i], pos = curpos, title = "substrate - solvent interaction",
        xlabel = distance_label, ylabel = force_label )


    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_substrate_surfactant_interaction[3]"]]*force_factor,
        legend = legend_prefix,
        fig = fig, ax=axes[i], pos = curpos, title = "substrate - surfactant interaction",
        xlabel = distance_label, ylabel = force_label )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_substrate_interaction[3]"]]*force_factor,
        legend = legend_prefix,
        fig = fig, ax=axes[i], pos = curpos, title = "indenter - substrate interaction",
        xlabel = distance_label, ylabel = force_label )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_surfactant_interaction[3]"]]*force_factor,
        legend = legend_prefix,
        fig = fig, ax=axes[i], pos = curpos, title = "indenter - surfactant interaction",
        xlabel = distance_label, ylabel = force_label )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_solvent_interaction[3]"]]*force_factor,
        legend = legend_prefix,
        fig = fig, ax=axes[i], pos = curpos, title = "indenter - solvent interaction",
        xlabel = distance_label, ylabel = force_label )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_ion_interaction[3]"]],
        legend = legend_prefix,
        fig = fig, ax=axes[i], pos = curpos, title = "indenter - ion interaction",
        xlabel = distance_label, ylabel = force_label )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_nonindenter_interaction[3]"]]*force_factor,
        legend = legend_prefix,
        fig = fig, ax=axes[i], pos = curpos, title = "indenter - non-indenter interaction",
        xlabel = distance_label, ylabel = force_label )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["c_indenter_substrate_interaction[3]"]]*force_factor,
        legend = "{} indenter - substrate interaction".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos, title = "normal interactions",
        xlabel = distance_label, ylabel = force_label )
    _, axes[i] = addSubplot( x, df[["c_indenter_surfactant_interaction[3]"]]*force_factor,
        legend = "{} indenter - surfactant interaction".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos )
    # _, axes[i] = addSubplot( x, df[["c_indenter_solvent_interaction[3]"]],
    #    legend = "{} indenter - solvent interaction".format(legend_prefix),
    #    fig = fig, ax=axes[i], pos = curpos )
    _, axes[i] = addSubplot( x, df[["c_indenter_ion_interaction[3]"]]*force_factor,
        legend = "{} indenter - counterion interaction".format(legend_prefix),
        fig = fig, ax=axes[i], pos = curpos )
    # _, axes[i] = addSubplot( x, df[["c_indenter_nonindenter_interaction[3]"]],
    #    legend = "{} indenter - non-indenter interaction".format(legend_prefix),
    #    fig = fig, ax=axes[i], pos = curpos )

    fig.tight_layout()

    # ret.append(fig)
    return fig, axes

    #production_thermo_pd[interactions_of_interest].plot()
    #production_thermo_pd[interactions_of_interest].rolling(window=10,center=True).mean().plot()

def makeVariableByDistPlot( df, colvars_traj_df, pmf_df,
        fig=None, axes=None,
        title         = None,
        x_label       = r'$\frac{d}{\AA}$',
        y_label       = r'$\frac{F}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1} \AA^{-1} }$',
        window        = 1,
        interval      = slice(20,80),
        legend        = None,
        x_offset      = 0,
        colvars_traj_df_dist_column = "com_com_dist_z" ):
    """Automizes the distance-resolved plotting of any time-resolved variable."""

    # use PMF xi grid as distance bins:
    z_bins = np.array(pmf_df.index)
    # bin colvars time steps by distance:
    binned_steps = pd.cut( colvars_traj_df[colvars_traj_df_dist_column], z_bins )
    # >>> type(binned_steps[0])
    #     pandas._libs.interval.Interval

    # assign each thermo output step its substrate - indenter distance
    # use the bin interval's midpoint
    df.loc[:,"com_com_dist"] = \
        binned_steps[df.index].apply( lambda r: r.mid )

    # group thermo output by distance
    grouped_df = df.groupby( df["com_com_dist"] )
    # ATTENTION:
    # >>> type( type(grouped_thermo_df.mean().index) )
    #     pandas.core.indexes.category.CategoricalIndex
    # still contains all bins, even empty ones

    # select only groups with entries
    df_by_dist = grouped_df.mean()[ grouped_df.size() > 0 ]

    # removes empty indices, but not inplace:
    # >>> thermo_df_by_dist.index.remove_unused_categories()

    # add number of values in bin as df column
    # df_by_dist.loc[:,"count"] = \
    #    grouped_df.size()[ grouped_df.size() > 0 ]

    # replaces categorical indices by values
    df_by_dist.index = df_by_dist.index.get_values()

    # standard plotting routine
    rows = 1
    cols = 1

    if fig is None:
        fig, ax_array = plt.subplots(rows, cols,squeeze=False)
            #figsize=(cols*8,rows*5))

    if axes is None:
        axes = [ax for ax_list in ax_array for ax in ax_list]

    df = df_by_dist.loc[interval,:].rolling(window=window,center=True).mean()
    x = df.index.get_values() - x_offset

    pos = subplotPosition(rows,cols)

    i = 0
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df,
        legend = legend,
        fig = fig, ax=axes[i], pos = curpos, title = title,
        xlabel = x_label, ylabel = y_label )

    fig.tight_layout()
    return fig, axes


###  Constraint-related plotting
def makeColvarsPlots( colvars_traj_df, fig=None, axes = None,
        time_label          = r'$\frac{\mathrm{Steps}}{2 \mathrm{fs}}$',
        temperature_label   = r'$\frac{T}{\mathrm{K}}$',
        pressure_label      = r'$\frac{P}{\mathrm{atm}}$',
        energy_label        = r'$\frac{E}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1}}$',
        force_label         = r'$\frac{F}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1} \AA^{-1} }$',
        distance_label      = r'$\frac{d}{\AA}$',
        window = 1,
        legend_prefix = None,
        interval = slice(None) ):
    """Automizes the plotting of colvars output."""
    # Expected columns

    # Index(['indenter_com_substrat', 'v_indenter_com_substr',
    #    'ft_indenter_com_subst', 'fa_indenter_com_subst',
    #    'indenter_com_substrat.1', 'v_indenter_com_substr.1',
    #    'ft_indenter_com_subst.1', 'fa_indenter_com_subst.1',
    #    'indenter_apex_substra', 'ft_indenter_apex_subs',
    #    'indenter_apex_substra.1', 'ft_indenter_apex_subs.1',
    #    'E_indenter_pulled', 'x0_indenter_com_subst', 'W_indenter_pulled'],
    #   dtype='object')

    rows = 2
    cols = 2

    if fig is None:
        fig = plt.figure(figsize=(cols*8,rows*5))
    if axes is None:
        axes = [None for _ in range(rows*cols)]

    pos = subplotPosition(rows,cols)

    # x = pd.DataFrame(colvars_traj_df.index).rolling(
    #         window=window,center=True).mean()[interval] * dt

    df = colvars_traj_df.rolling(window=window,center=True).mean()[interval]
    x = df.index.get_values()

    df.loc[:,"com_com_force_diff_z"] = \
        df["com_com_force_tot_z"] - df["com_com_force_applied_z"]

    i = 0
    curpos = next(pos)
    _, axes[i] = addSubplot( x, df[["com_com_dist_z"]],
                legend = '{} tip COM - substrate COM distance'.format(legend_prefix),
                fig = fig, ax = axes[i], pos = curpos, title = "z distances",
                xlabel = time_label, ylabel = distance_label )
    _, axes[i] = addSubplot( x, df[["apex_com_dist_z"]],
                legend = '{} tip apex - substrate COM distance'.format(legend_prefix),
                fig = fig, ax = axes[i], pos = curpos )
    _, axes[i] = addSubplot( x, df[["x0_constraint"]],
                legend = '{} constraint position'.format(legend_prefix),
                fig = fig, ax = axes[i], pos = curpos )

    i += 1
    curpos = next(pos)
    # constaraint energy & work
    _, axes[i] = addSubplot( x, df[["E_constraint"]],
               legend = "{} constraint energy".format(legend_prefix),
               fig = fig, ax = axes[i], pos = curpos,
               title = "constraint energy and accumulated work".format(legend_prefix),
               xlabel = time_label, ylabel = energy_label)
    _, axes[i] = addSubplot( x, df[["W_constraint"]],
               legend = "constraint work".format(legend_prefix),
               fig = fig, ax = axes[i], pos = curpos )


    # z total & applied force com com
    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot(
        x, df[["com_com_force_tot_z"]],
        fig = fig, ax = axes[i], pos = curpos,
        legend = "{} total force on COM-COM distance".format(legend_prefix),
        title = "total and applied force on COM COM dist",
        xlabel = time_label, ylabel = force_label )
    _, axes[i] = addSubplot( x, df[["com_com_force_applied_z"]],
                legend = "{} applied force on COM-COM distance".format(legend_prefix),
                fig = fig, ax = axes[i], pos = curpos )
    _, axes[i] = addSubplot( x, df[["com_com_force_diff_z"]],
                legend = "{} difference between total and applied force on COM-COM distance".format(legend_prefix),
                fig = fig, ax = axes[i], pos = curpos )

    # # # z total & applied force com com
    # addSubplot(
    #     df[["ft_indenter_com_subst.1", "fa_indenter_com_subst.1"]],
    #     colvars_traj_df[['ft_indenter_com_subst.1']].rolling(window=100,center=True).mean(),
    #            fig, next(pos), "Total and applied force on COM COM dist",
    #            time_label, energy_label)

    # addSubplot(
    #     df[['indenter_com_substrat.1']].rolling(window=window,center=True).mean(),
    #     df[['ft_indenter_com_subst.1']].rolling(window=window,center=True).mean(),
    #     fig = fig, pos = next(pos),
    #     title = 'Total force on COM COM distance',
    #     xlabel = distance_label, ylabel = force_label)
    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot(
        x, df[["com_com_force_diff_z"]],
        fig = fig, ax=axes[i], pos = curpos,
        title = "counter-force on COM-COM distance",
        legend = legend_prefix,
        xlabel = time_label, ylabel = force_label )

    fig.tight_layout()
    return fig, axes

    # plt.plot(colvars_traj_df[['ft_indenter_com_subst.1']].rolling(window=100,center=True).mean(), label='Total force on tip COM')
    # plt.plot(colvars_traj_df[['fa_indenter_com_subst.1']].rolling(window=100,center=True).mean(), label='Applied force on tip COM')
    # plt.legend()

    #addSubplot( df.index, df[["W_indenter_pulled"]],
    #            fig, next(pos), "Constraint work", time_label, energy_label)
    # addSubplot(['E_indenter_pulled'], label='Constraint eneryg')

    # # z distance COM - COM, apex COM, constraint
    # plt.plot(colvars_traj_df['indenter_com_substrat.1'], label='Tip COM - substrate COM distance')
    # plt.plot(colvars_traj_df['indenter_apex_substra.1'], label='Tip apex - substrate COM distance')
    # plt.plot(colvars_traj_df['x0_indenter_com_subst'], label='Constraint position')
    # plt.legend()

    # # constaraint energy & work
    # plt.plot(colvars_traj_df['E_indenter_pulled'], label='Constraint eneryg')
    # plt.plot(colvars_traj_df['W_indenter_pulled'], label='Constraint work')
    # plt.legend()
    #
    # # z total & applied force com com
    # plt.plot(colvars_traj_df['ft_indenter_com_subst.1'], label='Total force on tip COM')
    # plt.plot(colvars_traj_df['fa_indenter_com_subst.1'], label='Applied force on tip COM')
    #
    # # z total & applied force com com
    # plt.plot(colvars_traj_df[['ft_indenter_com_subst.1']].rolling(window=100,center=True).mean(), label='Total force on tip COM')
    # plt.plot(colvars_traj_df[['fa_indenter_com_subst.1']].rolling(window=100,center=True).mean(), label='Applied force on tip COM')
    # plt.legend()
    #
    # # z total force com com
    # plt.plot(colvars_traj_df[['ft_indenter_com_subst.1']].rolling(window=100,center=True).mean(), label='Total force on tip COM')
    #
    # # z distance com com
    # plt.plot(colvars_traj_df[['indenter_com_substrat.1']].rolling(window=100,center=True).mean(), colvars_traj_df[['ft_indenter_com_subst.1']].rolling(window=100,center=True).mean(), label='Total force on tip COM')
    #
    # # z velocity com com dist
    # plt.plot(colvars_traj_df[['v_indenter_com_substr.1']].rolling(window=100,center=True).mean(), label='Indenter approach velocity')


def makeColvarsPlotsByDist( colvars_traj_df, pmf_df, fig=None, axes=None,
        time_label          = r'$\frac{\mathrm{Steps}}{2 \mathrm{fs}}$',
        temperature_label   = r'$\frac{T}{\mathrm{K}}$',
        pressure_label      = r'$\frac{P}{\mathrm{atm}}$',
        energy_label        = r'$\frac{E}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1}}$',
        force_label         = r'$\frac{F}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1} \AA^{-1} }$',
        distance_label      = r'$\frac{d}{\AA}$',
        window = 1,
        interval=slice(40,100),
        legend_prefix = None ):
    """Automizes the plotting of colvars output."""
    # Expected columns

    # Index(['indenter_com_substrat', 'v_indenter_com_substr',
    #    'ft_indenter_com_subst', 'fa_indenter_com_subst',
    #    'indenter_com_substrat.1', 'v_indenter_com_substr.1',
    #    'ft_indenter_com_subst.1', 'fa_indenter_com_subst.1',
    #    'indenter_apex_substra', 'ft_indenter_apex_subs',
    #    'indenter_apex_substra.1', 'ft_indenter_apex_subs.1',
    #    'E_indenter_pulled', 'x0_indenter_com_subst', 'W_indenter_pulled'],
    #   dtype='object')

    # use PMF xi grid as distance bins:
    z_bins = np.array(pmf_df.index)
    # bin colvars time steps by distance:
    binned_steps = pd.cut( colvars_traj_df["com_com_dist_z"], z_bins )
    # >>> type(binned_steps[0])
    #     pandas._libs.interval.Interval

    # assign each thermo output step its substrate - indenter distance
    # use the bin interval's midpoint
    colvars_traj_df.loc[:,"binned_com_com_dist"] = \
        binned_steps[colvars_traj_df.index].apply( lambda r: r.mid )

    # group thermo output by distance
    grouped_colvars_traj_df = colvars_traj_df.groupby( colvars_traj_df[
        "binned_com_com_dist"] )
    # ATTENTION:
    # >>> type( type(grouped_thermo_df.mean().index) )
    #     pandas.core.indexes.category.CategoricalIndex
    # still contains all bins, even empty ones

    # select only groups with entries
    colvars_traj_df_by_dist = grouped_colvars_traj_df.mean()[
        grouped_colvars_traj_df.size() > 0 ]

    # removes empty indices, but not inplace:
    # >>> thermo_df_by_dist.index.remove_unused_categories()

    # add number of values in bin as df column
    colvars_traj_df_by_dist.loc[:,"count"] = \
        grouped_colvars_traj_df.size()[ grouped_colvars_traj_df.size() > 0 ]

    # replaces categorical indices by values
    colvars_traj_df_by_dist.index = colvars_traj_df_by_dist.index.get_values()

    rows = 3
    cols = 1

    if fig is None:
        fig = plt.figure(figsize=(cols*8,rows*5))
    if axes is None:
        axes = [None for _ in range(rows*cols)]

    pos = subplotPosition(rows,cols)

    #x = pd.DataFrame(colvars_traj_df_by_dist.index).rolling(
    #        window=window,center=True).mean()#.loc[interval,:]
    df = colvars_traj_df_by_dist.rolling(
            window=window,center=True).mean().loc[interval,:]
    x = df.index.get_values()

    i = 0
    curpos = next(pos)
    # constaraint energy & work
    _, axes[i] = addSubplot( x, df[["E_constraint"]],
               legend="{} constraint energy".format(legend_prefix),
               fig = fig, ax=axes[i], pos = curpos,
               title = "constraint energy and accumulated work",
               xlabel = distance_label, ylabel = energy_label)
    _, axes[i] = addSubplot( x, df[["W_constraint"]],
               legend="{} constraint work".format(legend_prefix),
               fig = fig, ax = axes[i], pos = curpos )

    df.loc[:,"com_com_force_diff_z"] = \
        df["com_com_force_tot_z"] - df["com_com_force_applied_z"]

    # z total & applied force com com
    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot(
        x, df[["com_com_force_tot_z"]],
        fig = fig, ax = axes[i], pos = curpos,
        legend = "{} total force on COM-COM distance".format(legend_prefix),
        title = "total and applied force on COM COM dist",
        xlabel = distance_label, ylabel = force_label )
    _, axes[i] = addSubplot( x, df[["com_com_force_applied_z"]],
                legend = "{} applied force on COM-COM distance".format(legend_prefix),
                fig = fig, ax = axes[i], pos = curpos )
    _, axes[i] = addSubplot( x, df[["com_com_force_diff_z"]],
                legend = "{} Difference between total and applied force on COM-COM distance".format(legend_prefix),
                fig = fig, ax = axes[i], pos = curpos )

    i += 1
    curpos = next(pos)
    _, axes[i] = addSubplot(
        x, df[["com_com_force_diff_z"]],
        fig = fig, ax = axes[i], pos = curpos,
        legend = legend_prefix,
        title = "counter-force on COM-COM distance",
        xlabel = distance_label, ylabel = force_label )

    fig.tight_layout()
    return fig, axes

def makePMEPlots( pmf_df, grad_df = None, count_df = None, fig=None, axes=None,
        time_label          = r'$\frac{\mathrm{Steps}}{2 \mathrm{fs}}$',
        temperature_label   = r'$\frac{T}{\mathrm{K}}$',
        pressure_label      = r'$\frac{P}{\mathrm{atm}}$',
        energy_label        = r'$\frac{E}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1}}$',
        force_label         = r'$\frac{F}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1} \AA^{-1} }$',
        distance_label      = r'$\frac{d}{\AA}$',
        interval            = slice(None),
        window              = 1,
        legend_prefix = None ):
    """Automizes the plotting of colvars output."""
    # Expected columns
    # xi A(xi)

    rows = 1
    if grad_df is not None:
        rows += 1
    if count_df is not None:
        rows += 1

    cols = 1

    if fig is None:
        fig = plt.figure(figsize=(cols*8,rows*5))
    if axes is None:
        axes = [None for _ in range(rows*cols)]

    pos = subplotPosition(rows,cols)

    pmf_df = pmf_df.rolling(window=window,center=True).mean().loc[interval,:]
    pmf_x = pmf_df.index.get_values()

    grad_df = grad_df.rolling(window=window,center=True).mean()[interval]
    grad_x = grad_df.index.get_values()

    count_df = count_df.rolling(window=window,center=True).mean()[interval]
    count_x = count_df.index.get_values()

    #curpos = next(pos)
    i = 0
    _, axes[i] = addSubplot( pmf_x, pmf_df[["pmf"]],
        fig = fig, ax = axes[i], pos = next(pos),
        title = "substrate COM - tip COM distance PMF",
        legend = legend_prefix,
        xlabel = distance_label, ylabel = energy_label )

    if grad_df is not None:
        i += 1
        _, axes[i] = addSubplot( grad_x, grad_df[["grad"]],
            fig = fig, ax = axes[i], pos = next(pos),
            title = "substrate COM - tip COM distance mean thermodynamic force",
            legend = legend_prefix,
            xlabel = distance_label, ylabel = force_label )

    if count_df is not None:
        i += 1
        _, axes[i] = addSubplot( count_x, count_df[["count"]],
            fig = fig, ax = axes[i], pos = next(pos),
            title = "substrate COM - tip COM distance sample histogram",
            legend = legend_prefix,
            xlabel = distance_label, ylabel = 'N' )

    fig.tight_layout()
    return fig, axes

# ASE by default infers elements from LAMMPS atom types, in our case they are unrelated
# During preprocessing, our system went through several formats, one of them the
# archaic .pdb format. Although ASE offers a pdb reader, it fails on our system.

# On the other hand, ParmEd is able to read .pdb and infer elements more or less accurately,
# but cannot process netCDF. Thus we combine both:
def inferTypes2NumbersFromPdbAndLmp(pdb_file,lmp_data_file):
    """Uses parmed's ability to infer elements from pdb files and constructs a dictionary and
    dictionary-like array for atom type -> atomic number assignments"""
    struct_pdb_pmd = pmd.read_PDB(pdb_file)
    struct_lmp_data = ase.io.read(lmp_data_file,format='lammps-data')

    resnames = np.unique([r.name for r in struct_pdb_pmd.residues])
    print("PDB contains following residue types {}".format(resnames))

    ions = [ a for a in struct_pdb_pmd.atoms if a.residue.name == 'ION' ] # sodium counterions

    # sodium correction: apparently, SOD and S are both interpreted as sulfur
    # bromide correction: in the pdb, the bromide ion is marked as CLA (chloride ion),
    #   because charmm2lammps.pl only supports placing Na+ or Cl- as counterions when preparing the system
    #   The chloride paramters have been changed to Horinek's bromide parameters manually after conversion
    #   within the LAMMPS data file. What is more, the element C (carbon) is wrongly inferred
    #   from residue name CLA by Parmed. Thus, atomic number 6 (C) is changed to Bromide her:
    for ion in ions:
        if ion.atomic_number == 16: # wrong: inferred S (sulfur)
            ion.atomic_number = 11 # Na
        elif ion.atomic_number == 6: # wrong: inferred C (sulfur)
            ion.atomic_number = 35 # Br

    # elements numbered as in periodic table
    atomic_numbers = np.array([ a.atomic_number for a in struct_pdb_pmd.atoms ])
    atomic_types = struct_lmp_data.get_atomic_numbers() # types as numbered in LAMMPS
    types2numbers = dict(zip(atomic_types,atomic_numbers)) # automatically picks unique tuples

    print("System contains {:d} atom types.".format(len(types2numbers)))

    # construct array, where indices represent LAMMPS type numbers and point to atomic numbers
    types2numbers_array = np.zeros(atomic_types.max()+1,dtype=np.uint)

    for k,v in types2numbers.items():
        types2numbers_array[k] = v

    # this kind of array representation allows for simple type2number conversion via an expression like
    #   types2numbers_array[atomic_types]

    return types2numbers, types2numbers_array

# helper function needs:
# trajectory, segement length, indices, element tuple list
# optional: start, end, rMax, nBins

def piecewiseRDF(traj, atom_indices, element_tuples,
                nSegment = 1000, nStart = 0, nEnd = None,
                rMax = 20, nBins = 1000):
    """Computes time-segment-wise element-element RDFs within a certain group of atoms of a trajectory"""

    # rMax is the rdf length (in Angstrom for LAMMPS output in real units)
    # nBins: can be understood as the number of data points on the RDF
    if not nEnd:
        nEnd = len(traj)-1

    nRDFs = len(element_tuples)

    listOfRdfLists = []
    for n in range(0,nRDFs):
        listOfRdfLists.append([])

    # actual distances
    rdf_x = (np.arange(nBins) + 0.5) * rMax / nBins

    # instead of computing an average rdf over the whole trajectory,
    # we split the trajectory into several timespans of nSegement timestep length
    for curStart in range(nStart,nEnd,nSegment):
        print(curStart) # some progress report
        rdfObj = None
        for frame in traj[curStart:(curStart+nSegment)]:
            # the asap rdf functionality is not that convenient, but explicitely choosing
            # only the atoms we are interested in a priori, we can get exactly the rdf we want by
            # making use of the "elements" option
            if rdfObj is None:
                rdfObj = RadialDistributionFunction(frame[atom_indices],
                                         rMax = rMax, nBins = nBins)
            else:
                rdfObj.atoms = frame[atom_indices]
            rdfObj.update()

        # np.where facilitates the selection of according atom numbers by specifying the chemical symbol
        for n in range(0,nRDFs):
            curRdf = rdfObj.get_rdf(elements=element_tuples[n])
            listOfRdfLists[n].append(curRdf)

    return listOfRdfLists, rdf_x, rdfObj

# code snippet for neat plotting of all time-segemtn rdfs
def plotPiecewiceRdf(rdf_x, listOfRdfLists, legend=None,
                    nSegment = 1000, nStart=0, nEnd = None, cols = 2):
    """Plots a set of rdf constructed by piecewiseRDF(...)"""
    # Todo: Implement arbitrary start and end points


    N = 0
    for rdfList in listOfRdfLists:
        if len(rdfList) > N:
            N = len(rdfList)

    rows = np.ceil(N/cols).astype(int)

    pos = subplotPosition(rows,cols)
    fig = plt.figure(figsize=(rows*5,cols*8))

    for i in range(0,N):
        p = next(pos)
        ax = None
        for j, rdfList in enumerate(listOfRdfLists):
            if i < len(rdfList):
                curLegend = None
                if legend and j < len(legend):
                    curLegend = legend[j]

                if not ax:
                    _, ax = addSubplot(rdf_x,rdfList[i],
                               legend = curLegend,
                               xlabel = r'$\frac{r}{\AA}$',
                               ylabel='arbitrary density',
                               title = "{} ps - {} ps".format(i*nSegment,(i+1)*nSegment),
                               fig = fig, pos = p)
                else:
                    _, _ = addSubplot(rdf_x, rdfList[i],
                              legend = curLegend,
                              ax = ax, pos = p)

    fig.tight_layout()
    return fig

def piecewiseAveragedDistance(traj, reference_index, atom_indices,
                nSegment = 1000, nStart = 0, nEnd = None):
    """Piecewise averaged distance from one reference atom to a group of other atoms"""
    if not nEnd:
        nEnd = len(traj)-1


    N = (np.floor(nEnd - nStart)/nSegment).astype(int)

    t = (np.arange(nStart,nEnd,nSegment) + nSegment / 2)

    averageDistance = np.atleast_2d(np.zeros((3,N)))

    for i, curStart in enumerate(range(nStart,nEnd,nSegment)):
        #print(curStart) # some progress report
        curAverageDistance = np.atleast_2d(np.zeros((3,nSegment)))
        for j,f in enumerate(traj[curStart:(curStart+nSegment)]):
            curAverageDistance[:,j] = np.abs( f.get_distances(
                reference_index, atom_indices, mic=True, vector=True).mean(axis=0) )
        averageDistance[:,i] = curAverageDistance.mean(axis=1)

    return averageDistance, t

def piecewiseAveragedComComDistance(traj, group1_indices, group2_indices,
                nSegment = 1000, nStart = 0, nEnd = None):
    """Piecewise averaged distance from one atom group's center of mass to another one's"""

    if not nEnd:
        nEnd = len(traj)-1


    N = (np.floor(nEnd - nStart)/nSegment).astype(int)

    t = (np.arange(nStart,nEnd,nSegment) + nSegment / 2)

    averageDistance = np.atleast_2d(np.zeros((3,N)))

    for i, curStart in enumerate(range(nStart,nEnd,nSegment)):
        #print(curStart) # some progress report
        curAverageDistance = np.atleast_2d(np.zeros((3,nSegment)))
        for j,f in enumerate(traj[curStart:(curStart+nSegment)]):
            curAverageDistance[:,j] = np.abs(
                f[group1_indices].get_center_of_mass() - f[group2_indices].get_center_of_mass() )
        averageDistance[:,i] = curAverageDistance.mean(axis=1)

    return averageDistance, t

def comDisplacement(traj, atom_indices, dt = 10):
    """Evaluates the displacement of an atom group's COM
    between each frame of the trajectory and another frame 'dt' indices later."""
    N = len(traj) - dt

    displacement = np.atleast_2d(np.zeros((3,N)))

    for j in range(N):
        reference_com   = traj[j][atom_indices].get_center_of_mass()
        dt_com           = traj[j+dt][atom_indices].get_center_of_mass()
        displacement[:,j] = np.abs(dt_com - reference_com)

    return displacement

def evaluateDisplacement(displacement, dt = 10, window = 500):
    """Evaluates an anisotropic displacement vector and also computes diffusivities
    based on Enstein relation"""
    isotropic_displacement = np.linalg.norm(displacement,axis=0)

    time = np.arange( len(isotropic_displacement) )* TimePerFrame/ps

    rows = 3
    cols = 1

    fig = plt.figure(figsize=(8*cols, 5*rows))

    pos = subplotPosition(cols=cols,rows=rows)

    p = next(pos)
    _, axes[i] = addSubplot(time[:-window+1], running_mean( isotropic_displacement,window),
            title = "{:.2f} ps displacement".format(dt*TimePerFrame/ps),
            xlabel = "time t / ps", ylabel= r'displacement $\frac{r}{\AA}$',
            legend = 'isotropic', fig = fig, pos = p)

    for i in range(0,3):
        addSubplot(time[:-window+1],
                   running_mean(displacement[i,:], window),
                   legend = distanceLabels[i], ax = ax, pos = p)

    p = next(pos)
    _, axes[i] = addSubplot(time[:-window+1], running_mean( isotropic_displacement**2 / 3.0, window),
            title = "{:.2f} ps MSD".format(dt*TimePerFrame/ps),
            xlabel = "time t / ps", ylabel= r'MSD $\frac{r^2}{\AA^2}$',
            legend = 'isotropic', fig = fig, pos = p)

    for i in range(0,3):
        addSubplot(time[:-window+1],
                   running_mean(displacement[i,:]**2, window),
                   legend = distanceLabels[i], ax = ax, pos = p)


    # Einstein relation and converion to SI units
    D    = 1.0/2.0 * displacement**2 * AA**2 / (dt*TimePerFrame)
    Diso = 1.0/6.0 * isotropic_displacement**2 * AA**2 / (dt*TimePerFrame)

    p = next(pos)
    _, axes[i] = addSubplot(time[:-window+1], running_mean( Diso, window),
            title = "diffusivities from {:.2f} ps MSD".format(dt*TimePerFrame/ps),
            xlabel = "time t / ps", ylabel= r'D $\frac{m^2}{s}$',
            legend = 'isotropic', fig = fig, pos = p)


    for i in range(0,3):
        addSubplot(time[:-window+1],
                   running_mean(D[i,:], window),
                   legend = distanceLabels[i], ax = ax, pos = p)

    fig.tight_layout()

    return fig

# ## Colvars
def read_data_with_hashed_header(filename ):
    header = pd.read_csv(filename,
                         delim_whitespace=True, nrows=0)
    columns = header.columns[1:]
    df = pd.read_csv( filename, delim_whitespace=True, header=None, comment='#',
        names=columns)

    return df

def read_colvars_traj(
    colvars_traj_file = None,
    prefix = None,
    column_dict = {
        'indenter_com_substrat'     : 'com_com_dist',
        'v_indenter_com_substr'     : 'com_com_vel',
        'ft_indenter_com_subst'     : 'com_com_force_tot',
        'fa_indenter_com_subst'     : 'com_com_force_applied',
        'indenter_com_substrat.1'   : 'com_com_dist_z',
        'v_indenter_com_substr.1'   : 'com_com_vel_z',
        'ft_indenter_com_subst.1'   : 'com_com_force_tot_z',
        'fa_indenter_com_subst.1'   : 'com_com_force_applied_z',
        'indenter_apex_substra'     : 'apex_com_dist',
        'ft_indenter_apex_subs'     : 'apex_com_force_tot',
        'indenter_apex_substra.1'   : 'apex_com_dist_z',
        'ft_indenter_apex_subs.1'   : 'apex_com_force_tot_z',
        'E_indenter_pulled'         : 'E_constraint', # current energy in constraint
        'x0_indenter_com_subst'     : 'x0_constraint', # current position of constraint
        'W_indenter_pulled'         : 'W_constraint' # work done by moving constraint
    } ):
    # expect only one file
    if colvars_traj_file is None:
        colvars_traj_file_glob_pattern = '*.colvars.traj'
        if prefix is not None:
            colvars_traj_file_glob_pattern = os.sep.join((
                prefix, colvars_traj_file_glob_pattern))

        colvars_traj_file = glob(colvars_traj_file_glob_pattern)[0]

    colvars_traj_df = read_data_with_hashed_header( colvars_traj_file )
    colvars_traj_df.set_index('step',inplace=True)

    colvars_traj_df.rename(columns=column_dict, inplace=True)

    return colvars_traj_df

def read_colvars_ti(prefix = None):
    # expect only one file

    colvars_pmf_file_glob_pattern = '*.ti.pmf'
    colvars_grad_file_glob_pattern = '*.ti.grad'
    colvars_count_file_glob_pattern = '*.ti.count'

    if prefix is not None:
        colvars_pmf_file_glob_pattern = os.sep.join((
            prefix, colvars_pmf_file_glob_pattern))
        colvars_grad_file_glob_pattern = os.sep.join((
            prefix, colvars_grad_file_glob_pattern))
        colvars_count_file_glob_pattern = os.sep.join((
            prefix, colvars_count_file_glob_pattern))

    colvars_pmf_file   = glob(colvars_pmf_file_glob_pattern)[0]
    colvars_grad_file  = glob(colvars_grad_file_glob_pattern)[0]
    colvars_count_file = glob(colvars_count_file_glob_pattern)[0]


    colvars_pmf_df = read_data_with_hashed_header( colvars_pmf_file )
    colvars_pmf_df.set_index('xi', inplace = True)
    colvars_pmf_df.columns = ['pmf']

    colvars_grad_df = pd.read_csv( colvars_grad_file,
        delim_whitespace=True, header=None, comment='#', skiprows=3,
        names=['xi', 'grad'] )
    colvars_grad_df.set_index('xi', inplace = True)
    #colvars_grad_df.columns = ['grad']
    #colvars_pmf_df['grad'] = colvars_grad_df['grad']

    colvars_count_df = pd.read_csv( colvars_count_file,
        delim_whitespace=True, header=None, comment='#', skiprows=3,
        names=['xi', 'count'] )
    colvars_count_df.set_index('xi', inplace = True)
    #colvars_count_df.columns = ['count']
    #colvars_pmf_df['count'] = colvars_count_df['count']

    #colvars_traj_df.set_index('step',inplace=True)
    return colvars_pmf_df, colvars_grad_df, colvars_count_df

def read_production_thermo( production_thermo_file = None, prefix = None ):

    if production_thermo_file is None:
        production_thermo_file_glob_pattern = '*thermo.out'
        if prefix is not None:
            production_thermo_file_glob_pattern = os.sep.join((
                prefix, production_thermo_file_glob_pattern))

        production_thermo_file = glob(production_thermo_file_glob_pattern)[0]


    # subprocess.run(
    #    ['./extract_thermo.sh',production_log_file,'production_thermo.out'],
    #    shell=False, check=True)

    production_thermo_pd = pd.read_csv(
        production_thermo_file, delim_whitespace=True )
    production_thermo_pd.set_index("Step",inplace=True)

    #makeThermoPlotsFromDataFrame(production_1ns_thermo_pd.iloc[::100].copy()); # only every 100th data point
    # makeThermoPlotsFromDataFrame(production_thermo_pd.copy()); # only every 100th data point
    # makeRollingAverageThermoPlotsFromDataFrame(production_thermo_pd.copy(),window=10);
    # production_thermo_pd[["PotEng","E_pair"]].rolling(window=10,center=True).mean().plot()
    return production_thermo_pd

# ## Energy evaluations with pandas

# ### Minimization

def evaulate_minimization():

    minimization_thermo_file = glob("*_minimization_thermo.out")[0]

    minimization_thermo_pd = pd.read_csv(minimization_thermo_file,delim_whitespace=True)

    minimization_thermo_pd.set_index("Step",inplace=True)

    makeThermoPlotsFromDataFrame(minimization_thermo_pd)

    # long Coulombic interaction (by PPPME)
    minimization_thermo_pd[["E_long"]][2:].plot()

    # The total energy decreases, but intramolecular energy increases during minimization:
    minimization_thermo_pd[["PotEng","E_pair"]][2:].plot()

    # double-check: total potential energy of system minus non-bonded energy (LJ & Coulomb)
    # should correspond to intramolecular energy:
    intramolecularEnergyValidation = minimization_thermo_pd["PotEng"] - minimization_thermo_pd["E_pair"]

    # intramolecularEnergyValidationDiff = (intramolecularEnergyValidation - minimization_thermo_pd["E_intramolecular"])
    # intramolecularEnergyValidationDiff.max()
    # intramolecularEnergyValidationDiff.abs().max() / intramolecularEnergyValidation.min()

    (intramolecularEnergyValidation - minimization_thermo_pd["E_intramolecular"])[1:].plot()
    # obviously "equal" (up to a tiny fraction)


    # descent to steep t the first few steps, excluded
    makeThermoPlotsFromDataFrame(minimization_thermo_pd.iloc[2:].copy());


# ## Trajectory visualization with ASE and ParmEd

# type dict, manually for SDS
sds_t2n = {1: 1, 2: 1, 3: 6, 4: 6, 5: 8, 6: 8, 7: 16, 8: 1, 9: 8, 10: 11, 12: 79}
sds_t2n_array = np.array([0,*list(sds_t2n.values())],dtype='uint64')
sds_t2e_array = np.array(ase.data.chemical_symbols)[sds_t2n_array] # double-check against LAMMPS data file

# type dict, manually for CTAB
ctab_t2n = {1: 1, 2: 1, 3: 1, 4: 6, 5: 6, 6: 6, 7: 7, 8: 1, 9: 8, 10: 35, 11: 79}
ctab_t2n_array = np.array([0,*list(ctab_t2n.values())],dtype='uint64')
ctab_t2e_array = np.array(ase.data.chemical_symbols)[ctab_t2n_array] # double-check against LAMMPS data file

t2n_array = ctab_t2n_array

def read_frames(lmp_files = None):
    global t2n_array
    if lmp_files is None:
        data_files = glob('*.lammps')
        data_file = data_files[-1]
        lmp_files  = { 'initial' : data_file }



    # pdb_file_initial_config = glob('*psfgen_ctrl.pdb')[0]
    # lmp_file_initial_config = glob('*psfgen.data')[0]

    #
    # lmp_files = { 'initial':         prefix + 'psfgen_CLA2BR.data',
    #               'minimized':       prefix + 'minimized.lammps',
    #               'nvtEquilibrated': prefix + 'nvtEquilibrated.lammps',
    #               'nptEquilibrated': prefix + 'nptEquilibrated.lammps',
    #               #'npt10ps':         prefix + '10ps_npt_final_config.lammps',
    #               #'npt100ps':        prefix + '100ps_npt_final_config.lammps',
    #               'npt1ns':          prefix + '1ns_npt_final_config.lammps',
    #               'nve1ns':          prefix + '1ns_nve_final_config.lammps'
    #         }
    # lmp_dumps = {
    #              'nvtEquilibration': prefix + 'nvtEquilibration.dump',
    #              #'nptEquilibration': prefix + 'nptEquilibration.dump',
    #              #'npt10ps':          prefix + '10ps_npt_nptProduction.dump',
    #              #'npt100ps':         prefix + '100ps_npt_nptProduction.dump',
    #              #'npt1ns':           prefix + '1ns_npt_with_restarts_nptProduction.dump',
    #              #'nve1ns':           prefix + '1ns_nve_with_restarts_nveProduction.dump'
    #             }
    # lmp_netcdf = {
    #               'nvtEquilibration': prefix + 'nvtEquilibration.nc',
    #               'nptEquilibration': prefix + 'nptEquilibration.nc',
    #               'npt1ns':           prefix + '1ns_npt_nptProduction.nc',
    #               'nve1ns':           prefix + '1ns_nve_nveProduction.nc'
    #              }
    # construct a dictionary-like atom type-> atom number array
    # t2n, t2n_array = inferTypes2NumbersFromPdbAndLmp(pdb_file_initial_config, lmp_file_initial_config

    # # create atom selections for later post-processing:
    # struct_pdb_pmd     = pmd.read_PDB(pdb_file_initial_config)
    # water              = [ a for a in struct_pdb_pmd.atoms if a.residue.name == 'HOH' ] # water
    # water_indices      = [ a.number - 1 for a in water ] # to remove water atoms later
    # surface            = [ a for a in struct_pdb_pmd.atoms if a.residue.name == 'AUM' ] # gold surface
    # surface_indices    = [ a.number - 1 for a in surface ]
    # surfactant         = [ a for a in struct_pdb_pmd.atoms if a.residue.name == 'CTA' ]
    # surfactant_indices = [ a.number - 1 for a in surfactant ]
    # ions               = [ a for a in struct_pdb_pmd.atoms if a.residue.name == 'ION' ] # sodium counterions
    # ion_indices        = [ a.number - 1 for a in ions ]


# ### LAMMPS data files

    # read frames of interest
    lmp_frames = {}
    for k,f in lmp_files.items():
        lmp_frames[k] = read(lmp_files[k],format='lammps-data')
        lmp_frames[k].set_atomic_numbers(
            t2n_array[lmp_frames[k].get_atomic_numbers() ] )

        # lmp_frames['initial']


    # lmp_views = []
    # for k, f in lmp_frames.items():
    #     lmp_views.append( nv.show_ase(f) )
    #     lmp_views[-1]._set_sync_camera()
    #     lmp_views[-1]._remote_call("setSize", target="Widget", args=["250px", "250px"])
    #     lmp_views[-1].center()
    #     lmp_views[-1].render_image()

    # frames stripped of all oxygen & hydrogen:
    # apparently, order of atoms in dump trajectories is fixed, but
    # in LAMMPS data files not, hence the indexing does not work based upon previously derived sets
    lmp_naked_frames = {}
    for k,f in lmp_frames.items():
        #lmp_naked_frames[k] = []
        #or f in lmp_frames[k]:
        #g = f.copy()
        g = f[ (np.array(f.get_chemical_symbols()) != 'O') & (np.array(f.get_chemical_symbols()) != 'H')]
        #del g[water_indices]
        lmp_naked_frames[k] = g.copy()

    # lmp_naked_frames["initial"]
    return lmp_naked_frames, lmp_frames

def read_trajs( lmp_traj_files = None ):
    global t2t2n_array
    # lmp_traj_files: {str: str} dict

    if traj_file is None:
        traj_files = glob('*.nc')
        traj_file = traj_files[-1]
        lmp_netcdf = { 'production' : traj_file }

    lmp_trajectrories = {}
    for k,t in lmp_netcdf.items():
        #lmp_trajectrories[k] = read(t, index=':',format='lammps-dump')
        tmp_traj = NetCDFTrajectory(lmp_netcdf[k], 'r',
                            types_to_numbers=list(t2n_array),
                            keep_open=True)
        # workaround to center trajecory in box
        # (ASE places the box origin at 0,0,0 by default)
        lmp_trajectrories[k] = []
        for i,frame in enumerate(tmp_traj):
            frame.center()
            lmp_trajectrories[k].append( frame )
        tmp_traj.close()


    # apparently, order of atoms changed in netcdf
    lmp_naked_trajectrories = {}
    for k in lmp_trajectrories:
        lmp_naked_trajectrories[k] = []
        for f in lmp_trajectrories[k]:
            #g = f.copy()
            #del g[water_indices]
            g = f[ (np.array(f.get_chemical_symbols()) != 'O') & (np.array(f.get_chemical_symbols()) != 'H')]
            lmp_naked_trajectrories[k].append(g)

    return lmp_naked_trajectrories, lmp_trajectories



def show_frames(lmp_frames):
    lmp_views = []
    for k, f in lmp_frames.items():
        lmp_views.append( nv.show_ase(f) )
        lmp_views[-1]._set_sync_camera()
        lmp_views[-1]._remote_call("setSize", target="Widget", args=["250px", "250px"])
        lmp_views[-1].center()
        lmp_views[-1].render_image()

    vbox = ipywidgets.VBox(lmp_views)
    return vbox


    # ### LAMMPS NETCDF trajectories

    # lmp_trajectrories = {}
    # for k,t in lmp_dumps.items():
    #     lmp_trajectrories[k] = read(t, index=':',format='lammps-dump')
    #     for f in lmp_trajectrories[k]:
    #         f.set_atomic_numbers(
    #             t2n_array[f.get_atomic_numbers() ] )
    #         f.center()


    # # trial
    # traj = NetCDFTrajectory(lmp_netcdf['nvtEquilibration'], 'r',
    #                         types_to_numbers=list(t2n_array),
    #                         keep_open=True)
    # # workaround to center trajecory in box
    # # (ASE places the box origin at 0,0,0 by default)
    # frames = []
    # for i,frame in enumerate(traj):
    #     frame.center()
    #     frames.append( frame )
    # traj.close()

# Several problems with nglview:
#  1) does not display or infer bonding for ASE trajectory
#  2) even with displayed gui, not clear how to activate



# ### LAMMPS trajectories, stripped of solvent


def show_traj(lmp_traj):
    trajectoryView = nv.show_asetraj(lmp_traj)
    trajectoryView.remove_ball_and_stick()
    trajectoryView.add_spacefill() # try a different representation sytle
    return trajectoryView

# view(lmp_naked_trajectrories['production'], viewer='ase') # opens ASE GUI
# view(lmp_naked_trajectrories['production'][0], viewer='ngl') # opens ASE GUI
# view(lmp_naked_trajectrories['production'][20], viewer='ngl') # opens ASE GUI

# view(lmp_naked_trajectrories['production'][-1], viewer='ngl') # opens ASE GUI

# Several problems with nglview:
#  1) does not display or infer bonding for ASE trajectory
#  2) even with displayed gui, not clear how to activate
#  3) somehow mixes up the atom type for netcdf

# ### Make a movie via .png frames and ffmpeg

# #### Preparation

# create a subdir from within the notebook
# get_ipython().run_line_magic('mkdir', 'png')

def animate_traj(traj,
        framesPerSecond=30, totalFramesAvailable=5000, desiredVideoDuration=30):
    global absolute_path

    try:
        os.mkdir( 'png' )
    except OSError:
        logging.warn("Subdir 'png' exists")

    # traj = lmp_naked_trajectrories['production']

    # len(traj)

    # from ~ frame 3500 to the end ~ means
    # totalFramesAvailable = 5000
    # desiredVideoDuration = 30 # s
    # framesPerSecond = 30 # s^-1
    neededFrames = desiredVideoDuration*framesPerSecond
    every_nth = np.ceil(totalFramesAvailable / neededFrames).astype(int)

    png_prefix = absolute_prefix + os.sep + 'png' + os.sep + 'traj'

    # #### Orientation and bounding box settings by trial & error

    # f = traj[0].copy()

    # find a desired orientation
    #f.rotate('x', (-1,2,-1), rotate_cell=True)
    #testframe.rotate(90, (1,0,0), rotate_cell=False)
    #f.rotate('z', (1,1,-1), rotate_cell=False)
    #f.rotate(-90,'x', rotate_cell=False)
    # strange, somehow opposite behavior with "in-notebook" viewer and png renderer

    # f.rotate(-60,'x', rotate_cell=False)
    # f.rotate(30,'y', rotate_cell=False)
    # view(f,viewer='ngl')
    # nv.show_ase(f)

    cell = f.get_cell()
    # bbox = [-20, 20, 80, 120 ]

    # one trial
    # the commented lines can be used to replicate the cell
    #cell = f.get_cell()
    #f = f.repeat((3, 1, 3))
    #f.set_cell(cell)
    # bbox = [-5, -5, cell[0,0] + 5, cell[1,1] + 5 ]
    #f.center()
    # ase.io.write(png_prefix + '_test.png', f, show_unit_cell=False,
    #                bbox=bbox)
    # the bounding vox's 1st coordinate corresponds to the horizontal direction
    # and ASE's x direction

    # Load image with
    #
    #     ![title](png/traj_1ns_test.png?arg)
    #
    # and change to some random string after the question mark in order to enforce reloading when image changed on disk ([https://github.com/jupyter/notebook/issues/1369])
    # ![title](png/traj_1ns_test.png?thrsxstw)

    # #### Batch rendering

    bbox = [-20, 20, 80, 120 ]

    # make a movie
    # https://wiki.fysik.dtu.dk/ase/development/making_movies.html
    #for i,f in enumerate(frames_1ns_stripped[0::every_nth]):
    for i,g in enumerate(traj[::every_nth]):
        #f.rotate('-y', 'z', rotate_cell=True)
        f = g.copy()
        f.rotate(-60,'x', rotate_cell=False)
        f.rotate(30,'y', rotate_cell=False)
        #f.center()

        #cell = f.get_cell()
        #f = f.repeat((3, 1, 3))
        #f.set_cell(cell)
        ase.io.write(png_prefix + '_{:05d}.png'.format(i), f, show_unit_cell=False,
                    bbox=bbox)
    # bbox measures chosen to exceed cell once in x direction and twice in (repeated) y direction
    # externally execute
    #    ffmpeg -r 30 -f image2 -i "traj_1ns_%05d.png" -vcodec libx264 -crf 25 -pix_fmt yuv420p "traj_1ns.mp4


    os.chdir('png')
    # group_ws = get_ipython().run_line_magic('env', 'GROUP_WS')
    # get_ipython().system('find $group_ws -name ffmpeg')
    # get_ipython().run_line_magic('pwd', '')
    subprocess.run( '; '.join((
        "source '/work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/local_Nov17/env.sh'",
        "cd '{:s}/png'".format( absolute_prefix ),
        "ffmpeg -r 30 -f image2 -i 'traj_%05d.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p 'traj.mp4'")),
    shell=True, check=True)

    os.chdir('..')
    # get_ipython().run_cell_magic('bash', '', 'source \'/work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/local_Nov17/env.sh\'\ncd /work/ws/nemo/fr_jh1130-201708-0/jobs/lmplab/ctab/201807/1_CTAB_on_AU_21x12x2_netcdf/png\nffmpeg -r 30 -f image2 -i "traj_1ns_%05d.png" -vcodec libx264 -crf 25 -pix_fmt yuv420p "traj_1ns.mp4"')


# TODO:
#
# # ## 4. Radial distribution functions
#
# # ### 4.0. Index selections
#
# # In[138]:
#
#
# # helps in selecting atom nmbers based on element abbreviations
# chem_sym = np.array(ase.data.chemical_symbols)
#
#
# # In[139]:
#
#
# sulfur_index = [ a.number - 1 for a in surfactant if a.name == 'S' ][0]
#
#
# # In[140]:
#
#
# tail_carbon_index = [ a.number - 1 for a in surfactant if a.name == 'C12' ][0]
#
#
# # In[141]:
#
#
# sulfur_index
#
#
# # In[142]:
#
#
# tail_carbon_index
#
#
# # ### 4.1. Headgroup, Tailgroud - Gold RDF
#
# # In[143]:
#
#
# # Quoted from https://wiki.fysik.dtu.dk/asap/Radial%20Distribution%20Functions
# #
# # Partial RDFs: Looking at specific elements or subsets of atoms
# # It is often useful to look at partial RDFs, for example RDFs only taking some
# # elements into account, for example to get the distribution of atoms of element
# # B around an atom of element A. Do do this, call get_rdf() with the optional argument,
# # elements. It must be a tuple of two atomic numbers (a, b), the returned RDF then tells
# # how many b neighbors an a atom has.
# #
# # It is also possible to group atoms according to other criteria, for example to
# # calculate RDFs in different parts of space. In this case, the atoms must be divided
# # into groups when the RadialDistributionFunction object is created. Pass the extra
# # argument groups when creating the object, it must be an array containing a
# # non-negative integer for each atom specifying its group. When calling get_rdf()
# # use the argument group to specify for which group of atoms you want the RDF.
#
# # IMPORTANT: The partial RDFs are normalized such that they sum up to the global RDF.
# # This means that integrating the first peak of a partial RDF obtained with
# # elements=(a,b) does not give you the number of B atoms in the first shell around
# # the A atom. Instead it gives this coordination number multiplied with the concentration
# # of A atoms.
#
# traj = lmp_trajectrories['npt1ns']
#
# # pick indices to look at during rdf computation
# surfaceSoluteRdfIndices = [sulfur_index] +[tail_carbon_index] + surface_indices
#
# nSegment = 1000
#
# absoluteEnd = len(traj)-1
# rdfSulfurGoldList     = []
# rdfTailCarbonGoldList = []
#
# # rMax is the rdf length (in Angstrom for LAMMPS output in real units)
# rMax  = 20
#
# # nBins: can be understood as the number of data points on the RDF
# nBins = 1000
# # actual distances
# rdf_x = (np.arange(nBins) + 0.5) * rMax / nBins
#
# # instead of computing an average rdf over the whole trajectory,
# # we split the trajectory into several timespans of nSegement timestep length
# for nStart in range(0,absoluteEnd,nSegment):
#     print(nStart) # some progress report
#     surfaceSoluteRdf = None
#     for frame in traj[nStart:(nStart+nSegment)]:
#         # the asap rdf functionality is not that convenient, but explicitely choosing
#         # only the atoms we are interested in a priori, we can get exactly the rdf we want by
#         # making use of the "elements" option
#         if surfaceSoluteRdf is None:
#             surfaceSoluteRdf = RadialDistributionFunction(frame[surfaceSoluteRdfIndices],
#                                      rMax = rMax, nBins = nBins)
#         else:
#             surfaceSoluteRdf.atoms = frame[surfaceSoluteRdfIndices]  # Fool RDFobj to use the new atoms
#         surfaceSoluteRdf.update()           # Collect data
#
#     # np.where facilitates the selection of according atom numbers by specifying the chemical symbol
#     rdfSulfurGold = surfaceSoluteRdf.get_rdf(elements=(
#         np.where(chem_sym == 'S')[0][0],
#         np.where(chem_sym == 'Au')[0][0] ))
#     rdfTailCarbonGold = surfaceSoluteRdf.get_rdf(elements=(
#         np.where(chem_sym == 'C')[0][0],
#         np.where(chem_sym == 'Au')[0][0] ))
#
#     rdfSulfurGoldList.append(rdfSulfurGold)
#     rdfTailCarbonGoldList.append(rdfTailCarbonGold)
#
#
# # In[176]:
#
#
# # code snippet for neat plotting of all time-segemtn rdfs
# cols = 2
# rows = np.ceil(len(rdfSulfurGoldList)/cols).astype(int)
# pos = subplotPosition(rows,cols)
# fig = plt.figure(figsize=(rows*5,cols*8))
#
# for i, (rdfSulfurGold, rdfTailCarbonGold) in enumerate(zip(rdfSulfurGoldList,rdfTailCarbonGoldList)):
#     p = next(pos)
#     _, ax = addSubplot(rdf_x,rdfSulfurGold,
#                        legend = "S - Au RDF",
#                        xlabel = r'$\frac{r}{\AA}$',
#                        ylabel='arbitrary density',
#                        title = "{} ps - {} ps".format(i*nSegment,(i+1)*nSegment),
#                        fig = fig, pos = p)
#     _, _ = addSubplot(rdf_x, rdfTailCarbonGold,
#                       legend="S - tail C RDF",
#                       ax = ax, pos = p)
#
# fig.tight_layout()
#
#
# # ### 4.2. head group sulfur, tail group carbon - water RDF
#
# # In[177]:
#
#
# # Element tuples
# element_tuples = [
#     ( np.where(chem_sym == 'S')[0][0], np.where(chem_sym == 'O')[0][0] ),
#     ( np.where(chem_sym == 'C')[0][0], np.where(chem_sym == 'O')[0][0] ) ]
#
#
# # In[178]:
#
#
# element_tuples # in atomic numbers
#
#
# # In[179]:
#
#
# surfactantSolventIndicesOfInterest = water_indices + [ sulfur_index ] + [ tail_carbon_index ]
#
#
# # In[180]:
#
#
# surfactantSolventRDFs, surfactantSolventRDFx, surfactantSolventRDFobj = piecewiseRDF(
#     lmp_trajectrories['npt1ns'], surfactantSolventIndicesOfInterest, element_tuples)
#
#
# # In[181]:
#
#
# plotPiecewiceRdf(surfactantSolventRDFx, surfactantSolventRDFs,
#                  legend= [ "head group sulfur - water RDF", "tail group carbon - water RDF"]);
#
#
# # ### 4.3. Sodium counterion RDF
#
# # In[182]:
#
#
# # Element tuples
# counterionRdf_element_tuples = [
#     ( np.where(chem_sym == 'Na')[0][0], np.where(chem_sym == 'S')[0][0] ),
#     ( np.where(chem_sym == 'Na')[0][0], np.where(chem_sym == 'C')[0][0] ),
#     ( np.where(chem_sym == 'Na')[0][0], np.where(chem_sym == 'O')[0][0] ),
#     ( np.where(chem_sym == 'Na')[0][0], np.where(chem_sym == 'Au')[0][0] ) ]
#
#
# # In[183]:
#
#
# counterionRdf_element_tuples
#
#
# # In[184]:
#
#
# counterionRdfLabels = [ "Na+ counterion - SDS head sulfur RDF",
#                         "Na+ counterion - SDS tail carbon RDF",
#                         "Na+ counterion - water oxygen RDF",
#                         "Na+ counterion - surface gold RDF"]
#
#
# # In[185]:
#
#
# counterionRdfIndicesOfInterest = water_indices + surface_indices +     ion_indices + [ sulfur_index ] + [ tail_carbon_index ]
#
#
# # In[186]:
#
#
# counterionRDFs, counterionRDFx, counterionRDFobj = piecewiseRDF(
#     lmp_trajectrories['npt1ns'], counterionRdfIndicesOfInterest, counterionRdf_element_tuples)
#
#
# # In[187]:
#
#
# plotPiecewiceRdf(counterionRDFx, counterionRDFs, legend= counterionRdfLabels);
#
#
# # In[189]:
#
#
# plotPiecewiceRdf(counterionRDFx, counterionRDFs[0:2],
#                  legend= (counterionRdfLabels[0:2]));
#
#
# # In[190]:
#
#
# plotPiecewiceRdf(counterionRDFx, counterionRDFs[2:],
#                  legend= (counterionRdfLabels[2:]));
#
#
# # ## 5. Distance analysis
#
# # ### 5.1. Headgroup - gold distance
# # Head approaches surfaces, apparently "stepwise"
#
# # In[191]:
#
#
# traj = lmp_trajectrories['npt1ns']
#
#
# # In[192]:
#
#
# traj[0][surfactant_indices].get_atomic_numbers()
#
#
# # In[193]:
#
#
# averageDistanceS2Au, averageDistanceS2AuTimes = piecewiseAveragedDistance(traj,
#                                 reference_index=sulfur_index,
#                                 atom_indices=surface_indices,
#                                 nSegment=50)
#
#
# # In[194]:
#
#
# len(averageDistanceS2AuTimes)
#
#
# # In[195]:
#
#
# averageDistanceS2Au.shape
#
#
# # In[196]:
#
#
# distanceLabels = ['x', 'y', 'z']
#
#
# # In[197]:
#
#
# distanceFig = plt.figure()
# for i in range(0,3):
#     plt.plot( averageDistanceS2AuTimes*0.2, averageDistanceS2Au[i,:], label= distanceLabels[i] )
# plt.legend()
# plt.title("Distance of head group sulfur from gold atoms")
# plt.xlabel("time t / ps")
# plt.ylabel(r'Distance $\frac{r}{\AA}$')
#
#
# # In[198]:
#
#
# traj = lmp_trajectrories['npt1ns']
#
#
# # In[199]:
#
#
# traj[0][surfactant_indices].get_atomic_numbers()
#
#
# # In[201]:
#
#
# # slight decrease in potential in comparison with approach towards surface
# # confirms the anticipated energetically favored adsorption state
# nptProduction_1ns_thermo_pd[["PotEng","E_pair"]].rolling(window=5000,center=True).mean().plot()
#
#
# # In[202]:
#
#
# nptProduction_1ns_thermo_pd
#
#
# # In[203]:
#
#
# nptProduction_1ns_thermo_pd[["E_intramolecular"]].rolling(window=5000,center=True).mean().plot()
#
#
# # In[204]:
#
#
# ax = nptProduction_1ns_thermo_pd[["E_intramolecular"]].plot()
# nptProduction_1ns_thermo_pd[["E_intramolecular"]].rolling(window=5000,center=True).mean().plot(ax=ax)
#
#
# # In[205]:
#
#
# # running average, non-bonded energy
# ax = nptProduction_1ns_thermo_pd[["E_pair"]].plot()
# nptProduction_1ns_thermo_pd[["E_pair"]].rolling(window=10000,center=True).mean().plot(ax=ax)
#
#
# # In[206]:
#
#
# # running average, total energy
# ax = nptProduction_1ns_thermo_pd[["TotEng"]].plot()
# nptProduction_1ns_thermo_pd[["TotEng"]].rolling(window=10000,center=True).mean().plot(ax=ax)
#
#
# # In[207]:
#
#
# # running average, total energy
# ax = nptProduction_1ns_thermo_pd[["PotEng"]].plot()
# nptProduction_1ns_thermo_pd[["PotEng"]].rolling(window=10000,center=True).mean().plot(ax=ax)
#
#
# # ### 5.2. Tailgroup - gold distance
# # z - direction: does not change much
#
# # In[210]:
#
#
# averageDistanceTailC2Au, averageDistanceTailC2AuTimes = piecewiseAveragedDistance(traj,
#                                 reference_index=tail_carbon_index,
#                                 atom_indices=surface_indices,
#                                 nSegment=50)
#
#
# # In[211]:
#
#
# for i in range(0,3):
#     plt.plot( averageDistanceTailC2AuTimes*0.2, averageDistanceTailC2Au[i,:], label= distanceLabels[i] )
# plt.legend()
# plt.title("Distance of tail group from gold atoms")
# plt.xlabel("time t / ps")
# plt.ylabel(r'Distance $\frac{r}{\AA}$')
#
#
# # ### 5.3. Surface COM - Surfactant COM distance
#
# # In[212]:
#
#
# averageDistanceComCom, averageDistanceComComTimes = piecewiseAveragedComComDistance(traj,
#                                 surfactant_indices,surface_indices,
#                                 nSegment=50)
#
#
# # In[213]:
#
#
# for i in range(0,3):
#     plt.plot( averageDistanceComComTimes*0.2, averageDistanceComCom[i,:], label= distanceLabels[i] )
# plt.legend()
# plt.title("Distance of SDS center of mass from gold layer center of mass")
# plt.xlabel("time t / ps")
# plt.ylabel(r'Distance $\frac{r}{\AA}$')
#
#
# # In[215]:
#
#
# # for comparison: Headgroup - surface distance:
# distanceFig
#
#
# # ### SDS chain length
#
# # In[216]:
#
#
# averageChainLength, averageChainLengthTimes = piecewiseAveragedDistance(traj,
#                                 reference_index=tail_carbon_index,
#                                 atom_indices=[sulfur_index],
#                                 nSegment=50)
#
#
# # In[217]:
#
#
# np.linalg.norm(averageChainLength,axis=0)
#
#
# # In[218]:
#
#
# np.linalg.norm(averageChainLength,axis=0).shape
#
#
# # In[219]:
#
#
# plt.plot( averageChainLengthTimes*0.2,
#          np.linalg.norm(averageChainLength,axis=0), label="SDS Chain length")
# plt.legend()
# plt.title("Distance of SDS head group sulfur to tail group carbon")
# plt.xlabel("time t / ps")
# plt.ylabel(r'Distance $\frac{r}{\AA}$')
#
#
# # ## MSD and diffusivities
#
# # In[220]:
#
#
# dt
#
#
# # In[221]:
#
#
# T = 1e-9 # 1 ns
#
#
# # In[222]:
#
#
# Nf = len(traj) - 1 # number of stored frames, corresponds to 1ns
#
#
# # In[223]:
#
#
# Nf
#
#
# # In[224]:
#
#
# Ns = T/dt # number of steps
#
#
# # In[225]:
#
#
# Ns
#
#
# # In[226]:
#
#
# StepsPerFrame = Ns / Nf
#
#
# # In[227]:
#
#
# StepsPerFrame
#
#
# # In[228]:
#
#
# TimePerFrame = StepsPerFrame*dt
#
#
# # In[229]:
#
#
# TimePerFrame
#
#
# # In[235]:
#
#
# # displacements over fixed time spans, i.e. from each frame to the 50th or 5th following frame
#
#
# # In[230]:
#
#
# displacement10ps = comDisplacement(traj, surfactant_indices, dt=50)
#
#
# # In[231]:
#
#
# displacement1ps = comDisplacement(traj, surfactant_indices, dt=5)
#
#
# # In[232]:
#
#
# # instead of averaging over many trajectories, we average over N time-wise close frames-frame displacements
# # crude (all displacements within each set very correlated)
# # default N = 500 ~ 1 ps
# evaluateDisplacement(displacement10ps, dt=50);
# # however, the results give a good idea about anisotropic and decreasing mobility
# # during approach towards surface
#
#
# # In[233]:
#
#
# evaluateDisplacement(displacement1ps, dt=5);
#
#
# # In[234]:
#
#
# evaluateDisplacement(displacement1ps, dt=5, window=1000);
