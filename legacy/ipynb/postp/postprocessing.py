#!/usr/bin/env python
# coding: utf-8
# %%

# # 41 CTAB on AU 111 21x12x2 LAMMPS

# This notebook contains a set of code snippets and functions, supporting analysis of LAMMPS output.
# Exemplarily, the minimization, equilibration and production trajectories of one CTA+ ion in
# the vicinity of a 111 gold layer are investigated.
# 
# With the help of numpy, pandas and matplotlib, LAMMPS output energy conbtributions are evaluated and plotted.
# With the help of ase, parmed, nglview and ipywidgets, trajectories are visulized.
# 
# With the help of ase and asap, radial distribution functions, distances, displacements and diffusivities are evaluated.

# ## Header

# ### Jupyter Notebook-related "magic" commands

# %%


## preferred installation method for netcdf on NEMO locally:
# module load mpi/openmpi/2.1-gnu-5.2
# module load {...}
# export CPPFLAGS="${CPPFLAGS} -I${MPI_INC_DIR}"
# export LDFLAGS="${LDFLAGS} -L${MPI_LIB_DIR}"
# pip install --user netCDF4

## alternatively:
# pip install --user --global-option=build_ext --global-option="-L${MPI_INC_DIR}" netCDF4


# %%


get_ipython().run_line_magic('lsmagic', '# notebook commands')


# %%


# list environment variables, like bash's 'printenv'
get_ipython().run_line_magic('env', '')


# %%


get_ipython().run_line_magic('load_ext', 'memory_profiler')


# %%


# for some reason, nglview sometimes changes into some temporary directory
# therefore ALWAY use absolute filenames and paths
# %cd /work/ws/nemo/fr_jh1130-201708-0/jobs/lmplab/sds/201806/1_SDS_on_AU_100_1x4x4/


# ### Imports

# %%


# system basics
import os
absolute_prefix = os.getcwd() # might be handy to get back to the initial working directory at any point
os.sep # '\' ond windows and '/' on unix-like


# %%


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
import nglview as nv
import matplotlib.pyplot as plt
import ipywidgets # just for jupyter notebooks


# %%


import ipyparallel as ipp


# ### Global options

# %%


get_ipython().run_line_magic('matplotlib', 'inline')


# %%


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


# %%


# numpy truncates the output of large array above the treshold length
np.set_printoptions(threshold=100) 


# ### Definition of helper functions

# %%


# conversion units, only for better readability
fs = 1e-15 # s
ps = 1e-12 # s


# %%


AA = 1e-10 # m


# %%


def fullprint(*args, **kwargs):
    """prints a long numpy array without altering numpy treshold options permanently"""
    from pprint import pprint
    opt = np.get_printoptions()
    np.set_printoptions(edgeitems=3,infstr='inf', 
        linewidth=75, nanstr='nan', precision=8,
        suppress=False, threshold=100000, formatter=None)
    pprint(*args, **kwargs)
    np.set_printoptions(**opt)


# %%


def runningMeanFast(x, N):
    """a quick way to compute the running or rolling mean on a numpy array"""
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


# %%


def running_mean(x, N):
    """another quick way to compute the running or rolling mean on a numpy array"""
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# %%


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


# %%


def makeThermoPlotsFromDataFrame(df, fig=None,
        time_label          = r'$\frac{\mathrm{Steps}}{2 \mathrm{fs}}$',
        temperature_label   = r'$\frac{T}{\mathrm{K}}$',
        pressure_label      = r'$\frac{P}{\mathrm{atm}}$',
        energy_label        = r'$\frac{E}{\mathrm{Kcal} \cdot \mathrm{mole}^{-1}}$'):
    """Automizes the plotting of thermo output."""

    rows = 3
    cols = 2
    if fig == None:
        fig = plt.figure(figsize=(cols*8,rows*5))
    
    def subplotPosition(rows,cols):
        for p in range(0,rows*cols):
            yield rows*100+cols*10+p+1
     
    def addSubplot(df,fig,pos,title,xlabel,ylabel):
        ax = fig.add_subplot(pos)
        df.plot(ax=ax) # taimed temperature
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    pos = subplotPosition(rows,cols)
    
    # sum up intramolecular contributions 
    df["E_intramolecular"] = df[["E_bond","E_angle","E_dihed"]].sum(axis=1)
    
    addSubplot(df[["Temp"]],
               fig, next(pos), "Temperature", time_label, temperature_label)
    addSubplot(df[["Press"]],
               fig, next(pos), "Pressure", time_label, pressure_label)
   
    # intramolecular contributions (without angle)
    addSubplot(df[["E_intramolecular","E_bond","E_angle","E_dihed"]],
               fig, next(pos), "Intramolecular energies", time_label, energy_label)
    # intermolecular ("non-bonded") energy contribtutions
    # E_pair is the sum of the three latter, just as E_intramolecular in the plot above
    addSubplot(df[["E_pair","E_vdwl","E_coul","E_long"]],
                fig, next(pos), "Intermolecular (non-bonded) energies", time_label, energy_label)
    
    # visualize the difference between total and non-bonded potential:
    addSubplot(df[["PotEng","E_pair"]],
               fig, next(pos), "Total potential and non-bonded potential", time_label, energy_label)
    
    addSubplot(df[["TotEng","KinEng","PotEng"]],
               fig, next(pos), "Total, kinetic and potential energies", time_label, energy_label)
    
    fig.tight_layout()
    return fig


# %%


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


# %%


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


# %%


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


# %%


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


# %%


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


# %%


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


# %%


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


# %%


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
    _, ax = addSubplot(time[:-window+1], running_mean( isotropic_displacement,window), 
            title = "{:.2f} ps displacement".format(dt*TimePerFrame/ps), 
            xlabel = "time t / ps", ylabel= r'displacement $\frac{r}{\AA}$',
            legend = 'isotropic', fig = fig, pos = p)

    for i in range(0,3):
        addSubplot(time[:-window+1], 
                   running_mean(displacement[i,:], window), 
                   legend = distanceLabels[i], ax = ax, pos = p)
        
    p = next(pos)
    _, ax = addSubplot(time[:-window+1], running_mean( isotropic_displacement**2 / 3.0, window), 
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
    _, ax = addSubplot(time[:-window+1], running_mean( Diso, window), 
            title = "diffusivities from {:.2f} ps MSD".format(dt*TimePerFrame/ps), 
            xlabel = "time t / ps", ylabel= r'D $\frac{m^2}{s}$',
            legend = 'isotropic', fig = fig, pos = p)

    
    for i in range(0,3):
        addSubplot(time[:-window+1], 
                   running_mean(D[i,:], window), 
                   legend = distanceLabels[i], ax = ax, pos = p)
        
    fig.tight_layout()
    
    return fig


# ## Energy evaluations with pandas

# %%


dt = 2e-15 # s, 2 fs timestep


# ### Minimization

# %%


get_ipython().run_line_magic('ls', '-lh *minimization.log')


# %%


get_ipython().system('./extract_thermo.sh *minimization.log 01_minimization_thermo.out')


# %%


minimization_thermo_file = absolute_prefix + os.sep + '01_minimization_thermo.out'


# %%


minimization_thermo_pd = pd.read_csv(minimization_thermo_file,delim_whitespace=True)


# %%


minimization_thermo_pd.set_index("Step",inplace=True)


# %%


minimization_thermo_pd


# %%


makeThermoPlotsFromDataFrame(minimization_thermo_pd);


# %%


minimization_thermo_pd[["E_long"]][2:].plot()


# %%


# The total energy decreases, but intramolecular energy increases during minimization:
minimization_thermo_pd[["PotEng","E_pair"]][2:].plot()


# %%


# double-check: total potential energy of system minus non-bonded energy (LJ & Coulomb) 
# should correspond to intramolecular energy:
intramolecularEnergyValidation = minimization_thermo_pd["PotEng"] - minimization_thermo_pd["E_pair"]


# %%


intramolecularEnergyValidationDiff = (intramolecularEnergyValidation - minimization_thermo_pd["E_intramolecular"])


# %%


intramolecularEnergyValidationDiff.max()


# %%


intramolecularEnergyValidationDiff.abs().max() / intramolecularEnergyValidation.min()


# %%


(intramolecularEnergyValidation - minimization_thermo_pd["E_intramolecular"])[1:].plot()
# obviously "equal" (up to a tiny fraction)


# %%


# descent to steep t the first few steps, excluded
makeThermoPlotsFromDataFrame(minimization_thermo_pd.iloc[2:].copy());


# ### NVT equilibration

# %%


get_ipython().run_line_magic('ls', '*nvtEquilibration.log')


# %%


get_ipython().system('./extract_thermo.sh *nvtEquilibration.log 02_nvtEquilibration_thermo.out')


# %%


nvtEquilibration_thermo_file = absolute_prefix + os.sep + '02_nvtEquilibration_thermo.out'


# %%


get_ipython().system('cat $nvtEquilibration_thermo_file | wc -l # count the lines in file, simple way for verification')
# at 2500 steps and output every 10 steps ~ 250 steps, 251 frames, 1 header line


# %%


nvtEquilibration_thermo_pd = pd.read_csv(nvtEquilibration_thermo_file,delim_whitespace=True)


# %%


nvtEquilibration_thermo_pd.set_index("Step",inplace=True)


# %%


nvtEquilibration_thermo_pd


# %%


makeThermoPlotsFromDataFrame(nvtEquilibration_thermo_pd);


# ### NPT equilibration

# %%


get_ipython().run_line_magic('ls', '*nptEquilibration.log')


# %%


get_ipython().system('./extract_thermo.sh *nptEquilibration.log 03_nptEquilibration_thermo.out')


# %%


nptEquilibration_thermo_file = absolute_prefix + os.sep + '03_nptEquilibration_thermo.out'


# %%


nptEquilibration_thermo_pd = pd.read_csv(nptEquilibration_thermo_file,delim_whitespace=True)


# %%


nptEquilibration_thermo_pd.set_index("Step",inplace=True)


# %%


nptEquilibration_thermo_pd


# %%


nptEquilibration_thermo_consecutive_pd = nptEquilibration_thermo_pd.copy()


# %%


nvtEquilibration_thermo_pd.index.max()


# %%


nptEquilibration_thermo_consecutive_pd.index = nptEquilibration_thermo_pd.index + nvtEquilibration_thermo_pd.index.max()


# %%


equilibration_pd = pd.concat([nvtEquilibration_thermo_pd,nptEquilibration_thermo_consecutive_pd])


# %%


equilibration_pd


# %%


makeThermoPlotsFromDataFrame(nptEquilibration_thermo_pd);


# %%


# show nvt and npt equilibration consecutively
makeThermoPlotsFromDataFrame(equilibration_pd);


# %%


makeRollingAverageThermoPlotsFromDataFrame(equilibration_pd,window=10);


# ### 1 ns NPT

# %%


get_ipython().run_line_magic('ls', '-lh *_1ns_npt_01_nptProduction.log')


# %%


get_ipython().system('./extract_thermo.sh *_1ns_npt_01_nptProduction.log 05_1ns_nptProduction_thermo.out')


# %%


nptProduction_1ns_thermo_file = absolute_prefix + os.sep + '05_1ns_nptProduction_thermo.out'
nptProduction_1ns_thermo_pd = pd.read_csv(nptProduction_1ns_thermo_file,delim_whitespace=True)
nptProduction_1ns_thermo_pd.set_index("Step",inplace=True)


# %%


makeThermoPlotsFromDataFrame(nptProduction_1ns_thermo_pd.iloc[::100].copy()); # only every 100th data point


# %%


makeRollingAverageThermoPlotsFromDataFrame(nptProduction_1ns_thermo_pd,window=5000);


# %%


makeRollingAverageThermoPlotsFromDataFrame(nptProduction_1ns_thermo_pd,window=25000);


# %%


nptProduction_1ns_thermo_pd[["PotEng","E_pair"]].rolling(window=5000,center=True).mean().plot()


# ### 1 ns NVE

# %%


get_ipython().run_line_magic('ls', '-lh *_1ns_nve_01_nveProduction.log')


# %%


get_ipython().system('./extract_thermo.sh *_1ns_nve_01_nveProduction.log 05_1ns_nveProduction_thermo.out')


# %%


nveProduction_1ns_thermo_file = absolute_prefix + os.sep + '05_1ns_nveProduction_thermo.out'
nveProduction_1ns_thermo_pd = pd.read_csv(nveProduction_1ns_thermo_file,delim_whitespace=True)
nveProduction_1ns_thermo_pd.set_index("Step",inplace=True)


# %%


makeThermoPlotsFromDataFrame(nveProduction_1ns_thermo_pd.iloc[::100].copy()); # only every 100th data point


# %%


makeRollingAverageThermoPlotsFromDataFrame(nveProduction_1ns_thermo_pd,window=5000);


# %%


makeRollingAverageThermoPlotsFromDataFrame(nveProduction_1ns_thermo_pd,window=25000);


# %%


nveProduction_1ns_thermo_pd[["PotEng","E_pair"]].rolling(window=5000,center=True).mean().plot()


# %%


ax = nveProduction_1ns_thermo_pd[["TotEng"]].rolling(window=5000,center=True).mean().plot()
nptProduction_1ns_thermo_pd[["TotEng"]].rolling(window=5000,center=True).mean().plot(ax=ax)


# %%


nveProduction_1ns_thermo_pd[["TotEng"]].rolling(window=5000,center=True).mean().plot()


# %%


nptProduction_1ns_thermo_pd[["TotEng"]].rolling(window=5000,center=True).mean().plot()


# ## Trajectory visualization with ASE and ParmEd

# %%


get_ipython().run_line_magic('ls', '*.lammps')


# %%


get_ipython().run_line_magic('ls', '-lh *.nc')


# %%


# File names
prefix = '1_CTAB_on_AU_111_21x12x2_'

pdb_file_initial_config = absolute_prefix + os.sep + prefix + 'psfgen_ctrl.pdb'

lmp_files = { 'initial':         prefix + 'psfgen_CLA2BR.data',
              'minimized':       prefix + 'minimized.lammps',
              'nvtEquilibrated': prefix + 'nvtEquilibrated.lammps',
              'nptEquilibrated': prefix + 'nptEquilibrated.lammps',
              #'npt10ps':         prefix + '10ps_npt_final_config.lammps',
              #'npt100ps':        prefix + '100ps_npt_final_config.lammps',
              'npt1ns':          prefix + '1ns_npt_final_config.lammps',
              'nve1ns':          prefix + '1ns_nve_final_config.lammps'
            }

# lmp_dumps = { 
#              'nvtEquilibration': prefix + 'nvtEquilibration.dump',
#              #'nptEquilibration': prefix + 'nptEquilibration.dump',
#              #'npt10ps':          prefix + '10ps_npt_nptProduction.dump',
#              #'npt100ps':         prefix + '100ps_npt_nptProduction.dump',
#              #'npt1ns':           prefix + '1ns_npt_with_restarts_nptProduction.dump',
#              #'nve1ns':           prefix + '1ns_nve_with_restarts_nveProduction.dump' 
#             }
lmp_netcdf = { 
              'nvtEquilibration': prefix + 'nvtEquilibration.nc',
              'nptEquilibration': prefix + 'nptEquilibration.nc',
              'npt1ns':           prefix + '1ns_npt_nptProduction.nc',
              'nve1ns':           prefix + '1ns_nve_nveProduction.nc' 
             }


# %%


# construct a dictionary-like atom type-> atom number array
t2n, t2n_array = inferTypes2NumbersFromPdbAndLmp(pdb_file_initial_config, lmp_files['initial'])


# %%


t2n


# %%


t2n_array


# %%


np.array(ase.data.chemical_symbols)[t2n_array] # double-check against LAMMPS data file


# ### Initial configuration from .pdb

# %%


# create atom selections for later post-processing:
struct_pdb_pmd     = pmd.read_PDB(pdb_file_initial_config)
water              = [ a for a in struct_pdb_pmd.atoms if a.residue.name == 'HOH' ] # water
water_indices      = [ a.number - 1 for a in water ] # to remove water atoms later
surface            = [ a for a in struct_pdb_pmd.atoms if a.residue.name == 'AUM' ] # gold surface
surface_indices    = [ a.number - 1 for a in surface ] 
surfactant         = [ a for a in struct_pdb_pmd.atoms if a.residue.name == 'CTA' ]
surfactant_indices = [ a.number - 1 for a in surfactant ] 
ions               = [ a for a in struct_pdb_pmd.atoms if a.residue.name == 'ION' ] # sodium counterions
ion_indices        = [ a.number - 1 for a in ions ] 


# %%


struct_pdb_pmd


# %%


surface_indices[0] # double check index begins at 0


# %%


len(surfactant_indices)


# %%


ions


# %%


ions[0].atomic_number # wrong: inferred C (carbon)


# %%


nv.show_parmed(struct_pdb_pmd)


# ### LAMMPS data files

# %%


# read frames of interest
lmp_frames = {}
for k,f in lmp_files.items():
    lmp_frames[k] = read(lmp_files[k],format='lammps-data')
    lmp_frames[k].set_atomic_numbers(
        t2n_array[lmp_frames[k].get_atomic_numbers() ] )


# %%


lmp_frames['minimized']


# %%


# lmp_views = []
# for k, f in lmp_frames.items():
#     lmp_views.append( nv.show_ase(f) )
#     lmp_views[-1]._set_sync_camera()
#     lmp_views[-1]._remote_call("setSize", target="Widget", args=["250px", "250px"])
#     lmp_views[-1].center()
#     lmp_views[-1].render_image()


# %%


# vbox = ipywidgets.VBox(lmp_views)


# %%


# vbox


# %%


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


# %%


lmp_naked_frames["nvtEquilibrated"]


# %%


lmp_frames["nvtEquilibrated"]


# %%


lmp_views = []
for k, f in lmp_naked_frames.items():
    lmp_views.append( nv.show_ase(f) )
    lmp_views[-1]._set_sync_camera()
    lmp_views[-1]._remote_call("setSize", target="Widget", args=["250px", "250px"])
    lmp_views[-1].center()
    lmp_views[-1].render_image()


# %%


vbox = ipywidgets.VBox(lmp_views)


# %%


vbox


# ### LAMMPS NETCDF trajectories

# %%


# lmp_trajectrories = {}
# for k,t in lmp_dumps.items():
#     lmp_trajectrories[k] = read(t, index=':',format='lammps-dump')
#     for f in lmp_trajectrories[k]:      
#         f.set_atomic_numbers(
#             t2n_array[f.get_atomic_numbers() ] )
#         f.center()


# %%


# lmp_netcdf['nvtEquilibration']


# %%


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


# %%


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
get_ipython().run_line_magic('memit', '')


# %%


get_ipython().run_line_magic('memit', '')


# %%


len(lmp_trajectrories['nvtEquilibration'])


# %%


view(lmp_trajectrories['nvtEquilibration'], viewer='ase') # opens ASE GUI


# %%


#nv.show_asetraj(lmp_trajectrories['npt100ps'])


# %%


# Several problems with nglview:
#  1) does not display or infer bonding for ASE trajectory
#  2) even with displayed gui, not clear how to activate


# %%


lmp_trajectrories['npt1ns'][0]


# ### LAMMPS trajectories, stripped of solvent

# %%


get_ipython().run_line_magic('memit', '')
# apparently, order of atoms changed in netcdf
lmp_naked_trajectrories = {}
for k in lmp_trajectrories:
    lmp_naked_trajectrories[k] = []
    for f in lmp_trajectrories[k]:
        #g = f.copy()
        #del g[water_indices]
        g = f[ (np.array(f.get_chemical_symbols()) != 'O') & (np.array(f.get_chemical_symbols()) != 'H')]
        lmp_naked_trajectrories[k].append(g)


# %%


trajectoryView = nv.show_asetraj(lmp_naked_trajectrories['npt1ns'][::100])


# %%


#trajectoryView = nv.show_asetraj(lmp_naked_trajectrories['nvtEquilibration'][::100])


# %%


trajectoryView.remove_ball_and_stick()


# %%


trajectoryView.add_spacefill() # try a different representation sytle


# %%


trajectoryView


# %%


view(lmp_naked_trajectrories['npt1ns'], viewer='ase') # opens ASE GUI


# %%


view(lmp_naked_trajectrories['nvtEquilibration'], viewer='ngl') # opens ASE GUI


# %%


# Several problems with nglview:
#  1) does not display or infer bonding for ASE trajectory
#  2) even with displayed gui, not clear how to activate
#  3) somehow mixes up the atom type for netcdf


# ### Make a movie via .png frames and ffmpeg

# #### Preparation 

# %%


# create a subdir from within the notebook
get_ipython().run_line_magic('mkdir', 'png')


# %%


traj = lmp_naked_trajectrories['npt1ns']


# %%


nv.show_ase(traj[200])


# %%


# from ~ frame 3500 to the end ~ means


# %%


totalFramesAvailable = 5000


# %%


desiredVideoDuration = 30 # s


# %%


framesPerSecond = 30 # s^-1


# %%


neededFrames = desiredVideoDuration*framesPerSecond


# %%


neededFrames


# %%


every_nth = np.ceil(totalFramesAvailable / neededFrames).astype(int)


# %%


every_nth


# %%


png_prefix = absolute_prefix + os.sep + 'png' + os.sep + 'traj_1ns'


# #### Orientation and bounding box settings by trial & error

# %%


f = traj[0].copy()


# %%


# find a desired orientation
#f.rotate('x', (-1,2,-1), rotate_cell=True)
#testframe.rotate(90, (1,0,0), rotate_cell=False)
#f.rotate('z', (1,1,-1), rotate_cell=False)
#f.rotate(-90,'x', rotate_cell=False)
# strange, somehow opposite behavior with "in-notebook" viewer and png renderer


# %%


f.rotate(-60,'x', rotate_cell=False)


# %%


f.rotate(30,'y', rotate_cell=False)


# %%


view(f,viewer='ngl')


# %%


nv.show_ase(f)


# %%


cell = f.get_cell()


# %%


cell


# %%


bbox = [-20, 20, 80, 120 ]


# %%


# one trial
# the commented lines can be used to replicate the cell
#cell = f.get_cell()
#f = f.repeat((3, 1, 3))
#f.set_cell(cell)
# bbox = [-5, -5, cell[0,0] + 5, cell[1,1] + 5 ]
#f.center()
ase.io.write(png_prefix + '_test.png', f, show_unit_cell=False, 
                bbox=bbox)
# the bounding vox's 1st coordinate corresponds to the horizontal direction
# and ASE's x direction


# Load image with
# 
#     ![title](png/traj_1ns_test.png?arg)
#     
# and change to some random string after the question mark in order to enforce reloading when image changed on disk ([https://github.com/jupyter/notebook/issues/1369])
# ![title](png/traj_1ns_test.png?thrsxstw)

# #### Batch rendering

# %%


bbox = [-20, 20, 80, 120 ]


# %%


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


# %%


# operating some bash commands from within jupyter nb
get_ipython().run_line_magic('cd', 'png')


# %%


group_ws = get_ipython().run_line_magic('env', 'GROUP_WS')


# %%


group_ws


# %%


get_ipython().system('find $group_ws -name ffmpeg')


# %%


get_ipython().run_line_magic('pwd', '')


# %%


get_ipython().run_cell_magic('bash', '', 'source \'/work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/local_Nov17/env.sh\'\ncd /work/ws/nemo/fr_jh1130-201708-0/jobs/lmplab/ctab/201807/1_CTAB_on_AU_21x12x2_netcdf/png\nffmpeg -r 30 -f image2 -i "traj_1ns_%05d.png" -vcodec libx264 -crf 25 -pix_fmt yuv420p "traj_1ns.mp4"')


# ## 4. Radial distribution functions

# ### 4.0. Index selections

# %%


# helps in selecting atom nmbers based on element abbreviations
chem_sym = np.array(ase.data.chemical_symbols)


# %%


sulfur_index = [ a.number - 1 for a in surfactant if a.name == 'S' ][0]


# %%


tail_carbon_index = [ a.number - 1 for a in surfactant if a.name == 'C12' ][0]


# %%


sulfur_index


# %%


tail_carbon_index


# ### 4.1. Headgroup, Tailgroud - Gold RDF

# %%


# Quoted from https://wiki.fysik.dtu.dk/asap/Radial%20Distribution%20Functions
#
# Partial RDFs: Looking at specific elements or subsets of atoms
# It is often useful to look at partial RDFs, for example RDFs only taking some 
# elements into account, for example to get the distribution of atoms of element 
# B around an atom of element A. Do do this, call get_rdf() with the optional argument, 
# elements. It must be a tuple of two atomic numbers (a, b), the returned RDF then tells 
# how many b neighbors an a atom has.
#
# It is also possible to group atoms according to other criteria, for example to 
# calculate RDFs in different parts of space. In this case, the atoms must be divided 
# into groups when the RadialDistributionFunction object is created. Pass the extra 
# argument groups when creating the object, it must be an array containing a 
# non-negative integer for each atom specifying its group. When calling get_rdf() 
# use the argument group to specify for which group of atoms you want the RDF.

# IMPORTANT: The partial RDFs are normalized such that they sum up to the global RDF. 
# This means that integrating the first peak of a partial RDF obtained with 
# elements=(a,b) does not give you the number of B atoms in the first shell around 
# the A atom. Instead it gives this coordination number multiplied with the concentration 
# of A atoms.

traj = lmp_trajectrories['npt1ns']

# pick indices to look at during rdf computation
surfaceSoluteRdfIndices = [sulfur_index] +[tail_carbon_index] + surface_indices

nSegment = 1000

absoluteEnd = len(traj)-1
rdfSulfurGoldList     = []
rdfTailCarbonGoldList = []

# rMax is the rdf length (in Angstrom for LAMMPS output in real units)
rMax  = 20

# nBins: can be understood as the number of data points on the RDF
nBins = 1000
# actual distances
rdf_x = (np.arange(nBins) + 0.5) * rMax / nBins

# instead of computing an average rdf over the whole trajectory, 
# we split the trajectory into several timespans of nSegement timestep length
for nStart in range(0,absoluteEnd,nSegment):
    print(nStart) # some progress report
    surfaceSoluteRdf = None
    for frame in traj[nStart:(nStart+nSegment)]:
        # the asap rdf functionality is not that convenient, but explicitely choosing
        # only the atoms we are interested in a priori, we can get exactly the rdf we want by 
        # making use of the "elements" option
        if surfaceSoluteRdf is None:
            surfaceSoluteRdf = RadialDistributionFunction(frame[surfaceSoluteRdfIndices],
                                     rMax = rMax, nBins = nBins)
        else:
            surfaceSoluteRdf.atoms = frame[surfaceSoluteRdfIndices]  # Fool RDFobj to use the new atoms
        surfaceSoluteRdf.update()           # Collect data
    
    # np.where facilitates the selection of according atom numbers by specifying the chemical symbol
    rdfSulfurGold = surfaceSoluteRdf.get_rdf(elements=(
        np.where(chem_sym == 'S')[0][0],
        np.where(chem_sym == 'Au')[0][0] ))
    rdfTailCarbonGold = surfaceSoluteRdf.get_rdf(elements=(
        np.where(chem_sym == 'C')[0][0],
        np.where(chem_sym == 'Au')[0][0] ))
    
    rdfSulfurGoldList.append(rdfSulfurGold)
    rdfTailCarbonGoldList.append(rdfTailCarbonGold)


# %%


# code snippet for neat plotting of all time-segemtn rdfs
cols = 2
rows = np.ceil(len(rdfSulfurGoldList)/cols).astype(int)
pos = subplotPosition(rows,cols)
fig = plt.figure(figsize=(rows*5,cols*8))

for i, (rdfSulfurGold, rdfTailCarbonGold) in enumerate(zip(rdfSulfurGoldList,rdfTailCarbonGoldList)):
    p = next(pos)
    _, ax = addSubplot(rdf_x,rdfSulfurGold, 
                       legend = "S - Au RDF", 
                       xlabel = r'$\frac{r}{\AA}$', 
                       ylabel='arbitrary density', 
                       title = "{} ps - {} ps".format(i*nSegment,(i+1)*nSegment),
                       fig = fig, pos = p)
    _, _ = addSubplot(rdf_x, rdfTailCarbonGold, 
                      legend="S - tail C RDF",
                      ax = ax, pos = p)
    
fig.tight_layout()


# ### 4.2. head group sulfur, tail group carbon - water RDF

# %%


# Element tuples
element_tuples = [
    ( np.where(chem_sym == 'S')[0][0], np.where(chem_sym == 'O')[0][0] ),
    ( np.where(chem_sym == 'C')[0][0], np.where(chem_sym == 'O')[0][0] ) ]


# %%


element_tuples # in atomic numbers


# %%


surfactantSolventIndicesOfInterest = water_indices + [ sulfur_index ] + [ tail_carbon_index ]


# %%


surfactantSolventRDFs, surfactantSolventRDFx, surfactantSolventRDFobj = piecewiseRDF(
    lmp_trajectrories['npt1ns'], surfactantSolventIndicesOfInterest, element_tuples)


# %%


plotPiecewiceRdf(surfactantSolventRDFx, surfactantSolventRDFs, 
                 legend= [ "head group sulfur - water RDF", "tail group carbon - water RDF"]);


# ### 4.3. Sodium counterion RDF

# %%


# Element tuples
counterionRdf_element_tuples = [
    ( np.where(chem_sym == 'Na')[0][0], np.where(chem_sym == 'S')[0][0] ),
    ( np.where(chem_sym == 'Na')[0][0], np.where(chem_sym == 'C')[0][0] ),
    ( np.where(chem_sym == 'Na')[0][0], np.where(chem_sym == 'O')[0][0] ),
    ( np.where(chem_sym == 'Na')[0][0], np.where(chem_sym == 'Au')[0][0] ) ]


# %%


counterionRdf_element_tuples


# %%


counterionRdfLabels = [ "Na+ counterion - SDS head sulfur RDF", 
                        "Na+ counterion - SDS tail carbon RDF",
                        "Na+ counterion - water oxygen RDF",
                        "Na+ counterion - surface gold RDF"]


# %%


counterionRdfIndicesOfInterest = water_indices + surface_indices +     ion_indices + [ sulfur_index ] + [ tail_carbon_index ]


# %%


counterionRDFs, counterionRDFx, counterionRDFobj = piecewiseRDF(
    lmp_trajectrories['npt1ns'], counterionRdfIndicesOfInterest, counterionRdf_element_tuples)


# %%


plotPiecewiceRdf(counterionRDFx, counterionRDFs, legend= counterionRdfLabels);


# %%


plotPiecewiceRdf(counterionRDFx, counterionRDFs[0:2], 
                 legend= (counterionRdfLabels[0:2]));


# %%


plotPiecewiceRdf(counterionRDFx, counterionRDFs[2:], 
                 legend= (counterionRdfLabels[2:]));


# ## 5. Distance analysis

# ### 5.1. Headgroup - gold distance
# Head approaches surfaces, apparently "stepwise"

# %%


traj = lmp_trajectrories['npt1ns']


# %%


traj[0][surfactant_indices].get_atomic_numbers()


# %%


averageDistanceS2Au, averageDistanceS2AuTimes = piecewiseAveragedDistance(traj,
                                reference_index=sulfur_index,
                                atom_indices=surface_indices,
                                nSegment=50)


# %%


len(averageDistanceS2AuTimes)


# %%


averageDistanceS2Au.shape


# %%


distanceLabels = ['x', 'y', 'z']


# %%


distanceFig = plt.figure()
for i in range(0,3):
    plt.plot( averageDistanceS2AuTimes*0.2, averageDistanceS2Au[i,:], label= distanceLabels[i] )
plt.legend()
plt.title("Distance of head group sulfur from gold atoms")
plt.xlabel("time t / ps")
plt.ylabel(r'Distance $\frac{r}{\AA}$')


# %%


traj = lmp_trajectrories['npt1ns']


# %%


traj[0][surfactant_indices].get_atomic_numbers()


# %%


# slight decrease in potential in comparison with approach towards surface
# confirms the anticipated energetically favored adsorption state
nptProduction_1ns_thermo_pd[["PotEng","E_pair"]].rolling(window=5000,center=True).mean().plot()


# %%


nptProduction_1ns_thermo_pd


# %%


nptProduction_1ns_thermo_pd[["E_intramolecular"]].rolling(window=5000,center=True).mean().plot()


# %%


ax = nptProduction_1ns_thermo_pd[["E_intramolecular"]].plot()
nptProduction_1ns_thermo_pd[["E_intramolecular"]].rolling(window=5000,center=True).mean().plot(ax=ax)


# %%


# running average, non-bonded energy
ax = nptProduction_1ns_thermo_pd[["E_pair"]].plot()
nptProduction_1ns_thermo_pd[["E_pair"]].rolling(window=10000,center=True).mean().plot(ax=ax)


# %%


# running average, total energy
ax = nptProduction_1ns_thermo_pd[["TotEng"]].plot()
nptProduction_1ns_thermo_pd[["TotEng"]].rolling(window=10000,center=True).mean().plot(ax=ax)


# %%


# running average, total energy
ax = nptProduction_1ns_thermo_pd[["PotEng"]].plot()
nptProduction_1ns_thermo_pd[["PotEng"]].rolling(window=10000,center=True).mean().plot(ax=ax)


# ### 5.2. Tailgroup - gold distance
# z - direction: does not change much

# %%


averageDistanceTailC2Au, averageDistanceTailC2AuTimes = piecewiseAveragedDistance(traj,
                                reference_index=tail_carbon_index,
                                atom_indices=surface_indices,
                                nSegment=50)


# %%


for i in range(0,3):
    plt.plot( averageDistanceTailC2AuTimes*0.2, averageDistanceTailC2Au[i,:], label= distanceLabels[i] )
plt.legend()
plt.title("Distance of tail group from gold atoms")
plt.xlabel("time t / ps")
plt.ylabel(r'Distance $\frac{r}{\AA}$')


# ### 5.3. Surface COM - Surfactant COM distance

# %%


averageDistanceComCom, averageDistanceComComTimes = piecewiseAveragedComComDistance(traj,
                                surfactant_indices,surface_indices,
                                nSegment=50)


# %%


for i in range(0,3):
    plt.plot( averageDistanceComComTimes*0.2, averageDistanceComCom[i,:], label= distanceLabels[i] )
plt.legend()
plt.title("Distance of SDS center of mass from gold layer center of mass")
plt.xlabel("time t / ps")
plt.ylabel(r'Distance $\frac{r}{\AA}$')


# %%


# for comparison: Headgroup - surface distance: 
distanceFig


# ### SDS chain length

# %%


averageChainLength, averageChainLengthTimes = piecewiseAveragedDistance(traj,
                                reference_index=tail_carbon_index,
                                atom_indices=[sulfur_index],
                                nSegment=50)


# %%


np.linalg.norm(averageChainLength,axis=0)


# %%


np.linalg.norm(averageChainLength,axis=0).shape


# %%


plt.plot( averageChainLengthTimes*0.2, 
         np.linalg.norm(averageChainLength,axis=0), label="SDS Chain length")
plt.legend()
plt.title("Distance of SDS head group sulfur to tail group carbon")
plt.xlabel("time t / ps")
plt.ylabel(r'Distance $\frac{r}{\AA}$')


# ## MSD and diffusivities

# %%


dt


# %%


T = 1e-9 # 1 ns


# %%


Nf = len(traj) - 1 # number of stored frames, corresponds to 1ns


# %%


Nf


# %%


Ns = T/dt # number of steps


# %%


Ns


# %%


StepsPerFrame = Ns / Nf


# %%


StepsPerFrame


# %%


TimePerFrame = StepsPerFrame*dt


# %%


TimePerFrame


# %%


# displacements over fixed time spans, i.e. from each frame to the 50th or 5th following frame


# %%


displacement10ps = comDisplacement(traj, surfactant_indices, dt=50)


# %%


displacement1ps = comDisplacement(traj, surfactant_indices, dt=5)


# %%


# instead of averaging over many trajectories, we average over N time-wise close frames-frame displacements
# crude (all displacements within each set very correlated)
# default N = 500 ~ 1 ps
evaluateDisplacement(displacement10ps, dt=50);
# however, the results give a good idea about anisotropic and decreasing mobility 
# during approach towards surface


# %%


evaluateDisplacement(displacement1ps, dt=5);


# %%


evaluateDisplacement(displacement1ps, dt=5, window=1000);

