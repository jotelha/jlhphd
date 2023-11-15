#!/usr/bin/env python
# coding: utf-8

# # AU 111 150 Ang cube gold substrate evaluation

# ## Initialization

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import datetime, io, os, shutil, sys, tarfile
import os.path
from glob import glob
from pprint import pprint
from tabulate import tabulate
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
import ase, ase.io
from ase.io import read, write
from ase.io import NetCDFTrajectory
from asap3.analysis.rdf import RadialDistributionFunction


# In[ ]:


import matplotlib.pyplot as plt
from ase.visualize import view
from ase.visualize.plot import plot_atoms
import nglview as nv
import ipywidgets # just for jupyter notebooks
from IPython.display import Image


# In[ ]:


from fireworks import Firework, Workflow
from fireworks import LaunchPad
from fireworks.utilities.filepad import FilePad 
from fireworks.utilities.wfb import WorkflowBuilder

fp = FilePad(
   host="localhost",
   port=27018,
   database="fireworks-jhoermann",
   username="fireworks",
   password="fireworks")


# In[270]:


from sympy.ntheory.factor_ import factorint


# In[ ]:


# matplotlib settings

# expecially for presentation, larger font settings for plotting are recommendable
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 26

plt.rc("font", size=MEDIUM_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams["figure.figsize"] = (8,5) # the standard figure size

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 14
plt.rcParams["lines.markeredgewidth"]=1


# ## Definitions

# In[272]:


def read_rdf(rdf_data, format='ovito', interval=None):
  """Reads rdf from input string into pandas.DataFrame

  Parameters
  ----------
  rdf_data: str
      input
  format: str
      'ovito' or 'plain'
  interval:
      overrides distances in file

  Returns
  -------
  pandas.Dataframe
      Data in 3-columns: bin index, corresponding distance and count
  """
  rdf_file = io.StringIO(rdf_data.decode())
  if format == 'ovito':
    # format is ovito 3.0.0-dev234 coordination analysis text output
    df = pd.read_csv(rdf_file,
      delim_whitespace=True,header=None,skiprows=3,index_col=0,
      names=['bin','distance','weight'])
  elif format == 'plain':
    df = pd.read_csv(rdf_file,
      delim_whitespace=True,header=None,skiprows=0,
      names=['distance','weight'])
    df.index.name = 'bin'

  if interval is not None:
    try:
      df.distance = np.linspace(interval[0],interval[1],df.shape[0]+1)[:-1]
    except:
      logger.exception(
        "Could not create grid points from specified interval {}".format(
          interval))
      raise ValueError()

  return df


# In[273]:


# Creates four polar axes, and accesses them through the returned array
#fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
def plotThermo(thermo, axes=None, legend=None, title=None):
    global line_list, label_list
    if axes is None:
        fig, axes = plt.subplots(5, 3, figsize=(20,20)) # constrained_layout=True)
        line_list = []
        label_list = []
    else:
        fig = axes[0,0].get_figure()
        
    l = axes[0, 0].plot(thermo["Step"], thermo["TotEng"])[0]
    axes[0, 1].plot(thermo["Step"], thermo["PotEng"])
    axes[0, 2].plot(thermo["Step"], thermo["KinEng"])

    axes[1, 0].plot(thermo["Step"], thermo["Temp"])
    axes[1, 1].plot(thermo["Step"], thermo["Press"])
    axes[1, 2].plot(thermo["Step"], thermo["Enthalpy"])

    axes[2, 0].plot(thermo["Step"], thermo["E_bond"])
    axes[2, 1].plot(thermo["Step"], thermo["E_angle"])
    axes[2, 2].plot(thermo["Step"], thermo["E_dihed"])

    axes[3, 0].plot(thermo["Step"], thermo["E_pair"])
    axes[3, 1].plot(thermo["Step"], thermo["E_vdwl"])
    axes[3, 2].plot(thermo["Step"], thermo["E_coul"])
    
    axes[4, 0].plot(thermo["Step"], thermo["E_long"])
    axes[4, 1].plot(thermo["Step"], thermo["Volume"])

    axes[0,0].set_title("Total Energy")
    axes[0,1].set_title("Potential Energy")
    axes[0,2].set_title("Kinetic Energy")

    axes[1,0].set_title("Temperature")
    axes[1,1].set_title("Pressure")
    axes[1,2].set_title("Enthalpy")

    axes[2, 0].set_title("E_bond")
    axes[2, 1].set_title("E_angle")
    axes[2, 2].set_title("E_dihed")

    axes[3, 0].set_title("E_pair")
    axes[3, 1].set_title("E_vdwl")
    axes[3, 2].set_title("E_coul")
    
    axes[4, 0].set_title("E_long")
    axes[4, 1].set_title("Volume")
    #axes[1, 1].scatter(x, y)

    axes[0,0].set_ylabel("$E$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")
    axes[0,1].set_ylabel("$V$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")
    axes[0,2].set_ylabel("$K$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")

    axes[1,0].set_ylabel("$T$ [ K ]")
    axes[1,1].set_ylabel("$P$ [ atm ]")
    axes[1,2].set_ylabel("$E$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")

    axes[2,0].set_ylabel("$E$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")
    axes[2,1].set_ylabel("$E$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")
    axes[2,2].set_ylabel("$E$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")

    axes[3,0].set_ylabel("$E$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")
    axes[3,1].set_ylabel("$E$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")
    axes[3,2].set_ylabel("$E$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")
    
    axes[4,0].set_ylabel("$E$ [ $ \mathrm{kcal} \ \mathrm{mol}^{-1}$]")
    axes[4,1].set_ylabel("$V$ [ $ \mathrm{\AA^3} $ ]")
    #axes[4,1].set_visible(False)
    axes[4,2].set_visible(False)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if legend is not None:
        #axes[0,0].legend()
        line_list.append(l)
        label_list.append(legend)
        fig.legend(line_list,label_list,loc="lower right", framealpha=1)
        
    if title is not None:
        fig.suptitle(title)
    
    return fig, axes

line_list = []
label_list = []


# In[274]:


# plot function for 
# 'Step', 'TotEng', 'KinEng', 'PotEng', 'Temp', 'Press', 'Enthalpy',
#       'E_bond', 'E_angle', 'E_dihed', 'E_impro', 'E_pair', 'E_vdwl', 'E_coul',
#       'E_long', 'E_tail', 'Volume', 'Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pxz', 'Pyz'
def plotThermoWithPressureTensor(thermo, axes=None, legend=None, title=None):
    global line_list, label_list
    if axes is None:
        fig, axes = plt.subplots(7, 3, figsize=(20,40)) # constrained_layout=True)
        line_list = []
        label_list = []
    else:
        fig = axes[0,0].get_figure()
    
    fig, axes = plotThermo(thermo, axes, legend, title)
    
    axes[5, 0].plot(thermo["Step"], thermo["Pxx"])
    axes[5, 1].plot(thermo["Step"], thermo["Pyy"])
    axes[5, 2].plot(thermo["Step"], thermo["Pzz"])
    
    axes[6, 0].plot(thermo["Step"], thermo["Pxy"])
    axes[6, 1].plot(thermo["Step"], thermo["Pyz"])
    axes[6, 2].plot(thermo["Step"], thermo["Pyz"])
    
    axes[5, 0].set_title("Pressure tensor component xx")
    axes[5, 1].set_title("Pressure tensor component yy")
    axes[5, 2].set_title("Pressure tensor component zz")

    axes[6, 0].set_title("Pressure tensor component xy")
    axes[6, 1].set_title("Pressure tensor component xz")
    axes[6, 2].set_title("Pressure tensor component yz")
    
    axes[5, 0].set_ylabel("$P_{xx}$ [ atm ]")
    axes[5, 1].set_ylabel("$P_{yy}$ [ atm ]")
    axes[5, 2].set_ylabel("$P_{zz}$ [ atm ]")
    
    axes[6, 0].set_ylabel("$P_{xy}$ [ atm ]")
    axes[6, 1].set_ylabel("$P_{xz}$ [ atm ]")
    axes[6, 2].set_ylabel("$P_{yz}$ [ atm ]")
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if legend is not None:
        fig.legend(line_list,label_list,loc="lower right", framealpha=1)
        
    if title is not None:
        fig.suptitle(title)
    
    return fig, axes

line_list = []
label_list = []


# ## Evaluation

# ### All system-related files

# In[163]:


query = {
    'metadata.sb_name':  'AU_111_150Ang_cube'
}


# In[164]:


files = fp.get_file_by_query(query)


# In[165]:


len(files)


# In[166]:


file_dict = { d["identifier"]: { 
        "metadata": d["metadata"],
        "content":  c } for c, d in files }
    


# In[167]:


pprint(list(file_dict.keys()))


# In[168]:


print(files[0][0][:1000].decode())


# ### Thermo .out files

# In[112]:


query = {
    'identifier': { '$regex': '.*thermo\.out$'},
    'metadata.sb_name':  'AU_111_150Ang_cube'
}


# In[113]:


files = fp.get_file_by_query(query)


# In[71]:


len(files)


# In[72]:


files[0][1]["metadata"]["step"]


# In[73]:


file_dict = { d["metadata"]["step"]: { 
        "metadata": d["metadata"],
        "content":  c } for c, d in files }
    


# In[74]:


pprint(list(file_dict.keys()))


# In[75]:


files[0][1]["metadata"]


# In[76]:


title_pattern = '''{substrate:} ({sb_crystal_plane:}) 
    {sb_base_length:} {sb_base_length_unit:} {sb_shape:} substrate.'''


# In[77]:


legend_pattern = '''{step:}.'''


# In[78]:


thermoData = []
for (cont,doc) in files:
    # create a title from document metadata
    title_raw = title_pattern.format(
        substrate = doc["metadata"]["substrate"],
        sb_crystal_plane = doc["metadata"]["sb_crystal_plane"],
        sb_base_length = doc["metadata"]["sb_base_length"],
        sb_base_length_unit = doc["metadata"]["sb_base_length_unit"],
        sb_shape = doc["metadata"]["sb_shape"])

    title = ' '.join(line.strip() for line in title_raw.splitlines())
    
    # create a plot legend from document metadata
    legend = legend_pattern.format(
            step = doc["metadata"]["step"])
    #legend = ' '.join(line.strip() for line in legend_raw.splitlines())
    
    contStream = io.StringIO(cont.decode())
    df = pd.read_csv(contStream,delim_whitespace=True)
        
    thermoData.append({'title': title, 'legend': legend, 'data': df})


# In[79]:


for i, d in enumerate(thermoData):
    if i == 0:
        fig, ax = plotThermo(d["data"], 
                     legend =d["legend"],
                     title  =d["title"])
    else:
        fig, ax = plotThermo(d["data"], axes=ax,
                        legend=d["legend"])


# ## NVT equilibration

# In[367]:


query = {
    'identifier': { '$regex': 'substrate/.*thermo\.out$'},
    'metadata.sb_name':  'AU_111_150Ang_cube',
    'metadata.step':    'equilibration_nvt',
}


# In[368]:


files = fp.get_file_by_query(query)


# In[369]:


len(files)


# In[370]:


for c,d in files:
    print(d["identifier"])


# ### Thermo .out files

# In[371]:


title_pattern = '''{substrate:} ({sb_crystal_plane:}) 
    {sb_base_length:} {sb_base_length_unit:} {sb_shape:} substrate.'''


# In[372]:


legend_pattern = '''{step:}.'''


# In[373]:


thermoData = []
for (cont,doc) in files:
    # create a title from document metadata
    title_raw = title_pattern.format(
        substrate = doc["metadata"]["substrate"],
        sb_crystal_plane = doc["metadata"]["sb_crystal_plane"],
        sb_base_length = doc["metadata"]["sb_base_length"],
        sb_base_length_unit = doc["metadata"]["sb_base_length_unit"],
        sb_shape = doc["metadata"]["sb_shape"])

    title = ' '.join(line.strip() for line in title_raw.splitlines())
    
    # create a plot legend from document metadata
    legend = legend_pattern.format(
            step = doc["metadata"]["step"])
    #legend = ' '.join(line.strip() for line in legend_raw.splitlines())
    
    contStream = io.StringIO(cont.decode())
    df = pd.read_csv(contStream,delim_whitespace=True)
        
    thermoData.append({'title': title, 'legend': legend, 'data': df})


# In[374]:


for i, d in enumerate(thermoData):
    if i == 0:
        fig, ax = plotThermoWithPressureTensor(d["data"], 
                     legend =d["legend"],
                     title  =d["title"])
    else:
        fig, ax = plotThermoWithPressureTensor(d["data"], axes=ax,
                        legend=d["legend"])


# ## NPT equilibration

# In[383]:


query = {
    'identifier': { '$regex': 'substrate/.*thermo\.out$'},
    'metadata.sb_name':  'AU_111_150Ang_cube',
    'metadata.step':    'equilibration_npt',
}


# In[384]:


files = fp.get_file_by_query(query)


# In[385]:


len(files)


# In[386]:


for c,d in files:
    print(d["identifier"])


# ### Thermo .out files

# In[388]:


title_pattern = '''{substrate:} ({sb_crystal_plane:}) 
    {sb_base_length:} {sb_base_length_unit:} {sb_shape:} substrate.'''


# In[389]:


legend_pattern = '''{step:}.'''


# In[390]:


thermoData = []
for (cont,doc) in files:
    # create a title from document metadata
    title_raw = title_pattern.format(
        substrate = doc["metadata"]["substrate"],
        sb_crystal_plane = doc["metadata"]["sb_crystal_plane"],
        sb_base_length = doc["metadata"]["sb_base_length"],
        sb_base_length_unit = doc["metadata"]["sb_base_length_unit"],
        sb_shape = doc["metadata"]["sb_shape"])

    title = ' '.join(line.strip() for line in title_raw.splitlines())
    
    # create a plot legend from document metadata
    legend = legend_pattern.format(
            step = doc["metadata"]["step"])
    #legend = ' '.join(line.strip() for line in legend_raw.splitlines())
    
    contStream = io.StringIO(cont.decode())
    df = pd.read_csv(contStream,delim_whitespace=True)
        
    thermoData.append({'title': title, 'legend': legend, 'data': df})


# In[391]:


for i, d in enumerate(thermoData):
    if i == 0:
        fig, ax = plotThermoWithPressureTensor(d["data"], 
                     legend =d["legend"],
                     title  =d["title"])
    else:
        fig, ax = plotThermoWithPressureTensor(d["data"], axes=ax,
                        legend=d["legend"])


# ### RDF

# In[392]:


from postprocessing import find_histogram_peak, plot_histogram


# In[393]:


query = {
    'identifier': { '$regex': '.*rdf\.txt$'},
    'metadata.sb_name':  'AU_111_150Ang_cube',
    'metadata.step':    'equilibration_npt',
}


# In[394]:


files = fp.get_file_by_query(query)


# In[395]:


len(files)


# In[396]:


for c,d in files:
    print(d["identifier"])


# In[397]:


dataset_names = list(map(lambda d: os.path.basename(d[1]["identifier"]), files))


# In[398]:


dataset_names


# In[399]:


df = read_rdf(files[0][0],format='plain')


# In[400]:


df.columns


# In[401]:


plt.plot(df["distance"],df["weight"])


# In[402]:


plot_histogram(df,interval=(3.5,4.5))


# In[403]:


peak_position = find_histogram_peak(df,interval=(3.5, 4.5))


# In[404]:


peak_position = {}
for n, d in zip(dataset_names, files):
    df = read_rdf(d[0],format='plain')

    #plt.plot(df["distance"],df["weight"])
    #plot_histogram(df,interval=(3.5,4.5))

    peak_position[n] = find_histogram_peak(df,interval=(3.5, 4.5))


# In[405]:


peak_position


# ### Box measures

# In[406]:


query = {
    'identifier': { '$regex': '.*box\.txt$'},
    'metadata.sb_name':  'AU_111_150Ang_cube',
    'metadata.step':    'equilibration_npt',
}


# In[407]:


files = fp.get_file_by_query(query)


# In[408]:


len(files)


# In[409]:


cont = files[0][0]
contStream = io.StringIO(cont.decode())
#df = pd.read_csv(contStream,delim_whitespace=True)


# In[410]:


cell_measures = np.loadtxt(contStream)


# In[411]:


cell_measures


# In[412]:


t = np.linspace(0,200.0,len(cell_measures))


# In[413]:


len(cell_measures)


# In[414]:


plt.subplot(131)
plt.plot(t, cell_measures[:,0], label='x' )
plt.title('x')
plt.xlabel("t (ps)")
plt.ylabel("d ($\AA$)")
plt.subplot(132)
plt.plot(t, cell_measures[:,1], label='y' )
plt.title('y')
plt.xlabel("t (ps)")
plt.ylabel("d ($\AA$)")
plt.subplot(133)
plt.plot(t, cell_measures[:,2], label='z' )
plt.title('z')
#plt.legend()
plt.xlabel("t (ps)")
plt.ylabel("d ($\AA$)")


# In[415]:


mean_measures = np.mean(cell_measures[100:,:],axis=0)


# In[416]:


# relation between plane_spacings in this orientation and lattice constant:
plane_spacing_to_lattice_constant = np.array(
    [np.sqrt(2), np.sqrt(6), np.sqrt(3)] )


# In[417]:


approximate_crystal_plane_spacing = 4.075 / plane_spacing_to_lattice_constant


# In[418]:


approximate_crystal_plane_spacing


# In[419]:


estimated_crystal_plane_count = np.round(mean_measures / approximate_crystal_plane_spacing)


# In[420]:


estimated_crystal_plane_count


# In[422]:


estimated_unit_cell_multiples = estimated_crystal_plane_count / [2,1,3]


# In[423]:


estimated_unit_cell_multiples


# In[424]:


np.prod(estimated_unit_cell_multiples)*4 # agrees with atom count


# In[425]:


exact_crystal_plane_spacing =  mean_measures / estimated_crystal_plane_count


# In[426]:


exact_crystal_plane_spacing


# In[427]:


approximate_crystal_plane_spacing


# In[428]:


# deviation from ideal crystal plane spacing 
100.0*( exact_crystal_plane_spacing - approximate_crystal_plane_spacing) / approximate_crystal_plane_spacing


# In[429]:


anisotropic_lattice_spacing = exact_crystal_plane_spacing*plane_spacing_to_lattice_constant


# In[430]:


anisotropic_lattice_spacing


# In[ ]:




