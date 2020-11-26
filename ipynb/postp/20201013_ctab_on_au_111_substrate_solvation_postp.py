#!/usr/bin/env python
# coding: utf-8

# # Analyze substrate solvation

# This notebook demonstrates deposition of an SDS adsorption layer on a non-spherical AFM tip model.

# ## Initialization

# ### IPython magic

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[14]:


get_ipython().run_line_magic('aimport', '')


# ### Imports

# In[2]:


import ase.io # here used for reading pdb files
from ase.visualize import view
from ase.visualize.plot import plot_atoms # has nasty offset issues
from cycler import cycler # here used for cycling through colors in plots
import datetime
import fabric # for pythonic ssh connections
from fireworks import LaunchPad, Firework, Tracker, Workflow 
from fireworks import FileTransferTask, PyTask, ScriptTask

# FireWorks functionality 
from fireworks import Firework, LaunchPad, ScriptTask, Workflow
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask, GetFilesTask, GetFilesByQueryTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask
from fireworks.utilities.filepad import FilePad # direct FilePad access, similar to the familiar LaunchPad

from collections.abc import Iterable
import glob
import gc # manually clean up memory with gc.collect()
import gromacs # GromacsWrapper, here used for evoking gmc commands, reading and writing .ndx files
# from io import StringIO, TextIOWrapper
import io
from IPython.display import display, Image #, Video # display image files within notebook
from ipywidgets import Video  # display video within notebook
import itertools # for products of iterables
import json # generic serialization of lists and dicts
import jinja2 # here used for filling packmol input script template
import jinja2.meta # for gathering variables in a jinja2 template
import logging 
import matplotlib.pyplot as plt
import MDAnalysis as mda # here used for reading and analyzing gromacs trajectories
import MDAnalysis.analysis.rdf as mda_rdf
import MDAnalysis.analysis.rms as mda_rms
from mpl_toolkits.mplot3d import Axes3D # here used for 3d point cloud scatter plot
import miniball # finds minimum bounding sphere of a point set
import nglview
import numpy as np
import os, os.path
import pandas as pd
import panedr # reads GROMACS edr into pandas df, requires pandas and pbr
import parmed as pmd # has quite a few advantages over ASE when it comes to parsing pdb
from pprint import pprint
import pymongo # for sorting in queries
import scipy.constants as sc
import subprocess # used for evoking external packmol
import sys
import tempfile
import yaml


# GromacsWrapper might need a file `~/.gromacswrapper.cfg` with content
# ```cfg
# [Gromacs]
# tools = gmx gmx_d 
# # gmx_mpi_d gmx_mpi_d
# 
# # name of the logfile that is written to the current directory
# logfilename = gromacs.log
# 
# # loglevels (see Python's logging module for details)
# #   ERROR   only fatal errors
# #   WARN    only warnings
# #   INFO    interesting messages
# #   DEBUG   everything
# 
# # console messages written to screen
# loglevel_console = INFO
# 
# # file messages written to logfilename
# loglevel_file = DEBUG
# ```
# in order to know the GROMACS executables it is allowed to use. Otherwise,
# calls to `gmx_mpi` or `gmx_mpi_d` without MPI wrapper might lead to MPI 
# warnings in output that cause GromacsWrapper to fail.

# ### Logging

# In[3]:


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ParmEd needs to know the GROMACS topology folder, usually get this from 
# envionment variable `GMXLIB`:

# ### Function definitions

# In[4]:


def find_undeclared_variables(infile):
    """identify all variables evaluated in a jinja 2 template file"""
    env = jinja2.Environment()
    with open(infile) as template_file:
        parsed = env.parse(template_file.read())

    undefined = jinja2.meta.find_undeclared_variables(parsed)
    return undefined


# In[5]:


def memuse():
    """Quick overview on memory usage of objects in Jupyter notebook"""
    # https://stackoverflow.com/questions/40993626/list-memory-usage-in-ipython-and-jupyter
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    return sorted([(x, sys.getsizeof(globals().get(x))) for x in dir(sys.modules['__main__']) if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# ### Global settings

# In[13]:


# pandas settings
pd.options.display.max_rows = 200
pd.options.display.max_columns = 16
pd.options.display.max_colwidth = 256


# In[6]:


os.environ['GMXLIB'] = '/gmx_top'


# In[7]:


# pmd.gromacs.GROMACS_TOPDIR = os.environ['GMXLIB']
pmd.gromacs.GROMACS_TOPDIR = '/gmx_top'


# In[8]:


# prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'
prefix = '/mnt/dat/work'


# In[9]:


work_prefix = '/mnt/dat/work/tmp'


# In[10]:


try:
    os.mkdir(work_prefix)
except FileExistsError as exc:
    print(exc)


# In[11]:


os.chdir(work_prefix)


# In[12]:


# the FireWorks LaunchPad
lp = LaunchPad.auto_load() #Define the server and database
# FilePad behaves analogous to LaunchPad
fp = FilePad.auto_load()


# ## Conversion from LAMMPS data format to PDB

# The following bash / tcl snippet converts a LAMMPS data file to PDB, assigning the desired names as mapped in a yaml file
# ```bash
# #!/bin/bash
# # echo "package require jlhvmd; jlh lmp2pdb indenter.lammps indenter.pdb" | vmd -eofexit
# vmd -eofexit << 'EOF'
# package require jlhvmd
# topo readlammpsdata indenter.lammps
# jlh type2name SDS_type2name.yaml
# jlh name2res  SDS_name2res.yaml
# set sel [atomselect top all]
# $sel writepdb indenter.pdb
# EOF
# 
# pdb_chain.py indenter.pdb > indenter_wo_chainid.pdb
# pdb_reres_by_atom_9999.py indenter_wo_chainid.pdb > indenter_reres.pdb
# ```
# 
# Requires
# 
# * VMD (tested with 1.9.3) with topotools
# * jlhvmd VMD plugin: https://github.com/jotelha/jlhvmd
# * pdb-tools: https://github.com/haddocking/pdb-tools/

# ## Overview

# ### Overview on projects in database

# In[14]:


query = {'metadata.datetime': {'$gt': '2020'} }


# In[15]:


fp.filepad.count_documents(query)


# In[16]:


aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 'project': '$metadata.project' },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # pull 'project' field up in hierarchy
        "$addFields": { 
            "project": "$_id.project",
        },
    },
    {  # drop nested '_id.project'
        "$project": { 
            "_id": False 
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

# sort_aggregation
#aggregation_pipeline = [ match_aggregation, group_aggregation, set_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [c for c in cursor]
res_df = pd.DataFrame(data=res) # pandas Dataframe is just nice for printing in notebook


# In[17]:


res_df


# ### Overview on steps in project

# In[18]:


project_id = '2020-10-13-ctab-on-au-111-substrate-passivation'


# In[19]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[20]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[21]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$metadata.step',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [ {**c['_id'], **c} for c in cursor]
columns = ['step', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[22]:


res_df


# In[23]:


res_df['step'].values


# ### Overview on objects in project

# In[24]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[25]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[26]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'type': '$metadata.type',
                'name': '$metadata.name',
                'step': '$metadata.step',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [ {**c['_id'], **c} for c in cursor]
columns = ['type', 'step', 'name', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[27]:


res_df


# ### Overview on images by distinct steps

# In[28]:


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
}


# In[29]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[30]:


# check files degenerate by 'metadata.type' ad 'metadata.name'

aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'type': '$metadata.type',
                'name': '$metadata.name',
                'step': '$metadata.step',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [ {**c['_id'], **c} for c in cursor]
columns = ['step', 'type', 'name', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[31]:


res_df


# In[32]:


res_df["step"][0]


# ## Packing visualization

# ### Surfactant measures

# In[33]:


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
    'metadata.step': {'$regex': 'SurfactantMoleculeMeasures'}
}


# In[34]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[35]:


# check files degenerate by 'metadata.type' ad 'metadata.name'

aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

obj_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["gfs_id"])
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
        tmp.write(content)
        obj_list.append(Image(filename=tmp.name)) 
    print('.',end='')


# In[36]:


obj_list[0]


# ## Energy minimization after solvation analysis

# ### Overview on objects in step

# In[37]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsEnergyMinimizationAfterSolvation'}
}


# In[38]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[39]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'type': '$metadata.type',
                'name': '$metadata.name',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [ {**c['_id'], **c} for c in cursor]
columns = ['type', 'name', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[40]:


res_df


# ### Global observables

# In[41]:


query = { 
    "metadata.project": project_id,
    'metadata.step': {'$regex': 'GromacsEnergyMinimizationAfterSolvation'},
    "metadata.type": 'energy_file',
}
fp.filepad.count_documents(query)


# In[60]:


#parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}
#parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules',
    'shape': 'metadata.system.surfactant.aggregates.shape'
}


# In[61]:


parameter_keys = list(parameter_dict.keys())


# In[62]:


aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)


# In[66]:


res_mi_list = []

aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    with tempfile.NamedTemporaryFile(suffix='.edr') as tmp:
        tmp.write(content)
        try:
            em_df = panedr.edr_to_df(tmp.name)
        except:
            logger.warning("Failed reading edr for {}".format(metadata))
            continue
        
        mi = pd.MultiIndex.from_product(
            [c["_id"].values(),em_df.index],
            names=[*c["_id"].keys(),'step'])
        em_mi_df = em_df.set_index(mi)        
        res_mi_list.append(em_mi_df)
    print('.',end='')
print('')

res_mi_df = pd.concat(res_mi_list)
res_df = res_mi_df.reset_index()


# In[ ]:


res_mi_df


# In[64]:


y_quantities = [
    'Potential',
    'Pressure',
    'Bond',
    'Coulomb (SR)',
    'Coul. recip.',
    ]

positions = [
    (0,0),
    (0,1),
    (1,0),
    (2,0),
    (2,1),
]
fig, ax = plt.subplots(3,2,figsize=(10,12))
for key, grp in res_df.groupby(parameter_keys[0]):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ###  Visualize trajectory

# In[67]:


query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsEnergyMinimizationAfterSolvation'},
    'metadata.type': 'mp4_file',
}


# In[68]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[69]:


# check

aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'nmolecules': '$metadata.system.surfactant.nmolecules'
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

# for i, c in enumerate(cursor): 
#    content, metadata = fp.get_file_by_id(c["latest"])
#    nmolecules = int(c["_id"]["nmolecules"])
    

res = [ {**c['_id'], **c} for c in cursor]
columns = ['nmolecules', 'name', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[70]:


res_df


# In[71]:


parameter_keys


# In[72]:


aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
    {
        "$sort": { 
            "_id.{}".format(parameter_keys[0]): pymongo.DESCENDING,
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

obj_dict = {}
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    # print(metadata['metadata'])
    with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as tmp:
        tmp.write(content)
        obj_dict.update({c["_id"][parameter_keys[0]]: Video.from_file(tmp.name)})
    print('.',end='')


# In[73]:


c


# In[74]:


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# ## NVT equilibration analysis

# ### Overview on objects in step

# In[75]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNVTEquilibration'}
}


# In[76]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[77]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'type': '$metadata.type',
                'name': '$metadata.name',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [ {**c['_id'], **c} for c in cursor]
columns = ['type', 'name', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[78]:


res_df


# ### Global observables

# In[88]:


query = { 
    "metadata.project": project_id,
    'metadata.step': {'$regex': 'GromacsNVTEquilibration'},
    "metadata.type":    'energy_file',
}
fp.filepad.count_documents(query)


# In[89]:


parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules',
    'shape': 'metadata.system.surfactant.aggregates.shape'
}

parameter_keys = list(parameter_dict.keys())


# In[99]:


aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)


# In[112]:


res_list = []

aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    with tempfile.NamedTemporaryFile(suffix='.edr') as tmp:
        tmp.write(content)
        res_df = panedr.edr_to_df(tmp.name)
        
        mi = pd.MultiIndex.from_product(
            [ *[[v] for v in c["_id"].values()],res_df.index],
            names=[*c["_id"].keys(),'step'])
        res_mi_df = res_df.set_index(mi)
        res_list.append(res_mi_df)
    print('.',end='')
print('')
res_df_mi = pd.concat(res_list)
res_df = res_df_mi.reset_index()


# In[113]:


res_df.columns


# In[114]:


res_df_mi


# In[117]:


#n = len(res_df['nmolecules'].unique())
y_quantities = [
    'Temperature',
    'Pressure',
    'Potential',
    'Bond',
    'Coulomb (SR)',
    'Coul. recip.',
    ]

n = len(y_quantities)
cols = 2
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for key, grp in res_df.groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ### Visualize trajectory

# In[118]:


query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNVTEquilibration'},
    'metadata.type': 'mp4_file',
}


# In[119]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[120]:


#parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}
#parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())

aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
    {
        "$sort": { 
            "_id.{}".format(parameter_keys[0]): pymongo.DESCENDING,
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

obj_dict = {}
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    # print(metadata['metadata'])
    with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as tmp:
        tmp.write(content)
        key = tuple(c["_id"].values())
        obj_dict.update({key: Video.from_file(tmp.name)})
    print('.',end='')


# In[121]:


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# ## NPT equilibration analysis

# ### Overview on objects in step

# In[122]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNPTEquilibration:push_filepad'}
}


# In[123]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[124]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'type': '$metadata.type',
                'name': '$metadata.name',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [ {**c['_id'], **c} for c in cursor]
columns = ['type', 'name', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[125]:


res_df


# ### Global observables

# In[126]:


query = { 
    "metadata.project": project_id,
    'metadata.step': {'$regex': 'GromacsNPTEquilibration'},
    "metadata.type":    'energy_file',
}
fp.filepad.count_documents(query)


# In[127]:


metadata = fp.filepad.find_one(query)


# In[128]:


metadata


# In[130]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[131]:


aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)


# In[132]:


[ c for c in cursor]


# In[136]:


res_list = []

cursor = fp.filepad.aggregate(aggregation_pipeline)

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    with tempfile.NamedTemporaryFile(suffix='.edr') as tmp:
        tmp.write(content)
        res_df = panedr.edr_to_df(tmp.name)
        
        mi = pd.MultiIndex.from_product(
            [*[[v] for v in c["_id"].values()],res_df.index],
            names=[*c["_id"].keys(),'step'])
        res_mi_df = res_df.set_index(mi)
        res_list.append(res_mi_df)
    print('.',end='')
print('')
res_df_mi = pd.concat(res_list)
res_df = res_df_mi.reset_index()


# In[137]:


res_df.columns


# In[138]:


res_df_mi


# In[148]:


#n = len(res_df['nmolecules'].unique())
y_quantities = [
    'Temperature',
    'Pressure',
    'Volume',
    'Potential',
    'Bond',
    'Coulomb (SR)',
    'Coul. recip.',
    ]

n = len(y_quantities)
cols = 2
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for key, grp in res_df.groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        # grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        grp.plot('Time',y_quantity,ax=ax[position],legend=False,title=y_quantity)
        
fig.tight_layout()


# ### Visualize trajectory

# In[153]:


query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNPTEquilibration'},
    'metadata.type': 'mp4_file',
}


# In[154]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[155]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[156]:


aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
    {
        "$sort": { 
            "_id.{}".format(parameter_keys[0]): pymongo.DESCENDING,
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

obj_dict = {}
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    # print(metadata['metadata'])
    with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as tmp:
        tmp.write(content)
        # obj_list.append(Video(filename=tmp.name)) 
        # obj_list.append(tmp.name)
        key = tuple(c["_id"].values())
        obj_dict.update({key: Video.from_file(tmp.name)})
    print('.',end='')


# In[157]:


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# 
# ### Pre-evaluated RDF

# #### Overview

# In[158]:


query = { 
    "metadata.project": project_id,
    "metadata.type": {'$regex': '.*rdf$'},
    "metadata.step": {'$regex': "GromacsNPTEquilibration"},
}

fp.filepad.count_documents(query)


# In[159]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'type': '$metadata.type',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [ {**c['_id'], **c} for c in cursor]
columns = ['type', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[160]:


res_df


# #### Substrate - surfactant head RDF

# In[161]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_head_rdf',
    "metadata.step": {'$regex': "GromacsNPTEquilibration"},
}

fp.filepad.count_documents(query)


# In[162]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[167]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# In[168]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['parameter_key']):
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'],data['rdf'][0], label='First frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][len(data)//2],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][-1],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Substrate - surfactant tail RDF

# In[169]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsNPTEquilibration"},
}

fp.filepad.count_documents(query)


# In[171]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
second_sort_aggregation = {
    "$sort": { 
        "_id.{}".format(parameter_keys[0]): pymongo.DESCENDING,
    }
}


aggregation_pipeline = [ 
    match_aggregation, sort_aggregation, group_aggregation, second_sort_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[172]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
    
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'],data['rdf'][0], label='First frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][len(data)//2],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][-1],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Surfactant head - surfactant tail RDF

# In[173]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'surfactant_head_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsNPTEquilibration"},
}

fp.filepad.count_documents(query)


# In[174]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
second_sort_aggregation = {
    "$sort": { 
        "_id.{}".format(parameter_keys[0]): pymongo.DESCENDING,
    }
}

aggregation_pipeline = [ 
    match_aggregation, sort_aggregation, group_aggregation, second_sort_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[175]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['nmolecules']):
for pos, (nmolecules, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'],data['rdf'][0], label='First frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][len(data)//2],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][-1],label='Last frame RDF')
    ax[pos].set_title(nmolecules)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# ## Relaxation analysis

# ### Overview on objects in step

# In[176]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsRelaxation'}
}


# In[177]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[178]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'type': '$metadata.type',
                'name': '$metadata.name',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [ {**c['_id'], **c} for c in cursor]
columns = ['type', 'name', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[179]:


res_df


# ### Global observables

# In[194]:


query = { 
    "metadata.project": project_id,
    'metadata.step': {'$regex': 'GromacsRelaxation'},
    "metadata.type":    'energy_file',
}
fp.filepad.count_documents(query)


# In[195]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[196]:


aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)


# In[197]:


res_list = []

cursor = fp.filepad.aggregate(aggregation_pipeline)

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    with tempfile.NamedTemporaryFile(suffix='.edr') as tmp:
        tmp.write(content)
        res_df = panedr.edr_to_df(tmp.name)
        
        mi = pd.MultiIndex.from_product(
            [*[[v] for v in c["_id"].values()],res_df.index],
            names=[*c["_id"].keys(),'step'])
        res_mi_df = res_df.set_index(mi)
        res_list.append(res_mi_df)
    print('.',end='')
print('')
res_df_mi = pd.concat(res_list)
res_df = res_df_mi.reset_index()


# In[198]:


res_df.columns


# In[199]:


res_df_mi


# In[200]:


#n = len(res_df['nmolecules'].unique())
y_quantities = [
    'Temperature',
    'Pressure',
    'Volume',
    'Potential',
    'Bond',
    'Coulomb (SR)',
    'Coul. recip.',
    ]

n = len(y_quantities)
cols = 2
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for key, grp in res_df.groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        # grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        grp.plot('Time', y_quantity, ax=ax[position], legend=False, title=y_quantity)
        
fig.tight_layout()


# ### Visualize trajectory

# In[205]:


query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsRelaxation'},
    'metadata.type': 'mp4_file',
}


# In[206]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[207]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[208]:


aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    },
    { 
        "$group": { 
            "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
            "degeneracy": {"$sum": 1}, # number matching data sets
            "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
        }
    },
    {
        "$sort": { 
            "_id.{}".format(parameter_keys[0]): pymongo.DESCENDING,
        }
    },
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

obj_dict = {}
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    # print(metadata['metadata'])
    with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as tmp:
        tmp.write(content)
        # obj_list.append(Video(filename=tmp.name)) 
        # obj_list.append(tmp.name)
        key = tuple(c["_id"].values())
        obj_dict.update({key: Video.from_file(tmp.name)})
    print('.',end='')


# In[209]:


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# 
# ### Pre-evaluated RDF

# #### Overview

# In[210]:


query = { 
    "metadata.project": project_id,
    "metadata.type": {'$regex': '.*rdf$'},
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[211]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'type': '$metadata.type',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$metadata.datetime' },
            "latest":  {'$max': '$metadata.datetime' },
        },
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

cursor = fp.filepad.aggregate(aggregation_pipeline)

res = [ {**c['_id'], **c} for c in cursor]
columns = ['type', 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[212]:


res_df


# #### Substrate - surfactant head RDF

# In[219]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_head_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[220]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[221]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[222]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['parameter_key']):
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'],data['rdf'][0], label='First frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][len(data)//2],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][-1],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Substrate - surfactant tail RDF

# In[223]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[224]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
second_sort_aggregation = {
    "$sort": { 
        "_id.{}".format(parameter_keys[0]): pymongo.DESCENDING,
    }
}


aggregation_pipeline = [ 
    match_aggregation, sort_aggregation, group_aggregation, second_sort_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[225]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
    
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'],data['rdf'][0], label='First frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][len(data)//2],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][-1],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Surfactant head - surfactant tail RDF

# In[226]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'surfactant_head_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[227]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
second_sort_aggregation = {
    "$sort": { 
        "_id.{}".format(parameter_keys[0]): pymongo.DESCENDING,
    }
}

aggregation_pipeline = [ 
    match_aggregation, sort_aggregation, group_aggregation, second_sort_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[228]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['nmolecules']):
for pos, (nmolecules, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'],data['rdf'][0], label='First frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][len(data)//2],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][-1],label='Last frame RDF')
    ax[pos].set_title(nmolecules)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Surfactant head - surfactant head RDF

# In[232]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'surfactant_head_surfactant_head_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[233]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[234]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[235]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['parameter_key']):
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'][1:],data['rdf'][0][1:], label='First frame RDF')
    ax[pos].plot(data['dist'][1:],data['rdf'][len(data)//2][1:],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'][1:],data['rdf'][-1][1:],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# In[236]:


n = 1
cols = 2 if n > 1 else 1
rows = round(n/cols)

fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for (parameter_key, data) in res_dict.items():
    ax.plot(data['dist'][1:],data['rdf'][-1][1:],label=parameter_key)
    ax.legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Surfactant tail - surfactant tail RDF

# In[241]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'surfactant_tail_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[242]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[243]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[244]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['parameter_key']):
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'][1:],data['rdf'][0][1:], label='First frame RDF')
    ax[pos].plot(data['dist'][1:],data['rdf'][len(data)//2][1:],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'][1:],data['rdf'][-1][1:],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# In[245]:


n = 1
cols = 2 if n > 1 else 1
rows = round(n/cols)

fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for (parameter_key, data) in res_dict.items():
    ax.plot(data['dist'][1:],data['rdf'][-1][1:],label=parameter_key)
    ax.legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Counterion - surfactant head RDF

# In[249]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'counterion_surfactant_head_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[250]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[251]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[252]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['parameter_key']):
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'],data['rdf'][0], label='First frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][len(data)//2],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][-1],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# In[253]:


n = 1
cols = 2 if n > 1 else 1
rows = round(n/cols)

fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for (parameter_key, data) in res_dict.items():
    ax.plot(data['dist'],data['rdf'][-1],label=parameter_key)
    ax.legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Counterion - surfactant tail RDF

# In[254]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'counterion_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[255]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[256]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[257]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['parameter_key']):
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'],data['rdf'][0], label='First frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][len(data)//2],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][-1],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# In[258]:


n = 1
cols = 2 if n > 1 else 1
rows = round(n/cols)

fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for (parameter_key, data) in res_dict.items():
    ax.plot(data['dist'],data['rdf'][-1],label=parameter_key)
    ax.legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Counterion - substrate RDF

# In[259]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'counterion_substrate_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[260]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[261]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[262]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['parameter_key']):
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'],data['rdf'][0], label='First frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][len(data)//2],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'],data['rdf'][-1],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# In[263]:


n = 1
cols = 2 if n > 1 else 1
rows = round(n/cols)

fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for (parameter_key, data) in res_dict.items():
    ax.plot(data['dist'],data['rdf'][-1],label=parameter_key)
    ax.legend()

fig.tight_layout()
# fig.legend()
fig.show()


# #### Counterion - counterion RDF

# In[269]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'counterion_counterion_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# In[270]:


parameter_dict = {
    'shape': 'metadata.system.surfactant.aggregates.shape',
    'nmolecules': 'metadata.system.surfactant.nmolecules'
}
parameter_keys = list(parameter_dict.keys())


# In[271]:


res_dict = {}
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

# res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    parameter_values = tuple(c["_id"][parameter_key] for parameter_key in parameter_keys)
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_values] = {'dist': d, 'rdf': rdf}
    print('.',end='')
print('')


# In[272]:


n = len(res_dict)
cols = 2 if n > 1 else 1
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
if not isinstance(ax, Iterable):
    ax = [ax]
# for key, grp in res_df.groupby(['parameter_key']):
for pos, (parameter_key, data) in zip(positions, res_dict.items()):
    ax[pos].plot(data['dist'][1:],data['rdf'][0][1:], label='First frame RDF')
    ax[pos].plot(data['dist'][1:],data['rdf'][len(data)//2][1:],label='Intermediate frame RDF')
    ax[pos].plot(data['dist'][1:],data['rdf'][-1][1:],label='Last frame RDF')
    ax[pos].set_title(parameter_key)
    ax[pos].legend()

fig.tight_layout()
# fig.legend()
fig.show()


# In[273]:


n = 1
cols = 2 if n > 1 else 1
rows = round(n/cols)

fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for (parameter_key, data) in res_dict.items():
    ax.plot(data['dist'][1:],data['rdf'][-1][1:],label=parameter_key)
    ax.legend()

fig.tight_layout()
# fig.legend()
fig.show()


# In[ ]:




