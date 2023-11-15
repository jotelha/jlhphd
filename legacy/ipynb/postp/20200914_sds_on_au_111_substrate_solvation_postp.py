#!/usr/bin/env python
# coding: utf-8
# %%

# # Analyze substrate solvation

# This notebook demonstrates deposition of an SDS adsorption layer on a non-spherical AFM tip model.

# ## Initialization

# ### IPython magic

# %%


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%


get_ipython().run_line_magic('aimport', '')


# ### Imports

# %%


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

# %%


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ParmEd needs to know the GROMACS topology folder, usually get this from 
# envionment variable `GMXLIB`:

# ### Function definitions

# %%


def find_undeclared_variables(infile):
    """identify all variables evaluated in a jinja 2 template file"""
    env = jinja2.Environment()
    with open(infile) as template_file:
        parsed = env.parse(template_file.read())

    undefined = jinja2.meta.find_undeclared_variables(parsed)
    return undefined


# %%


def memuse():
    """Quick overview on memory usage of objects in Jupyter notebook"""
    # https://stackoverflow.com/questions/40993626/list-memory-usage-in-ipython-and-jupyter
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    return sorted([(x, sys.getsizeof(globals().get(x))) for x in dir(sys.modules['__main__']) if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# ### Global settings

# %%


# pandas settings
pd.options.display.max_rows = 200
pd.options.display.max_columns = 16
pd.options.display.max_colwidth = 256


# %%


os.environ['GMXLIB'] = '/gmx_top'


# %%


# pmd.gromacs.GROMACS_TOPDIR = os.environ['GMXLIB']
pmd.gromacs.GROMACS_TOPDIR = '/gmx_top'


# %%


# prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'
prefix = '/mnt/dat/work'


# %%


work_prefix = '/mnt/dat/work/tmp'


# %%


try:
    os.mkdir(work_prefix)
except FileExistsError as exc:
    print(exc)


# %%


os.chdir(work_prefix)


# %%


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

# %%


query = {'metadata.datetime': {'$gt': '2020'} }


# %%


fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# ### Overview on steps in project

# %%


project_id = '2020-09-10-sds-on-au-111-substrate-passivation-trial'


# %%


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# %%


res_df['step'].values


# ### Overview on objects in project

# %%


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# ### Overview on images by distinct steps

# %%


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# %%


res_df["step"][0]


# ## Packing visualization

# ### Surfactant measures

# %%


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
    'metadata.step': {'$regex': 'SurfactantMoleculeMeasures'}
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


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


# %%


obj_list[0]


# ## Energy minimization after solvation analysis

# ### Overview on objects in step

# %%


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsEnergyMinimizationAfterSolvation'}
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# ### Global observables

# %%


query = { 
    "metadata.project": project_id,
    'metadata.step': {'$regex': 'GromacsEnergyMinimizationAfterSolvation'},
    "metadata.type": 'energy_file',
}
fp.filepad.count_documents(query)


# %%


#parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}
parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}


# %%


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


# %%


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
        em_df = panedr.edr_to_df(tmp.name)
        
        mi = pd.MultiIndex.from_product(
            [c["_id"].values(),em_df.index],
            names=[*c["_id"].keys(),'step'])
        em_mi_df = em_df.set_index(mi)        
        res_mi_list.append(em_mi_df)
    print('.',end='')
print('')

res_mi_df = pd.concat(res_mi_list)
res_df = res_mi_df.reset_index()


# %%


res_mi_df


# %%


parameter_keys = list(parameter_dict.keys())


# %%


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

# %%


query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsEnergyMinimizationAfterSolvation'},
    'metadata.type': 'mp4_file',
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# %%


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


# %%


c


# %%


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# ## NVT equilibration analysis

# ### Overview on objects in step

# %%


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNVTEquilibration'}
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# ### Global observables

# %%


query = { 
    "metadata.project": project_id,
    'metadata.step': {'$regex': 'GromacsNVTEquilibration'},
    "metadata.type":    'energy_file',
}
fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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


# %%


[ c for c in cursor]


# %%


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
            [c["_id"].values(),res_df.index],
            names=[*c["_id"].keys(),'step'])
        res_mi_df = res_df.set_index(mi)
        res_list.append(res_mi_df)
    print('.',end='')
print('')
res_df_mi = pd.concat(res_list)
res_df = res_df_mi.reset_index()


# %%


res_df.columns


# %%


res_df_mi


# %%


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
for key, grp in res_df.groupby([parameter_keys[0]]):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ### Visualize trajectory

# %%


query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNVTEquilibration'},
    'metadata.type': 'mp4_file',
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


#parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}
parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
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
        # obj_list.append(Video(filename=tmp.name)) 
        # obj_list.append(tmp.name)
        key = tuple(c["_id"].values())
        obj_dict.update({key: Video.from_file(tmp.name)})
    print('.',end='')


# %%


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# ## NPT equilibration analysis

# ### Overview on objects in step

# %%


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNPTEquilibration:push_filepad'}
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# ### Global observables

# %%


query = { 
    "metadata.project": project_id,
    'metadata.step': {'$regex': 'GromacsNPTEquilibration'},
    "metadata.type":    'energy_file',
}
fp.filepad.count_documents(query)


# %%


metadata = fp.filepad.find_one(query)


# %%


metadata


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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


# %%


[ c for c in cursor]


# %%


res_list = []

cursor = fp.filepad.aggregate(aggregation_pipeline)

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    with tempfile.NamedTemporaryFile(suffix='.edr') as tmp:
        tmp.write(content)
        res_df = panedr.edr_to_df(tmp.name)
        
        mi = pd.MultiIndex.from_product(
            [c["_id"].values(),res_df.index],
            names=[*c["_id"].keys(),'step'])
        res_mi_df = res_df.set_index(mi)
        res_list.append(res_mi_df)
    print('.',end='')
print('')
res_df_mi = pd.concat(res_list)
res_df = res_df_mi.reset_index()


# %%


res_df.columns


# %%


res_df_mi


# %%


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
for key, grp in res_df.groupby([parameter_keys[0]]):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ### Visualize trajectory

# %%


query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNPTEquilibration'},
    'metadata.type': 'mp4_file',
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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


# %%


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# 
# ### Pre-evaluated RDF

# #### Overview

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": {'$regex': '.*rdf$'},
    "metadata.step": {'$regex': "GromacsNPTEquilibration"},
}

fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# #### Substrate - surfactant head RDF

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_head_rdf',
    "metadata.step": {'$regex': "GromacsNPTEquilibration"},
}

fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsNPTEquilibration"},
}

fp.filepad.count_documents(query)


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'surfactant_head_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsNPTEquilibration"},
}

fp.filepad.count_documents(query)


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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

# %%


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsRelaxation'}
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# ### Global observables

# %%


query = { 
    "metadata.project": project_id,
    'metadata.step': {'$regex': 'GromacsRelaxation'},
    "metadata.type":    'energy_file',
}
fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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


# %%


[ c for c in cursor]


# %%


res_list = []

cursor = fp.filepad.aggregate(aggregation_pipeline)

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    with tempfile.NamedTemporaryFile(suffix='.edr') as tmp:
        tmp.write(content)
        res_df = panedr.edr_to_df(tmp.name)
        
        mi = pd.MultiIndex.from_product(
            [c["_id"].values(),res_df.index],
            names=[*c["_id"].keys(),'step'])
        res_mi_df = res_df.set_index(mi)
        res_list.append(res_mi_df)
    print('.',end='')
print('')
res_df_mi = pd.concat(res_list)
res_df = res_df_mi.reset_index()


# %%


res_df.columns


# %%


res_df_mi


# %%


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
for key, grp in res_df.groupby([parameter_keys[0]]):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ### Visualize trajectory

# %%


query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsRelaxation'},
    'metadata.type': 'mp4_file',
}


# %%


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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


# %%


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# 
# ### Pre-evaluated RDF

# #### Overview

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": {'$regex': '.*rdf$'},
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


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


# %%


res_df


# #### Substrate - surfactant head RDF

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_head_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'surfactant_head_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'surfactant_head_surfactant_head_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'surfactant_tail_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'counterion_surfactant_head_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'counterion_surfactant_tail_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'counterion_substrate_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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


# %%


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

# %%


query = { 
    "metadata.project": project_id,
    "metadata.type": 'counterion_counterion_rdf',
    "metadata.step": {'$regex': "GromacsRelaxation"},
}

fp.filepad.count_documents(query)


# %%


parameter_dict = {'shape': 'metadata.system.surfactant.aggregates.shape'}
parameter_keys = list(parameter_dict.keys())


# %%


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
    parameter_key = c["_id"][parameter_keys[0]]
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[parameter_key] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# %%


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


# %%


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


# %%




