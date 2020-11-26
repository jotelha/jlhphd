#!/usr/bin/env python
# coding: utf-8

# # Analyze AFM tip solvation

# This notebook demonstrates deposition of an SDS adsorption layer on a non-spherical AFM tip model.

# ## Initialization

# ### IPython magic

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[16]:


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

# In[6]:


# os.environ['GMXLIB']


# In[7]:


# pmd.gromacs.GROMACS_TOPDIR = os.environ['GMXLIB']
pmd.gromacs.GROMACS_TOPDIR = '/usr/share/gromacs/top'


# In[8]:


# prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'
prefix = '/home/jotelha/git/jlhphd'


# In[9]:


work_prefix = '/home/jotelha/tmp/20200421_fw/'


# In[10]:


try:
    os.mkdir(work_prefix)
except FileExistsError as exc:
    print(exc)


# In[ ]:


os.chdir(work_prefix)


# In[ ]:


# the FireWorks LaunchPad
lp = LaunchPad.auto_load() #Define the server and database
# FilePad behaves analogous to LaunchPad
fp = FilePad.auto_load()


# ## Conversion from LAMMPS data format to PDB

# The vollowing bash / tcl snippet converts a LAMMPS data file to PDB, assigning the desired names as mapped in a yaml file
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

# In[25]:


query = {'metadata.datetime': {'$gt': '2020'} }


# In[26]:


fp.filepad.count_documents(query)


# In[27]:


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


# In[28]:


res_df


# ### Overview on steps in project

# In[29]:


project_id = '2020-05-08-final'


# In[30]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[31]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[32]:


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


# In[33]:


res_df


# In[34]:


res_df['step'].values


# ### Overview on objects in project

# In[35]:


project_id = '2020-05-08-final'


# In[36]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[37]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[38]:


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


# In[39]:


res_df


# ### Overview on images by distinct steps

# In[40]:


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
}


# In[41]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[42]:


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


# In[43]:


res_df


# In[44]:


res_df["step"][0]


# ## Packing visualization

# ### Indenter bounding sphere

# In[45]:


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
    'metadata.step': {'$regex': 'IndenterBoundingSphere'}
}


# In[46]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[47]:


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


# In[49]:


for obj in obj_list:
    display(obj)


# ### Surfactant measures

# In[38]:


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
    'metadata.step': {'$regex': 'SurfactantMoleculeMeasures'}
}


# In[39]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[40]:


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


# In[41]:


obj_list[0]


# ### Packing constraints

# In[43]:


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
    'metadata.step': {'$regex': 'PackingConstraintSpheres'}
}


# In[44]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[45]:


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


# In[46]:


obj_list[0]


# ### Packed film

# In[50]:


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
    'metadata.step': {'$regex': 'SphericalSurfactantPacking'}
}


# In[51]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[52]:


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


# In[53]:


for obj in obj_list:
    display(obj)


# ## Energy minimization analysis

# ### Overview on objects in step

# In[54]:


project_id = '2020-05-08-final'


# In[55]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsEnergyMinimization:ProcessAnalyzeAndVisualize:push_filepad'
}


# In[56]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[57]:


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


# In[58]:


res_df


# ### Global observables

# In[59]:


query = { 
    "metadata.project": project_id,
    'metadata.step': 'GromacsEnergyMinimization:ProcessAnalyzeAndVisualize:push_filepad',  #{'$regex': 'GromacsEnergyMinimization'}
    "metadata.type": 'energy_file',
}
fp.filepad.count_documents(query)


# In[60]:


parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}


# In[61]:


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


# In[62]:


[ c for c in cursor]


# In[74]:


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


# In[75]:


res_mi_df


# In[76]:


res_df


# In[89]:


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
for key, grp in res_df.groupby(['nmolecules']):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ### Trajectory

# In[83]:


query = { 
    "metadata.project": project_id,
    "metadata.step": 'GromacsEnergyMinimization:ProcessAnalyzeAndVisualize:push_filepad',
    "metadata.type":  { '$in': ['trajectory_file','data_file'] },
}
fp.filepad.count_documents(query)


# In[84]:


# Building a rather sophisticated aggregation pipeline

# first group by nmolecules and type ...

parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules',
    'type': 'metadata.type'}
# parameter_names = ['nmolecules', 'type']

query = { 
    "metadata.project": project_id,
    "metadata.step": 'GromacsEnergyMinimization:ProcessAnalyzeAndVisualize:push_filepad',
    "metadata.type":  { '$in': ['trajectory_file','data_file'] },
}

aggregation_pipeline = []

aggregation_pipeline.append({ 
    "$match": query
})

aggregation_pipeline.append({ 
    "$sort": { 
        "metadata.system.surfactant.nmolecules": pymongo.ASCENDING,
        "metadata.datetime": pymongo.DESCENDING,
    }
})

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
})

# second group by nmolecules

parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules'}

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { field: '$_id.{}'.format(field) for field in parameter_dict.keys() },
        "type":     {"$addToSet": "$_id.type"},
        "gfs_id":   {"$addToSet": "$latest"} 
        #"$_id.type": "$latest"
    }
})

aggregation_pipeline.append({
    '$project': {
         '_id': False,
        **{field: '$_id.{}'.format(field) for field in parameter_dict.keys()},
        'objects': { 
            '$zip': {
                'inputs': [ '$type', '$gfs_id' ],
                'useLongestLength': True,
                'defaults':  [None,None]
            }
        }
    }
})

aggregation_pipeline.append({ 
    '$project': {
        **{p: True for p in parameter_dict.keys()},
        'objects': {'$arrayToObject': '$objects'}
        #'objects': False 
    }
})

aggregation_pipeline.append({ 
    '$addFields': {
        'objects': {**{ field: '${}'.format(field) for field in parameter_dict.keys()}}
    }
})

aggregation_pipeline.append({ 
    '$replaceRoot': { 'newRoot': '$objects' }
})

# display results with
# for i, c in enumerate(cursor): 
#    print(c)
# yields documents in the form
# {'em_gro': '5e6a4e3d6c26f976ceae5e38', 'em_trr': '5e6a4e3a6c26f976ceae5e14', 'nmolecules': '44'}
# i.e. most recent topology file and trajectory file per concentration

cursor = fp.filepad.aggregate(aggregation_pipeline)

mda_trr_list = []
for i, c in enumerate(cursor): 
    em_gro_content, _ = fp.get_file_by_id(c["data_file"])
    em_trr_content, _ = fp.get_file_by_id(c["trajectory_file"])
    # Stream approach won't work
    # with io.TextIOWrapper( io.BytesIO(em_gro_content) ) as gro, \
    #    io.BytesIO(em_trr_content) as trr:   
        #mda_trr_list.append( 
        #    mda.Universe( 
        #        gro,trr, topology_format = 'GRO', format='TRR') )
    with tempfile.NamedTemporaryFile(suffix='.gro') as gro,         tempfile.NamedTemporaryFile(suffix='.trr') as trr:
        gro.write(em_gro_content)
        trr.write(em_trr_content)
        mda_trr_list.append( mda.Universe(gro.name,trr.name) )
    print('.',end='')
print('')
del em_gro_content
del em_trr_content


# In[85]:


mda_trr = mda_trr_list[0]

mda_view = nglview.show_mdanalysis(mda_trr)

mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation('ball+stick')
mda_view


# ###  Rendered videos

# In[86]:


project_id = '2020-05-08-final'


# In[89]:


query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsEnergyMinimization:ProcessAnalyzeAndVisualize:push_filepad',
    'metadata.type': 'mp4_file',
}


# In[90]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[91]:


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


# In[92]:


res_df


# In[103]:


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
    with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as tmp:
        tmp.write(content)
        # obj_list.append(Video(filename=tmp.name)) 
        # obj_list.append(tmp.name)
        obj_list.append(Video.from_file(tmp.name))
    print('.',end='')


# In[105]:


obj_list[0]


# ## Pulling analysis

# ### Overview on objects in step

# In[94]:


project_id = '2020-05-08-final'


# In[95]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsPull:ProcessAnalyzeAndVisualize:push_filepad'
}


# In[96]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[97]:


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


# In[98]:


res_df


# ### Global observables

# The `gmx energy` table:

# ```
#   1  Restraint-Pot.   2  U-B              3  Proper-Dih.      4  LJ-14         
#   5  Coulomb-14       6  LJ-(SR)          7  Coulomb-(SR)     8  Coul.-recip.  
#   9  Position-Rest.  10  COM-Pull-En.    11  Potential       12  Kinetic-En.   
#  13  Total-Energy    14  Temperature     15  Pressure        16  Constr.-rmsd  
#  17  Vir-XX          18  Vir-XY          19  Vir-XZ          20  Vir-YX        
#  21  Vir-YY          22  Vir-YZ          23  Vir-ZX          24  Vir-ZY        
#  25  Vir-ZZ          26  Pres-XX         27  Pres-XY         28  Pres-XZ       
#  29  Pres-YX         30  Pres-YY         31  Pres-YZ         32  Pres-ZX       
#  33  Pres-ZY         34  Pres-ZZ         35  #Surf*SurfTen   36  T-rest       
#  ```
#  converted to dict with regex
#  
#      \s+([0-9]+)\s+([^\s]+)
#  
#  and replacement
#  
#      '$2': $1,\n

# In[99]:


gmx_energy_dict = {
    'Restraint-Pot.': 1,
    'U-B': 2,
    'Proper-Dih.': 3,
    'LJ-14': 4,
    'Coulomb-14': 5,
    'LJ-(SR)': 6,
    'Coulomb-(SR)': 7,
    'Coul.-recip.': 8,
    'Position-Rest.': 9,
    'COM-Pull-En.': 10,
    'Potential': 11,
    'Kinetic-En.': 12,
    'Total-Energy': 13,
    'Temperature': 14,
    'Pressure': 15,
    'Constr.-rmsd': 16,
    'Vir-XX': 17,
    'Vir-XY': 18,
    'Vir-XZ': 19,
    'Vir-YX': 20,
    'Vir-YY': 21,
    'Vir-YZ': 22,
    'Vir-ZX': 23,
    'Vir-ZY': 24,
    'Vir-ZZ': 25,
    'Pres-XX': 26,
    'Pres-XY': 27,
    'Pres-XZ': 28,
    'Pres-YX': 29,
    'Pres-YY': 30,
    'Pres-YZ': 31,
    'Pres-ZX': 32,
    'Pres-ZY': 33,
    'Pres-ZZ': 34,
    '#Surf*SurfTen': 35,
    'T-rest': 36,
}


# In[100]:


project_id = '2020-05-08-final'


# In[101]:


query = { 
    "metadata.project": project_id,
    "metadata.type":    'energy_file',
    "metadata.step": "GromacsPull:ProcessAnalyzeAndVisualize:push_filepad",
}

fp.filepad.count_documents(query)


# In[102]:


parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules'}


# In[103]:


gmx_energy_selection = [
    'Restraint-Pot.',
    'Position-Rest.',
    'COM-Pull-En.',
    'Potential',
    'Kinetic-En.',
    'Total-Energy',
    'Temperature',
    'Pressure',
    'Constr.-rmsd',
]


# In[116]:


res_list = []
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

res_df_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    nmolecules = int(c["_id"]["nmolecules"])
    
    #df = panedr.edr_to_df(tmp.name), fails
    tmpin = tempfile.NamedTemporaryFile(mode='w+b',suffix='.edr', delete=False)
    
    # cur_res_dict = {}
    with tmpin:
        tmpin.write(content)
        #tmpin.seek(0)
       
    res_df = None
    for sel in gmx_energy_selection:  
        try:
            tmpout = tempfile.NamedTemporaryFile(suffix='.xvg', delete=False)
            res = gromacs.energy(f=tmpin.name,o=tmpout.name,
                                 input=str(gmx_energy_dict[sel]))
            #with open(tmpout.name,'r') as f:
            #    xvg = f.read()
            #tmpout.delete()
            xvg = mda.auxiliary.XVG.XVGReader(tmpout.name)
            xvg_time = xvg.read_all_times()
            xvg_data = np.array([ f.data[1:] for f in xvg ]).flatten() # 1st entry contains times
            os.unlink(tmpout.name)
        except: 
            logger.warning("Failed to read '{:s}' from data set {:d}.".format(sel,i))
            failed_list.append((nmolecules, sel))
        else:
            r = {'nmolecules': [nmolecules]*len(xvg_time), 'time': xvg_time, sel: xvg_data}
            cur_df = pd.DataFrame(r)
            if res_df is None:
                res_df = cur_df
            else:
                res_df = pd.merge(res_df, cur_df, how='outer', on=['nmolecules', 'time'])
    res_df_list.append(res_df)
    os.unlink(tmpin.name)
    print('.',end='')
print('')
res_df = pd.concat(res_df_list)
res_df_mi = res_df.set_index(['nmolecules','time'])


# In[122]:


res_mi_df


# In[121]:


cols = 2
y_quantities = [
    'Restraint-Pot.',
    'Position-Rest.',
    'COM-Pull-En.',
    'Potential',
    'Kinetic-En.',
    'Total-Energy',
    'Temperature',
    'Pressure',
    'Constr.-rmsd',
    ]
n = len(y_quantities)
rows = round(n/cols)
positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for key, grp in res_df.groupby(['nmolecules']):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ### Pulling forces

# In[130]:


res_df_list = []
failed_list = []

query = { 
    "metadata.project": project_id,
    "metadata.step": "GromacsPull:ProcessAnalyzeAndVisualize:push_filepad",
    "metadata.type": 'pullf_file',
}

fp.filepad.count_documents(query)
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

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    nmolecules = int(c["_id"]["nmolecules"])
    
    tmpin = tempfile.NamedTemporaryFile(mode='w+b',suffix='.xvg', delete=False)
    
    with tmpin:
        tmpin.write(content)
        
    try:
        xvg = mda.auxiliary.XVG.XVGReader(tmpin.name)
        xvg_time = xvg.read_all_times()
        xvg_data = np.array([ f.data[1:] for f in xvg ])# .flatten() # 1st entry contains times
    except: 
        logger.warning("Failed to read data set {:d}.".format(i))
        failed_list.append(nmolecules)
    else:
        res_df_list.append(pd.DataFrame({
            'nmolecules': np.array([nmolecules]*len(xvg_time), dtype=int),
            'time': xvg_time, 
            **{i: xvg_data[:,i] for i in range(nmolecules)}
        }))
    os.unlink(tmpin.name)
    print('.',end='')
print('')
res_df = pd.concat(res_df_list)
res_df_mi = res_df.set_index(['nmolecules','time'])


# In[166]:


n = len(res_df['nmolecules'].unique())
cols = 2
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for pos, (key, grp) in zip(positions,res_df.groupby(['nmolecules'])):
    columns = list(set(grp.columns) - set(['nmolecules','time']))
    grp.plot('time', columns, ax=ax[pos],title=key,legend=None)
fig.tight_layout()


# In[176]:


fig, ax = plt.subplots(1,1,figsize=(5,4))
for key, grp in res_df.groupby(['nmolecules']):
    columns = list(set(grp.columns) - set(['nmolecules','time']))
    grp = grp.set_index('time')
    grp = grp.drop(columns='nmolecules')
    grp.mean(axis=1).plot(legend=True, label=key, ax=ax)
fig.tight_layout()
#fig.legend()


# ### Pulling groups movement

# In[200]:


res_df_list = []
failed_list = []

query = { 
    "metadata.project": project_id,
    "metadata.step": "GromacsPull:ProcessAnalyzeAndVisualize:push_filepad",
    "metadata.type":    'pullx_file',
}

fp.filepad.count_documents(query)
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


for i, c in enumerate(cursor): 
    print(c["_id"])
    content, metadata = fp.get_file_by_id(c["latest"])
    nmolecules = int(c["_id"]["nmolecules"])  # int(metadata["metadata"]["nmolecules"])
    
    tmpin = tempfile.NamedTemporaryFile(mode='w+b',suffix='.xvg', delete=False)
    
    with tmpin:
        tmpin.write(content)
        
    try:
        xvg = gromacs.fileformats.XVG(tmpin.name)
        xvg_time = xvg.array[0,:]
        
        #xvg_labels = ['1', '1 ref', '1 dX', '1 dY', '1 dZ', '1 g 1 X', '1 g 1 Y', '1 g 1 Z', '1 g 2 X', '1 g 2 Y', '1 g 2 Z']
        N_pull_coords = nmolecules
        N_cols = len(xvg.names)
        N_cols_per_coord = int(N_cols / N_pull_coords)
        
        xvg_labels = xvg.names[:N_cols_per_coord]
        xvg_data = {}
        for j in range(N_pull_coords):
            for k in range(N_cols_per_coord):
                xvg_data[(j,xvg_labels[k])] = xvg.array[
                    1+j*N_cols_per_coord+k,:]
        
    except: 
        logger.exception("Failed to read data set {:d}.".format(i))
        failed_list.append(nmolecules)

    else:
        # res_list.append({
        #    'nmolecules': nmolecules, # np.array([nmolecules]*len(xvg_time), dtype=int),
        #    'time': xvg_time, 
        #    **xvg_data})
        res_df_list.append(pd.DataFrame({
            'nmolecules': np.array([nmolecules]*len(xvg_time), dtype=int),
            'time': xvg_time, 
            **xvg_data # {i: xvg_data[:,i] for i in range(nmolecules)}
        }))
    #os.unlink(tmpin.name)
    #print('.',end='')
#print('')
res_df = pd.concat(res_df_list)
res_df_mi = res_df.set_index(['nmolecules','time'])
res_df_mi.columns = pd.MultiIndex.from_tuples(res_df_mi.columns, names=['nmolecule', 'coord'])


# In[201]:


res_df_mi


# In[205]:


res_df = res_df_mi.groupby(axis=1,level='coord').mean().reset_index()


# In[206]:


res_df


# In[207]:


cols = 2
y_quantities = [
    '1', 
    '1 ref', 
    '1 dX', 
    '1 dY', 
    '1 dZ', 
    '1 g 1 X', 
    '1 g 1 Y', 
    '1 g 1 Z', 
    '1 g 2 X', 
    '1 g 2 Y', 
    '1 g 2 Z'
]
n = len(y_quantities)
rows = round(n/cols)
positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
for key, grp in res_df.groupby(['nmolecules']):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ### Visualize trajectory

# In[138]:


parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules',
    'type': 'metadata.type'}

query = { 
    "metadata.project": project_id,
    "metadata.step": "GromacsPull:ProcessAnalyzeAndVisualize:push_filepad",
    "metadata.type": { '$in': ['trajectory_file','data_file'] },
}

aggregation_pipeline = []

aggregation_pipeline.append({ 
    "$match": query
})



aggregation_pipeline.append({ 
    "$sort": { 
        "metadata.system.surfactant.nmolecules": pymongo.ASCENDING,
        "metadata.datetime": pymongo.DESCENDING,
    }
})

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
})

parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules'}

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { p: '$_id.{}'.format(p) for p in parameter_dict.keys() },
        "type":     {"$addToSet": "$_id.type"},
        "gfs_id":   {"$addToSet": "$latest"} 
        #"$_id.type": "$latest"
    }
})

aggregation_pipeline.append({
    '$project': {
         '_id': False,
        **{ p: '$_id.{}'.format(p) for p in parameter_dict.keys()},
        'objects': { 
            '$zip': {
                'inputs': [ '$type', '$gfs_id' ],
                'useLongestLength': True,
                'defaults':  [None,None]
            }
        }
    }
})

aggregation_pipeline.append({ 
    '$project': {
        **{ p: True for p in parameter_dict.keys()},
        'objects': {'$arrayToObject': '$objects'}
        #'objects': False 
    }
})

aggregation_pipeline.append({ 
    '$addFields': {
        'objects': {**{ p: '${}'.format(p) for p in parameter_dict.keys()}}
    }
})

aggregation_pipeline.append({ 
    '$replaceRoot': { 'newRoot': '$objects' }
})

# display results with
# for i, c in enumerate(cursor): 
#    print(c)
# yields documents in the form
# {'em_gro': '5e6a4e3d6c26f976ceae5e38', 'em_trr': '5e6a4e3a6c26f976ceae5e14', 'nmolecules': '44'}
# i.e. most recent topology file and trajectory file per concentration

cursor = fp.filepad.aggregate(aggregation_pipeline)


# In[139]:


failed_list = []
mda_trr_list = []
for i, c in enumerate(cursor): 
    try:
        gro_content, _ = fp.get_file_by_id(c["data_file"])
        trr_content, _ = fp.get_file_by_id(c["trajectory_file"])
        with tempfile.NamedTemporaryFile(suffix='.gro') as gro,             tempfile.NamedTemporaryFile(suffix='.trr') as trr:
            gro.write(gro_content)
            trr.write(trr_content)
            mda_trr_list.append( mda.Universe(gro.name,trr.name) )
    except: 
        logger.exception("Failed to read data set {}.".format(c))
        failed_list.append(c)
    print('.',end='')
print('')


# In[140]:


failed_list


# In[141]:


mda_trr = mda_trr_list[0]

mda_view = nglview.show_mdanalysis(mda_trr)

mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation('ball+stick')
mda_view


# ###  Rendered videos

# In[220]:


project_id = '2020-05-08-final'


# In[221]:


query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsPull:ProcessAnalyzeAndVisualize:push_filepad',
    'metadata.type': 'mp4_file',
}


# In[222]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[223]:


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


# In[224]:


res_df


# In[225]:


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
    with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as tmp:
        tmp.write(content)
        # obj_list.append(Video(filename=tmp.name)) 
        # obj_list.append(tmp.name)
        obj_list.append(Video.from_file(tmp.name))
    print('.',end='')


# In[226]:


for obj in obj_list:
    display(obj)


# ### MSD & RDF

# In[208]:


# reuse previous aggregation pipeline
cursor = fp.filepad.aggregate(aggregation_pipeline)

failed_list = []
# mda_trr_list = []
rms_substrate_list = []
rms_surfactant_head_list = []
rdf_substrate_headgroup_list = []


for i, c in enumerate(cursor): 
    try:
        gro_content, _ = fp.get_file_by_id(c["data_file"])
        trr_content, _ = fp.get_file_by_id(c["trajectory_file"])
        with tempfile.NamedTemporaryFile(suffix='.gro') as gro,             tempfile.NamedTemporaryFile(suffix='.trr') as trr:
            gro.write(gro_content)
            trr.write(trr_content)
            
            mda_trr = mda.Universe(gro.name,trr.name)
            substrate = mda_trr.atoms[mda_trr.atoms.names == 'AU']
            surfactant_head = mda_trr.atoms[mda_trr.atoms.names == 'S']
            
            rms_substrate = mda_rms.RMSD(substrate,ref_frame=0)
            rms_substrate.run()
            
            rms_surfactant_head = mda_rms.RMSD(surfactant_head,ref_frame=0)
            rms_surfactant_head.run()
            
            rdf_substrate_headgroup = mda_rdf.InterRDF(
                substrate,surfactant_head,range=(0.0,80.0),verbose=True)
            
            bins = []
            rdf  = []
            for i in range(len(mda_trr.trajectory)):
                rdf_substrate_headgroup = mda_rdf.InterRDF(
                    substrate,surfactant_head,range=(0.0,80.0),verbose=True)
                rdf_substrate_headgroup.run(start=i,stop=i+1)
                bins.append(rdf_substrate_headgroup.bins.copy())
                rdf.append(rdf_substrate_headgroup.rdf.copy())
            bins = np.array(bins)
            rdf = np.array(rdf)
            
            rms_substrate_list.append(rms_substrate)
            rms_surfactant_head_list.append(rms_surfactant_head)
            rdf_substrate_headgroup_list.append(rms_surfactant_head)
            
    except: 
        logger.exception("Failed to read data set {}.".format(c))
        failed_list.append(c)
    print('.',end='')
print('')


# #### Substrate MSD (none)

# In[145]:


rmsd = rms_substrate_list[0].rmsd.T   # transpose makes it easier for plotting
time = rmsd[1]


# In[226]:


rdf_substrate_headgroup.n_frames


# In[227]:


plt.plot(time,rmsd[2])


# #### Surfactant head MSD

# In[148]:


rmsd = rms_surfactant_head_list[0].rmsd.T   # transpose makes it easier for plotting
time = rmsd[1]


# In[149]:


plt.plot(time,rmsd[2])


# #### Au-S (substrate - head group ) RDF

# In[151]:


# indicates desired approach towards substrate
plt.plot(bins[0],rdf[0],label="Initial RDF")
plt.plot(bins[3],rdf[4],label="Intermediat RDF")
plt.plot(bins[-1],rdf[-1],label="Final RDF")
plt.legend()


# 
# ### Pre-evaluated RDF

# #### Overview

# In[209]:


query = { 
    "metadata.project": project_id,
    "metadata.type": {'$regex': '.*rdf$'},
    "metadata.step": "GromacsPull:ProcessAnalyzeAndVisualize:push_filepad",
}

fp.filepad.count_documents(query)


# In[210]:


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


# In[211]:


res_df


# #### Substrate - surfactant head RDF

# In[212]:


parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules'}


# In[213]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_head_rdf',
    "metadata.step": "GromacsPull:ProcessAnalyzeAndVisualize:push_filepad",
}

fp.filepad.count_documents(query)


# In[215]:


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
    nmolecules = int(c["_id"]["nmolecules"])
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    d = data[0] # distance bins
    rdf = data[1:]
    res_dict[nmolecules] = {'dist': d, 'rdf': rdf}
    # res_list.append(data)
    print('.',end='')
print('')


# In[219]:


cols = 2
n = len(res_dict)
rows = round(n/cols)
if rows > 1:
    positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
else:
    positions = [i for i in range(cols)][:n]
    
fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
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


# In[177]:


t = data[0]


# In[189]:


for nmolecules, data in res_dict.items():
    plt.plot(data[0],data[1],label='First frame RDF')
    plt.plot(data[0],data[len(data)//2],label='Intermediate frame RDF')
    plt.plot(data[0],data[-1],label='Last frame RDF')
    plt.legend()
    plt.show()


# #### Substrate - surfactant tail RDF

# In[190]:


parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules'}


# In[195]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'substrate_surfactant_tail_rdf',
    "metadata.step": "GromacsPull:ProcessAnalyzeAndVisualize:push_filepad",
}

fp.filepad.count_documents(query)


# In[196]:


res_list = []
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

res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    nmolecules = int(c["_id"]["nmolecules"])
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    res_list.append(data)
    print('.',end='')
print('')


# In[197]:


t = data[0]


# In[198]:


plt.plot(data[0],data[1],label='First frame RDF')
plt.plot(data[0],data[len(data)//2],label='Intermediate frame RDF')
plt.plot(data[0],data[-1],label='Last frame RDF')
plt.legend()
plt.show()


# #### Surfactant head - surfactant tail RDF

# In[199]:


parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules'}


# In[200]:


query = { 
    "metadata.project": project_id,
    "metadata.type": 'surfactant_head_surfactant_tail_rdf',
    "metadata.step": "GromacsPull:ProcessAnalyzeAndVisualize:push_filepad",
}

fp.filepad.count_documents(query)


# In[201]:


res_list = []
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

res_list = []
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    nmolecules = int(c["_id"]["nmolecules"])
    data_str = io.StringIO(content.decode())
    data = np.loadtxt(data_str, comments='#')
    res_list.append(data)
    print('.',end='')
print('')


# In[202]:


plt.plot(data[0],data[1],label='First frame RDF')
plt.plot(data[0],data[len(data)//2],label='Intermediate frame RDF')
plt.plot(data[0],data[-1],label='Last frame RDF')
plt.legend()
plt.show()


# ## Energy minimization after solvation analysis

# ### Overview on objects in step

# In[237]:


project_id = '2020-05-08-final'


# In[238]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsEnergyMinimizationAfterSolvation:ProcessAnalyzeAndVisualize:push_filepad'
}


# In[239]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[240]:


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


# In[241]:


res_df


# ### Global observables

# In[242]:


query = { 
    "metadata.project": project_id,
    'metadata.step': 'GromacsEnergyMinimizationAfterSolvation:ProcessAnalyzeAndVisualize:push_filepad',  #{'$regex': 'GromacsEnergyMinimization'}
    "metadata.type": 'energy_file',
}
fp.filepad.count_documents(query)


# In[243]:


parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}


# In[244]:


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


# In[245]:


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


# In[246]:


res_mi_df


# In[247]:


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
for key, grp in res_df.groupby(['nmolecules']):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ###  Rendered videos

# In[260]:


project_id = '2020-05-08-final'


# In[261]:


query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsEnergyMinimizationAfterSolvation:ProcessAnalyzeAndVisualize:push_filepad',
    'metadata.type': 'mp4_file',
}


# In[262]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[266]:


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


# In[267]:


res_df


# In[270]:


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

obj_dict = {}
for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    # print(metadata['metadata'])
    with tempfile.NamedTemporaryFile(suffix='.mp4',delete=False) as tmp:
        tmp.write(content)
        # obj_list.append(Video(filename=tmp.name)) 
        # obj_list.append(tmp.name)
        obj_dict.update({metadata['metadata']['system']['surfactant']['nmolecules']: Video.from_file(tmp.name)})
    print('.',end='')


# In[271]:


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# ### Trajectory

# In[243]:


query = { 
    "metadata.project": project_id,
    'metadata.step': 'GromacsEnergyMinimizationAfterSolvation:ProcessAnalyzeAndVisualize:push_filepad',
    "metadata.type":    { '$in': ['trajectory_file','data_file'] },
}
fp.filepad.count_documents(query)


# In[244]:


# Building a rather sophisticated aggregation pipeline

# first group by nmolecules and type ...

parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules',
    'type': 'metadata.type'}
# parameter_names = ['nmolecules', 'type']

query = { 
    "metadata.project": project_id,
    "metadata.type":    { '$in': ['trajectory_file','data_file'] },
}

aggregation_pipeline = []

aggregation_pipeline.append({ 
    "$match": query
})

aggregation_pipeline.append({ 
    "$sort": { 
        "metadata.system.surfactant.nmolecules": pymongo.ASCENDING,
        "metadata.datetime": pymongo.DESCENDING,
    }
})

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { field: '${}'.format(key) for field, key in parameter_dict.items() },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
})

# second group by nmolecules

parameter_dict = {
    'nmolecules': 'metadata.system.surfactant.nmolecules'}

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { field: '$_id.{}'.format(field) for field in parameter_dict.keys() },
        "type":     {"$addToSet": "$_id.type"},
        "gfs_id":   {"$addToSet": "$latest"} 
        #"$_id.type": "$latest"
    }
})

aggregation_pipeline.append({
    '$project': {
         '_id': False,
        **{field: '$_id.{}'.format(field) for field in parameter_dict.keys()},
        'objects': { 
            '$zip': {
                'inputs': [ '$type', '$gfs_id' ],
                'useLongestLength': True,
                'defaults':  [None,None]
            }
        }
    }
})

aggregation_pipeline.append({ 
    '$project': {
        **{p: True for p in parameter_dict.keys()},
        'objects': {'$arrayToObject': '$objects'}
        #'objects': False 
    }
})

aggregation_pipeline.append({ 
    '$addFields': {
        'objects': {**{ field: '${}'.format(field) for field in parameter_dict.keys()}}
    }
})

aggregation_pipeline.append({ 
    '$replaceRoot': { 'newRoot': '$objects' }
})

# display results with
# for i, c in enumerate(cursor): 
#    print(c)
# yields documents in the form
# {'em_solvated_gro': '5e6a4e3d6c26f976ceae5e38', 'em_solvated_trr': '5e6a4e3a6c26f976ceae5e14', 'nmolecules': '44'}
# i.e. most recent topology file and trajectory file per concentration

cursor = fp.filepad.aggregate(aggregation_pipeline)

mda_trr_list = []
for i, c in enumerate(cursor): 
    em_gro_content, _ = fp.get_file_by_id(c["data_file"])
    em_trr_content, _ = fp.get_file_by_id(c["trajectory_file"])
    # Stream approach won't work
    # with io.TextIOWrapper( io.BytesIO(em_gro_content) ) as gro, \
    #    io.BytesIO(em_trr_content) as trr:   
        #mda_trr_list.append( 
        #    mda.Universe( 
        #        gro,trr, topology_format = 'GRO', format='TRR') )
    with tempfile.NamedTemporaryFile(suffix='.gro') as gro,         tempfile.NamedTemporaryFile(suffix='.trr') as trr:
        gro.write(em_gro_content)
        trr.write(em_trr_content)
        mda_trr_list.append( mda.Universe(gro.name,trr.name) )
    print('.',end='')
print('')


# In[245]:


mda_trr = mda_trr_list[0]
# check unique resiude names in system
resnames = np.unique([ r.resname for r in mda_trr.residues ])


# In[246]:


resnames


# In[247]:


mda_view = nglview.show_mdanalysis(mda_trr)

# setSize: https://cloud.githubusercontent.com/assets/22888066/21761120/e47e2988-d68b-11e6-9e11-a894d7833d50.png
mda_view._remote_call('setSize', target='Widget', args=['600px','600px'])

mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation(repr_type='ball+stick',selection='SDS')
mda_view.add_representation(repr_type='ball+stick',selection='NA')
mda_view.add_representation(repr_type='spacefill',selection='AUM',color='yellow')
mda_view.center()
mda_view


# #### Render images

# In[248]:


len(mda_trr.trajectory)


# In[68]:


# https://ambermd.org/tutorials/analysis/tutorial_notebooks/nglview_movie/index.html
    
# make sure to change your web browser option to save files as default (vs open file by external program)
# NGLView will render each snapshot and save image to your web browser default download location
# uncomment all the commands below to render

from time import sleep

# # to save time for this tutorial, we make a movie with only 50 frames
for frame in range(0, len(mda_trr.trajectory)):
     # set frame to update coordinates
    mda_view.frame = frame
    # make sure to let NGL spending enough time to update coordinates
    sleep(0.5)
    mda_view.download_image(filename='frame{:04d}.png'.format(frame))
    # make sure to let NGL spending enough time to render before going to next frame
    sleep(2.0)


# In[ ]:


# ffmpeg -r 20 -f image2 -i frame%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p out.mp4


# #### Render gif

# In[44]:


from nglview.contrib.movie import MovieMaker


# In[50]:


output_folder = os.path.join(work_prefix,'mov')


# In[52]:


try:
    os.mkdir(output_folder)
except FileExistsError as exc:
    print(exc)


# In[64]:


mov = MovieMaker(
    mda_view, download_folder=output_folder, output="em_solvated.gif")


# In[65]:


mov.make()


# #### Make avi

# In[66]:


# write avi format
from nglview.contrib.movie import MovieMaker
moviepy_params = {
     'codec': 'mpeg4'
}
movie = MovieMaker(mda_view, output='em_solvated.avi', in_memory=True, moviepy_params=moviepy_params)
movie.make()


# Again, relax the system a little with positional constraints applied to all ions.

# ## NVT equilibration analysis

# ### Overview on objects in step

# In[272]:


project_id = '2020-05-08-final'


# In[273]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsNVTEquilibration:ProcessAnalyzeAndVisualize:push_filepad'
}


# In[274]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[275]:


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


# In[276]:


res_df


# ### Global observables

# In[278]:


query = { 
    "metadata.project": project_id,
    'metadata.step': 'GromacsNVTEquilibration:ProcessAnalyzeAndVisualize:push_filepad',
    "metadata.type":    'energy_file',
}
fp.filepad.count_documents(query)


# In[279]:


parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}


# In[280]:


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


# In[281]:


[ c for c in cursor]


# In[289]:


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


# In[290]:


res_df.columns


# In[307]:


res_df_mi


# In[293]:


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
for key, grp in res_df.groupby(['nmolecules']):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ### Visualize trajectory

# In[294]:


query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsNVTEquilibration:ProcessAnalyzeAndVisualize:push_filepad',
    'metadata.type': 'mp4_file',
}


# In[295]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[303]:


parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}

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


# In[306]:


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# ## NPT equilibration analysis

# ### Overview on objects in step

# In[308]:


project_id = '2020-05-08-final'


# In[309]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsNPTEquilibration:ProcessAnalyzeAndVisualize:push_filepad'
}


# In[310]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[311]:


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


# In[312]:


res_df


# ### Global observables

# In[313]:


query = { 
    "metadata.project": project_id,
    'metadata.step': 'GromacsNPTEquilibration:ProcessAnalyzeAndVisualize:push_filepad',
    "metadata.type":    'energy_file',
}
fp.filepad.count_documents(query)


# In[314]:


parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}


# In[315]:


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


# In[316]:


[ c for c in cursor]


# In[317]:


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


# In[318]:


res_df.columns


# In[319]:


res_df_mi


# In[320]:


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
for key, grp in res_df.groupby(['nmolecules']):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()


# ### Visualize trajectory

# In[325]:


query = {
    'metadata.project': project_id,
    'metadata.step': 'GromacsNPTEquilibration:ProcessAnalyzeAndVisualize:push_filepad',
    'metadata.type': 'mp4_file',
}


# In[326]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[327]:


parameter_dict = {'nmolecules': 'metadata.system.surfactant.nmolecules'}

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


# In[328]:


for key, obj in obj_dict.items():
    print(key)
    display(obj)


# ### Global observables

# ### Visualize trajectory

# In[61]:


mda_trr = mda.Universe('nvt.gro','npt.trr')


# In[62]:


# check unique resiude names in system
resnames = np.unique([ r.resname for r in mda_trr.residues ])


# In[63]:


resnames


# In[64]:


mda_view = nglview.show_mdanalysis(mda_trr)
mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation(repr_type='ball+stick',selection='SDS')
mda_view.add_representation(repr_type='ball+stick',selection='NA')
mda_view.add_representation(repr_type='spacefill',selection='AUM',color='yellow')
mda_view.center()


# In[65]:


mda_view


# In[67]:


substrate = mda_trr.select_atoms('resname AUM')


# In[103]:


substrate.masses= ase.data.atomic_masses[ase.data.atomic_numbers['Au']]


# In[105]:


substrtate_com_traj = np.array([substrate.center_of_mass() for ts in mda_trr.trajectory ])


# In[106]:


substrtate_rgyr_traj = np.array([substrate.radius_of_gyration() for ts in mda_trr.trajectory ])


# In[107]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d',azim=-30)
ax.plot(*substrtate_com_traj.T)
ax.scatter(*substrtate_com_traj[0,:],color='green')
ax.scatter(*substrtate_com_traj[-1,:],color='red')


# In[109]:


plt.plot(substrtate_rgyr_traj)


# In[119]:


try:
    del mda_trr
except:
    pass
try:
    del mda_view
except:
    pass

