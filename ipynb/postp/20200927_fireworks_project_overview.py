#!/usr/bin/env python
# coding: utf-8

# # Fireworks overview

# This notebook demonstrates querying of Fireworks workflows and Filepad objects

# ## Initialization

# ### IPython magic

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


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

# In[152]:


def highlight_bool(s):
    """color boolean values in pandas dataframe"""
    return ['background-color: green' if v else 'background-color: red' for v in s]


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

# In[51]:


# pandas settings
# https://songhuiming.github.io/pages/2017/04/02/jupyter-and-pandas-display/
pd.options.display.max_rows = 200
pd.options.display.max_columns = 16
pd.options.display.max_colwidth = 256
pd.options.display.max_colwidth = None


# In[52]:


os.environ['GMXLIB'] = '/gmx_top'


# In[53]:


# pmd.gromacs.GROMACS_TOPDIR = os.environ['GMXLIB']
pmd.gromacs.GROMACS_TOPDIR = '/gmx_top'


# In[54]:


# prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'
prefix = '/mnt/dat/work'


# In[55]:


work_prefix = '/mnt/dat/work/tmp'


# In[56]:


try:
    os.mkdir(work_prefix)
except FileExistsError as exc:
    print(exc)


# In[57]:


os.chdir(work_prefix)


# In[58]:


# the FireWorks LaunchPad
lp = LaunchPad.auto_load() #Define the server and database
# FilePad behaves analogous to LaunchPad
fp = FilePad.auto_load()


# # Fireworks

# In[13]:


project = '2020-09-28-ctab-on-au-111-substrate-passivation'


# In[14]:


query={'spec.metadata.project': project}


# In[15]:


fw_ids = lp.get_fw_ids(query)


# In[16]:


len(fw_ids)


# In[17]:


wf_query = {'nodes': {'$in': fw_ids}}


# In[18]:


lp.workflows.count_documents(wf_query)


# In[19]:


wf = lp.workflows.find_one(wf_query)


# In[20]:


wf.keys()


# In[21]:


fw_ids = wf['nodes']


# In[22]:


fw = lp.fireworks.find_one()


# In[23]:


fw


# In[24]:


query = {'fw_id': {'$in': fw_ids}}


# In[25]:


lp.fireworks.count_documents(query)


# In[26]:


query = {'fw_id': {'$in': fw_ids}, 'name': {'$regex':'NPT'}}


# In[27]:


lp.fireworks.count_documents(query)


# In[28]:


query = {'fw_id': {'$in': fw_ids}, 'name': {'$regex':'NPT.*mdrun'}, 'state': 'COMPLETED'}


# In[29]:


lp.fireworks.count_documents(query)


# In[30]:


fw = lp.fireworks.find_one(query)


# In[31]:


fw['fw_id']


# In[32]:


fw['name']


# In[33]:


fw['state']


# In[34]:


fw['spec']['metadata']['step_specific']


# In[35]:


fw['spec']['metadata']['step_specific']


# # Filepad

# ## Overview

# ### Overview on recent projects in database

# In[36]:


query = {'metadata.datetime': {'$gt': '2020'} }


# In[37]:


fp.filepad.count_documents(query)


# In[38]:


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


# In[59]:


res_df


# In[ ]:





# ### Overview on recent production projects in database

# In[42]:


query = {
    'metadata.datetime': {'$gt': '2020'},
    'metadata.mode': 'production'
}


# In[43]:


fp.filepad.count_documents(query)


# In[44]:


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


# In[45]:


res_df


# ### Overview on steps in project

# In[46]:


#project_id = '2020-09-28-ctab-on-au-111-substrate-passivation'
project_id = '2020-10-13-ctab-on-au-111-substrate-passivation'


# In[47]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[48]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[49]:


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


# In[60]:


res_df


# #### Pivot overview on steps and parameters in project

# In[64]:


project_id = '2020-10-13-ctab-on-au-111-substrate-passivation'


# In[85]:


query = {
    'metadata.project': project_id,
    'metadata.system.surfactant.nmolecules': {'$exists': True},
    'metadata.system.surfactant.aggregates.shape': {'$exists': True},
}


# In[87]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[95]:


parameters = { 
    'nmolecules': 'metadata.system.surfactant.nmolecules',
    'shape': 'metadata.system.surfactant.aggregates.shape',
}


# In[97]:


distinct_parameter_values = {}
for label, key in parameters.items():
    values = fp.filepad.distinct(key, query)
    if None in values:
        values.remove(None)
    distinct_parameter_values[label] = values


# In[153]:


print(distinct_parameter_values)


# #### Refined aggregation for hemicylindrical systems

# In[163]:


distinct_parameter_values['shape'].remove('bilayer')


# In[164]:


query = {
    'metadata.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}


# In[165]:


print(query)


# In[166]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$metadata.step',
                **{label: '${}'.format(key) for label, key in parameters.items()},
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
columns = ['step', *parameters.keys(), 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[167]:


res_pivot = res_df.pivot_table(values='object_count', index=['step'], columns=list(parameters.keys()), aggfunc=pd.notna, fill_value=False)
res_pivot.style.apply(highlight_bool)


# #### Refined aggregation for bilayer systems

# In[173]:


distinct_parameter_values['shape'].remove('cylinders')
distinct_parameter_values['shape'].append('bilayer')


# In[174]:


query = {
    'metadata.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}


# In[175]:


print(query)


# In[176]:


# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$metadata.step',
                **{label: '${}'.format(key) for label, key in parameters.items()},
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
columns = ['step', *parameters.keys(), 'earliest', 'latest', 'object_count', '_id']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
del res_df["_id"]


# In[177]:


res_pivot = res_df.pivot_table(values='object_count', index=['step'], columns=list(parameters.keys()), aggfunc=pd.notna, fill_value=False)
res_pivot.style.apply(highlight_bool)


# In[134]:


res_df.groupby(['step', *parameters.keys()])


# In[131]:


res_df.set_index('step').stack()


# In[ ]:


(res_df.set_index('step').stack()
 .groupby(level=[0,1])
 .value_counts()
 .unstack(level=[1,2])
 .fillna(0)
 .sort_index(axis=1))


# In[129]:


res_df.groupby()


# In[123]:


parameters.keys()


# In[127]:


res_df.set_index(list(parameters.keys()))


# In[ ]:


pd.MultiIndex.from_frame()


# In[124]:



res_df.pivot(index='step', columns='shape', values='object_count')


# In[112]:


res_df.multiply(*parameters.keys())


# ### Overview on objects in project

# In[112]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[113]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[116]:


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
                #'step': '$metadata.step',
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


# In[117]:


res_df


# ### Overview on images by distinct steps

# In[99]:


query = {
    'metadata.project': project_id,
    'metadata.type': 'png_file',
}


# In[100]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[101]:


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


# In[102]:


res_df


# In[103]:


res_df["step"][0]


# ### Overview on objects in specific step

# In[126]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNPTEquilibration:push_filepad'}
}


# In[127]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[128]:


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


# In[129]:


res_df


# ### Overview on specific objects in specific steps

# In[131]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNPTEquilibration:push_filepad'},
    'metadata.type': 'log_file',
}


# In[132]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[133]:


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


# In[134]:


res_df


# ### Inspect specific file

# In[135]:


metadata = fp.filepad.find_one(query)


# In[137]:


metadata.keys()


# In[142]:


metadata['gfs_id']


# In[149]:


content, doc = fp.get_file_by_id(metadata['gfs_id'])


# In[155]:


print(content.decode())


# In[ ]:




