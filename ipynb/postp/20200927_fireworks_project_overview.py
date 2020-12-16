# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fireworks overview

# %% [markdown]
# This notebook demonstrates querying of Fireworks workflows and Filepad objects

# %% [markdown]
# ## Initialization

# %% [markdown]
# ### IPython magic

# %% init_cell=true
# %load_ext autoreload
# %autoreload 2

# %%
# %aimport

# %% [markdown]
# ### Imports

# %% init_cell=true
import ase.io # here used for reading pdb files
from ase.visualize import view
from ase.visualize.plot import plot_atoms # has nasty offset issues
from cycler import cycler # here used for cycling through colors in plots
import datetime

# dtool functionality
import dtool_lookup_api.asynchronous as dl
import dtoolcore

# FireWorks functionality 
from fireworks import Firework, LaunchPad, ScriptTask, Workflow
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask, GetFilesTask, GetFilesByQueryTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from fireworks.utilities.filepad import FilePad # direct FilePad access, similar to the familiar LaunchPad

from collections.abc import Iterable
import copy
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

# %% [markdown]
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

# %% [markdown]
# ### Logging

# %% init_cell=true
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# %% [markdown]
# ParmEd needs to know the GROMACS topology folder, usually get this from 
# envionment variable `GMXLIB`:

# %% [markdown]
# ### Function definitions

# %% init_cell=true
def highlight_bool(s):
    """color boolean values in pandas dataframe"""
    return ['background-color: green' if v else 'background-color: red' for v in s]


# %% init_cell=true
def highlight_nan(s):
    """color boolean values in pandas dataframe"""
    return ['background-color: green' if not isinstance(v, np.floating) or not np.isnan(v) else 'background-color: red' for v in s]


# %% init_cell=true
def find_undeclared_variables(infile):
    """identify all variables evaluated in a jinja 2 template file"""
    env = jinja2.Environment()
    with open(infile) as template_file:
        parsed = env.parse(template_file.read())

    undefined = jinja2.meta.find_undeclared_variables(parsed)
    return undefined


# %% init_cell=true
def memuse():
    """Quick overview on memory usage of objects in Jupyter notebook"""
    # https://stackoverflow.com/questions/40993626/list-memory-usage-in-ipython-and-jupyter
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    return sorted([(x, sys.getsizeof(globals().get(x))) for x in dir(sys.modules['__main__']) if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# %% [markdown]
# ### Global settings

# %% init_cell=true
# pandas settings
# https://songhuiming.github.io/pages/2017/04/02/jupyter-and-pandas-display/
pd.options.display.max_rows = 200
pd.options.display.max_columns = 16
pd.options.display.max_colwidth = 256
pd.options.display.max_colwidth = None

# %% init_cell=true
gmxtop = os.path.join( os.path.expanduser("~"),
                       'git', 'gromacs', 'share', 'top')

# %% init_cell=true
os.environ['GMXLIB'] = gmxtop

# %% init_cell=true
# pmd.gromacs.GROMACS_TOPDIR = os.environ['GMXLIB']
pmd.gromacs.GROMACS_TOPDIR = gmxtop

# %% init_cell=true
# prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'
# prefix = '/mnt/dat/work'

# %% init_cell=true
date_prefix = datetime.datetime.now().strftime("%Y%m%d")

# %% init_cell=true
work_prefix = os.path.join( os.path.expanduser("~"), 'sandbox', date_prefix + '_fireworks_project_overview')

# %% init_cell=true
try:
    os.mkdir(work_prefix)
except FileExistsError as exc:
    print(exc)

# %% init_cell=true
os.chdir(work_prefix)

# %% init_cell=true
# the FireWorks LaunchPad
lp = LaunchPad.auto_load() #Define the server and database
# FilePad behaves analogous to LaunchPad
# fp = FilePad.auto_load()
fp = FilePad.from_db_file(
    os.path.join(os.path.expanduser("~"), '.fireworks', 'fireworks_sandbox_mongodb_auth.yaml'))

# %% [markdown]
# # Fireworks

# %%
project = '2020-09-28-ctab-on-au-111-substrate-passivation'

# %%
query={'spec.metadata.project': project}

# %%
fw_ids = lp.get_fw_ids(query)

# %%
len(fw_ids)

# %%
wf_query = {'nodes': {'$in': fw_ids}}

# %%
lp.workflows.count_documents(wf_query)

# %%
wf = lp.workflows.find_one(wf_query)

# %%
wf.keys()

# %%
fw_ids = wf['nodes']

# %%
fw = lp.fireworks.find_one()

# %%
fw

# %%
query = {'fw_id': {'$in': fw_ids}}

# %%
lp.fireworks.count_documents(query)

# %%
query = {'fw_id': {'$in': fw_ids}, 'name': {'$regex':'NPT'}}

# %%
lp.fireworks.count_documents(query)

# %%
query = {'fw_id': {'$in': fw_ids}, 'name': {'$regex':'NPT.*mdrun'}, 'state': 'COMPLETED'}

# %%
lp.fireworks.count_documents(query)

# %%
fw = lp.fireworks.find_one(query)

# %%
fw['fw_id']

# %%
fw['name']

# %%
fw['state']

# %%
fw['spec']['metadata']['step_specific']

# %%
fw['spec']['metadata']['step_specific']


# %% [markdown]
# # dtool

# %% init_cell=true
def make_query(d:dict={}):
    q = {'creator_username': 'hoermann4'}
    for k, v in d.items():
        q['readme.'+k] = v
    return q


# %% [markdown]
# ## Overview on recent projects

# %%
import dtool_lookup_api.asynchronous as dl

# %%
res = await dl.query({'readme.owners.name': {'$regex': 'Johannes'}})

# %%
len(res)

# %%
res = await dl.query({'creator_username': 'hoermann4'})

# %%
len(res)

# %%
readme = await dl.readme(res[0]['uri'])

# %%
readme

# %%
query = make_query({'datetime': {'$gt': '2020'} })

# %%
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 'project': '$readme.project' },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$readme.datetime' },
            "latest":  {'$max': '$readme.datetime' },
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

# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %% [markdown]
# ## Overview on recent production projects

# %%
res = await dl.query(
    {
        'readme.owners.name': {'$regex': 'Johannes'},
        'readme.mode': 'production'
    })

# %%
len(res)

# %%
query = make_query(
    {
        'datetime': {'$gt': '2020'},
        'mode': 'production'
    })

# %%
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 'project': '$readme.project' },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$readme.datetime' },
            "latest":  {'$max': '$readme.datetime' },
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

# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %% [markdown]
# ## Overview on steps in project (SDS on substrate)

# %%
#project_id = '2020-11-25-au-111-150x150x150-fcc-substrate-creation'
project_id = '2020-09-14-sds-on-au-111-substrate-passivation'
#project_id = '2020-10-13-ctab-on-au-111-substrate-passivation'

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$readme.step',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$frozen_at' },
            "latest":  {'$max': '$frozen_at' },
        },
    },
    {
        "$set": {
            'step': '$_id.step',
        }
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
columns = ['step', 'earliest', 'latest', 'object_count']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook

# %%
res_df

# %% [markdown]
# ### Pivot overview on steps and parameters in project

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
len(await dl.query(query))

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    'shape': 'readme.system.surfactant.aggregates.shape',
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": {k: '${}'.format(v) for k, v in parameters.items()},
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$frozen_at' },
            "latest":  {'$max': '$frozen_at' },
        },
    },
    {
        "$set": {k: '$_id.{}'.format(k) for k in parameters.keys()}
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
res_df['nmolecules'].unique()[-1]

# %%
type(res_df['nmolecules'].unique()[-1])

# %%
type(np.nan)

# %%
type(np.float64(np.nan))

# %%
res_df['shape'].unique()

# %%
distinct_parameter_values = {k: set(res_df[k].unique()) for k in parameters.keys()}

# %%
# filter out unwanted values
immutable_distinct_parameter_values = {k: [e for e in p if (
        isinstance(e, np.float64) and not np.isnan(e)) or (
        not isinstance(e, np.float64) and e is not None)] 
    for k, p in distinct_parameter_values.items()}

# %%
print(immutable_distinct_parameter_values)

# %% [markdown]
# ### Refined aggregation for hemicylindrical systems

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}
distinct_parameter_values['shape'].remove('monolayer')

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$readme.step',
                **{label: '${}'.format(key) for label, key in parameters.items()},
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$readme.datetime' },
            "latest":  {'$max': '$readme.datetime' },
        },
    },
    {
        "$set": {
            'step': '$_id.step',
            **{k: '$_id.{}'.format(k) for k in parameters.keys()}
        }
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
res_pivot = res_df.pivot_table(values='object_count', index=['step'], columns=list(parameters.keys()), aggfunc=pd.notna, fill_value=False)
res_pivot.style.apply(highlight_bool)

# %% [markdown]
# #### Concentrations

# %%
query = {
    'readme.project': project_id,
    #'readme.step': {'$regex': 'GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool'},
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
doc_id = {
    'step': '$readme.step',
    'length': '$readme.system.box.length',
    'width': '$readme.system.box.width',
    **{label: '${}'.format(key) for label, key in parameters.items()},
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": doc_id,
            "object_count": {"$sum": 1}, # count matching data sets
            "uuid": {"$addToSet": "$uuid"},
            "earliest":  {'$min': '$readme.datetime'},
            "latest":  {'$max': '$readme.datetime'},
        },
    },
    {
        "$set": {k: '$_id.{}'.format(k) for k in doc_id.keys()}
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df['surfc'] = res_df['nmolecules']/(res_df['length']*res_df['width'])

# %%
res_df

# %%
doc_id

# %%
res_pivot = res_df.pivot(values='uuid', index=['surfc'], columns=['step'])

# %%
res_pivot

# %%
substrate_hemicylinders_res_df = res_pivot.copy()

# %% [markdown]
# ### Refined aggregation for monolayer systems

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}
distinct_parameter_values['shape'].remove('hemicylinders')

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$readme.step',
                **{label: '${}'.format(key) for label, key in parameters.items()},
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$readme.datetime' },
            "latest":  {'$max': '$readme.datetime' },
        },
    },
    {
        "$set": {
            'step': '$_id.step',
            **{k: '$_id.{}'.format(k) for k in parameters.keys()}
        }
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
res_pivot = res_df.pivot_table(values='object_count', index=['step'], columns=list(parameters.keys()), aggfunc=pd.notna, fill_value=False)
res_pivot.style.apply(highlight_bool)

# %% [markdown]
# #### Concentrations (at final step)

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
doc_id = {
    'step': '$readme.step',
    'length': '$readme.system.box.length',
    'width': '$readme.system.box.width',
    **{label: '${}'.format(key) for label, key in parameters.items()},
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": doc_id,
            "object_count": {"$sum": 1}, # count matching data sets
            "uuid": {"$addToSet": "$uuid"},
            "earliest":  {'$min': '$readme.datetime'},
            "latest":  {'$max': '$readme.datetime'},
        },
    },
    {
        "$set": {k: '$_id.{}'.format(k) for k in doc_id.keys()}
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df['surfc'] = res_df['nmolecules']/(res_df['length']*res_df['width'])

# %%
res_pivot = res_df.pivot(values='uuid', index=['surfc'], columns=['step'])

# %%
res_pivot

# %%
substrate_monolayer_res_df = res_pivot.copy()

# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}
distinct_parameter_values['shape'].remove('hemicylinders')

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$readme.step',
                **{label: '${}'.format(key) for label, key in parameters.items()},
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "uuid": {"$addToSet": "$uuid"},
            "earliest":  {'$min': '$readme.datetime'},
            "latest":  {'$max': '$readme.datetime'},
        },
    },
    {
        "$set": {
            'step': '$_id.step',
            **{k: '$_id.{}'.format(k) for k in parameters.keys()}
        }
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %% [markdown]
# ## Overview on steps in project (SDS on probe)

# %%
project_id = '2020-07-29-sds-on-au-111-indenter-passivation'

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$readme.step',
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$frozen_at' },
            "latest":  {'$max': '$frozen_at' },
        },
    },
    {
        "$set": {
            'step': '$_id.step',
        }
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
columns = ['step', 'earliest', 'latest', 'object_count']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook

# %%
res_df

# %% [markdown]
# ### Pivot overview on steps and parameters in project

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
len(await dl.query(query))

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'shape': 'readme.system.surfactant.aggregates.shape',
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": {k: '${}'.format(v) for k, v in parameters.items()},
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$frozen_at' },
            "latest":  {'$max': '$frozen_at' },
        },
    },
    {
        "$set": {k: '$_id.{}'.format(k) for k in parameters.keys()}
    },
    {  # sort by earliest date, descending
        "$sort": { 
            **{k: pymongo.ASCENDING for k in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
distinct_parameter_values = {k: set(res_df[k].unique()) for k in parameters.keys()}

# %%
# filter out unwanted values
immutable_distinct_parameter_values = {k: [e for e in p if (
        isinstance(e, np.float64) and not np.isnan(e)) or (
        not isinstance(e, np.float64) and e is not None)] 
    for k, p in distinct_parameter_values.items()}

# %%
immutable_distinct_parameter_values

# %% [markdown]
# ### Refined overview on steps

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$readme.step',
                **{label: '${}'.format(key) for label, key in parameters.items()},
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "earliest":  {'$min': '$readme.datetime' },
            "latest":  {'$max': '$readme.datetime' },
        },
    },
    {
        "$set": {
            'step': '$_id.step',
            **{k: '$_id.{}'.format(k) for k in parameters.keys()}
        }
    },
    {  # sort by earliest date, descending
        "$sort": { 
            **{k: pymongo.ASCENDING for k in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
res_pivot = res_df.pivot_table(values='object_count', index=['step'], columns=list(parameters.keys()), aggfunc=pd.notna, fill_value=False)
res_pivot.style.apply(highlight_bool)

# %% [markdown]
# ### Overview on UUIDs

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$readme.step',
                **{label: '${}'.format(key) for label, key in parameters.items()},
            },
            "object_count": {"$sum": 1}, # count matching data sets
            "uuid": {"$addToSet": "$uuid"},
            "earliest":  {'$min': '$readme.datetime'},
            "latest":  {'$max': '$readme.datetime'},
        },
    },
    {
        "$set": {
            'step': '$_id.step',
            **{k: '$_id.{}'.format(k) for k in parameters.keys()}
        }
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %% [markdown]
# ### Concentrations

# %%
query = {
    'readme.project': project_id,
    #'readme.step': 'GromacsRelaxation:ProcessAnalyzeAndVisualize:push_dtool',
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
doc_id = {
    'step': '$readme.step',
    'radius': '$readme.system.indenter.bounding_sphere.radius',
    **{label: '${}'.format(key) for label, key in parameters.items()},
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {  # group by unique project id
        "$group": { 
            "_id": doc_id,
            "object_count": {"$sum": 1}, # count matching data sets
            "uuid": {"$addToSet": "$uuid"},
            "earliest":  {'$min': '$readme.datetime'},
            "latest":  {'$max': '$readme.datetime'},
        },
    },
    {
        "$set": {k: '$_id.{}'.format(k) for k in doc_id.keys()}
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.aggregate(aggregation_pipeline)

# %%
res_df = pd.DataFrame(res)

# %%
res_df['surfc'] = res_df['nmolecules']/(4.0*np.pi*np.square(res_df['radius']))

# %%
res_pivot = res_df.pivot(values='uuid', index=['surfc'], columns=['step'])

# %%
res_pivot

# %%
probe_res_df = res_pivot.copy()

# %% [markdown]
# ## Match concentrations

# %% [markdown]
# ### substrate

# %%
substrate_hemicylinders_res_df

# %%
substrate_monolayer_res_df

# %% [markdown]
# ### probe

# %%
probe_res_df

# %%
substrate_monolayer_res_df['SubstratePassivation:MonolayerPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool']

# %%
substrate_hemicylinders_res_df['SubstratePassivation:HemicylindricalPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool']

# %%
probe_res_df['GromacsRelaxation:ProcessAnalyzeAndVisualize:push_dtool']

# %%

# %%
probe_on_monolayer_matches = pd.concat([
    substrate_monolayer_res_df['SubstratePassivation:MonolayerPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool'].reset_index(),
    #substrate_hemicylinders_res_df['SubstratePassivation:HemicylindricalPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool'].reset_index(),
    probe_res_df['GromacsRelaxation:ProcessAnalyzeAndVisualize:push_dtool'].reset_index()
], axis=1)

# %%
probe_on_monolayer_matches

# %%
probe_on_monolayer_tuples = probe_on_monolayer_matches[['surfc',
    'GromacsRelaxation:ProcessAnalyzeAndVisualize:push_dtool',
    'SubstratePassivation:MonolayerPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool',
]].dropna()

# %%
[ tuple(c[0] if isinstance(c, list) else c for c in r) for r in probe_on_monolayer_tuples.values.tolist() ]

# %%
probe_on_hemicylinders_matches = pd.concat([
    substrate_hemicylinders_res_df['SubstratePassivation:HemicylindricalPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool'].reset_index(),
    probe_res_df['GromacsRelaxation:ProcessAnalyzeAndVisualize:push_dtool'].reset_index()
], axis=1)

# %%
probe_on_hemicylinders_filtered = probe_on_hemicylinders_matches[['surfc',
    'GromacsRelaxation:ProcessAnalyzeAndVisualize:push_dtool',
    'SubstratePassivation:HemicylindricalPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool'
]].dropna()

# %%
probe_on_hemicylinders_tuples = [ tuple(c[0] if isinstance(c, list) else c for c in r) for r in probe_on_hemicylinders_filtered.values.tolist() ]

# %%
probe_on_hemicylinders_tuples

# %%
probe_res_df[['surfc','nmolecules']]

# %%
# test 506, 197

# %% [markdown]
# # Filepad

# %% [markdown]
# ## Overview

# %% [markdown]
# ### Overview on recent projects in database

# %%
query = {'metadata.datetime': {'$gt': '2020'}}

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

# %% [markdown]
# ### Overview on recent production projects in database

# %%
query = {
    'metadata.datetime': {'$gt': '2020'},
    'metadata.mode': 'production'
}

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

# %% [markdown]
# ### Overview on steps in project

# %%
project_id = '2020-09-28-ctab-on-au-111-substrate-passivation'
#project_id = '2020-10-13-ctab-on-au-111-substrate-passivation'

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

# %% [markdown]
# #### Pivot overview on steps and parameters in project

# %%
project_id = '2020-10-13-ctab-on-au-111-substrate-passivation'

# %%
query = {
    'metadata.project': project_id,
    'metadata.system.surfactant.nmolecules': {'$exists': True},
    'metadata.system.surfactant.aggregates.shape': {'$exists': True},
}

# %%
# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)

# %%
parameters = { 
    'nmolecules': 'metadata.system.surfactant.nmolecules',
    'shape': 'metadata.system.surfactant.aggregates.shape',
}

# %%
distinct_parameter_values = {}
for label, key in parameters.items():
    values = fp.filepad.distinct(key, query)
    if None in values:
        values.remove(None)
    distinct_parameter_values[label] = values

# %%
print(distinct_parameter_values)

# %% [markdown]
# #### Refined aggregation for hemicylindrical systems

# %%
distinct_parameter_values['shape'].remove('bilayer')

# %%
query = {
    'metadata.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
print(query)

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

# %%
res_pivot = res_df.pivot_table(values='object_count', index=['step'], columns=list(parameters.keys()), aggfunc=pd.notna, fill_value=False)
res_pivot.style.apply(highlight_bool)

# %% [markdown]
# #### Refined aggregation for bilayer systems

# %%
distinct_parameter_values['shape'].remove('cylinders')
distinct_parameter_values['shape'].append('bilayer')

# %%
query = {
    'metadata.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()}
}

# %%
print(query)

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

# %%
res_pivot = res_df.pivot_table(values='object_count', index=['step'], columns=list(parameters.keys()), aggfunc=pd.notna, fill_value=False)
res_pivot.style.apply(highlight_bool)

# %%
res_df.groupby(['step', *parameters.keys()])

# %%
res_df.set_index('step').stack()

# %%
(res_df.set_index('step').stack()
 .groupby(level=[0,1])
 .value_counts()
 .unstack(level=[1,2])
 .fillna(0)
 .sort_index(axis=1))

# %%
res_df.groupby()

# %%
parameters.keys()

# %%
res_df.set_index(list(parameters.keys()))

# %%
pd.MultiIndex.from_frame()

# %%

res_df.pivot(index='step', columns='shape', values='object_count')

# %%
res_df.multiply(*parameters.keys())

# %% [markdown]
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

# %%
res_df

# %% [markdown]
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

# %% [markdown]
# ### Overview on objects in specific step

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

# %% [markdown]
# ### Overview on specific objects in specific steps

# %%
# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.step': {'$regex': 'GromacsNPTEquilibration:push_filepad'},
    'metadata.type': 'log_file',
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

# %% [markdown]
# ### Inspect specific file

# %%
metadata = fp.filepad.find_one(query)

# %%
metadata.keys()

# %%
metadata['gfs_id']

# %%
content, doc = fp.get_file_by_id(metadata['gfs_id'])

# %%
print(content.decode())

# %%
