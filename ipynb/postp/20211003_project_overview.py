# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
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

# %%
# %config Completer.use_jedi = False

# %% init_cell=true
# %load_ext autoreload
# %autoreload 2

# %%
# %aimport

# %%
# see https://stackoverflow.com/questions/40536560/ipython-and-jupyter-autocomplete-not-working
# %config Completer.use_jedi = False

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
def as_std_type(value):
    """Convert numpy type to standard type."""
    return getattr(value, "tolist", lambda: value)()


# %% init_cell=true
def highlight_bool(s):
    """color boolean values in pandas dataframe"""
    return ['background-color: green' if v else 'background-color: red' for v in s]


# %% init_cell=true
def highlight_nan(s):
    """color boolean values in pandas dataframe"""
    l = []
    for v in s:
        try:
            ret = np.isnan(v)
        except: # isnan not applicable
            l.append('background-color: green')
        else:
            if ret:
                l.append('background-color: red')
            else:
                l.append('background-color: green')
      
    return l
    # return ['background-color: green' if not isinstance(v, np.floating) or not np.isnan(v) else 'background-color: red' for v in s]
    


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
    os.path.join(os.path.expanduser("~"), '.fireworks', 'fireworks_mongodb_auth.yaml'))

# %% [markdown]
# # Source datasets

# %%
# c, substrate (monolayer), probe uuid
probe_on_monolayer_input_datasets = [
    (0.0025,
      'e1766c11-ec23-488e-adfe-cbefc630ac68',
      '43a09eb9-a27b-42fb-a424-66d2cdbdf605'),
     (0.005,
      '13da3027-2a2f-45e7-bd8d-73bee135f24f',
      '9835cce9-b7d0-4e1f-9bdf-fd9767fea72c'),
     (0.0075,
      '0537047b-04c0-42b7-8aac-8de900c9e357',
      '30b97009-7d73-4e65-aa4a-04e1dc4cb2d2'),
     (0.01,
      'e11c3cb7-f53e-4024-af1c-dadbc8a01119',
      'a72b124b-c5aa-43d8-900b-f6b6ddc05d39'),
     (0.0125,
      'cda86934-50f1-4c7d-bdb5-3782c9f39a5a',
      '02c578b1-b331-42cf-8aef-4e3dcd0b4c77'),
     (0.015,
      'fe9bfb8d-d671-4c2d-bb1e-55dc1bfeec93',
      '974b41b2-de1c-421c-897b-7e091facff3a'),
     (0.0175,
      '90dbac16-9f05-4610-b765-484198116042',
      '86d2a465-61b8-4f1d-b13b-912c8f1f814b'),
     (0.02,
      'f0648f54-9a5d-488c-a913-ea53b88c99ce',
      '7e128862-4221-4554-bc27-22812a6047ae'),
     (0.0225,
      '9fb66e67-d08d-4686-972a-078bae8ef723',
      '1bc8bb4a-f4cf-4e4f-96ee-208b01bc3d02'),
     (0.025,
      'a9245caa-1439-4515-926f-c35b0476df44',
      '5b45ef1f-d24b-4a86-ab3c-3f3063b4def2'),
     (0.0275,
      'e96ff87c-7880-4d76-810e-e1a468d6b872',
      'b789ebc7-daec-488b-ba8f-e1c9b2d8fb47')]

probe_on_hemicylinders_input_datasets = [
     (0.005,
      '88c14189-f072-4df0-a04b-57bf27760b9d',
      '9835cce9-b7d0-4e1f-9bdf-fd9767fea72c'),
     (0.0075,
      'fcc304df-219a-4170-a6e0-bee06eed14e2',
      '30b97009-7d73-4e65-aa4a-04e1dc4cb2d2'),
     (0.01,
      '0899dd47-5659-408c-8dc1-253980adc975',
      'a72b124b-c5aa-43d8-900b-f6b6ddc05d39'),
     (0.0125,
      '01339270-76df-40c2-bec6-c69072f5a5f7',
      '02c578b1-b331-42cf-8aef-4e3dcd0b4c77'),
     (0.015,
      'b14873d7-0bba-4c2d-9915-ac9ee99f43c7',
      '974b41b2-de1c-421c-897b-7e091facff3a'),
     (0.0175,
      'b5d0bbfe-9c69-4dfd-9189-417bfa367882',
      '86d2a465-61b8-4f1d-b13b-912c8f1f814b')]

# %%
monolayer_input_datasets = [tup[1] for tup in probe_on_monolayer_input_datasets]

# %%
hemicylinders_input_datasets = [tup[1] for tup in probe_on_hemicylinders_input_datasets]

# %%
monolayer_input_datasets

# %%
hemicylinders_input_datasets

# %% [markdown]
# # Fireworks

# %%
project = '2020-12-23-sds-on-au-111-probe-and-substrate-conversion'

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
# ## Reconstruct shape
#
# shape attribute lost, reconstruct from source dataset uuids
#
#

# %%
query={'spec.metadata.project': project, 
       'spec.metadata.files_in_info.substrate_data_file.query.uuid': {
           '$in':}}

# %%
fw_ids = lp.get_fw_ids(query)

# %%
len(fw_ids)


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
res

# %%
readme = await dl.readme(res[0]['uri'])

# %%
readme

# %%
query = make_query({'datetime': {'$gt': '2020'} })

# %%
str(datetime.datetime(2020, 1, 1).timestamp())

# %%
aggregation_pipeline = [
    {
        "$match": {
            'creator_username': 'hoermann4',
            'readme.mode': 'production'
            #'frozen_at': {'$lt': datetime.datetime(2020, 1, 1).timestamp()}}
        }
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 'project': '$readme.project' },
            "object_count": {"$sum": 1}, # count matching data sets
            #"earliest":  {'$min': '$readme.datetime' },
            #"latest":  {'$max': '$readme.datetime' },
            "earliest":  {'$min': '$frozen_at' },
            "latest":  {'$max': '$frozen_at' },
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
type(res[0]['earliest'])

# %%
res_df = pd.DataFrame(res)

# %%
res

# %%
# %config Completer.use_jedi = False

# %%
res_df[['earliest','object_count','project']].iloc[:5]

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
# ## Overview on SDS-passivated indenters

# %%
project_id = "2020-07-29-sds-on-au-111-indenter-passivation"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
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
print(immutable_distinct_parameter_values)

# %% [markdown]
# #### Identify derived parameter values (i.e. concentrations from molecule numbers)

# %%
indenter_radius_Ang = readme["system"]["indenter"]["bounding_sphere"]["radius"] # Ang
indenter_radius_nm = indenter_radius_Ang / 10
indenter_surface_nm_sq = 4*np.pi*indenter_radius_nm**2

# %%
indenter_surface_nm_sq

# %%
np.array(immutable_distinct_parameter_values['nmolecules'])/indenter_surface_nm_sq

# %%
concentrations_by_nmolecules = {
    int(nmol): nmol / indenter_surface_nm_sq for nmol in sorted(immutable_distinct_parameter_values['nmolecules'])
}

# %%
concentrations_by_nmolecules

# %% [markdown]
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
}

# %%
query

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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            'shape': '$_id.shape'
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
len(res_df)

# %%
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %% [markdown]
# ### Look at last step

# %%
step_of_interest = "GromacsRelaxation:ProcessAnalyzeAndVisualize:push_dtool" # last step

# %%
final_config_df = res_df[res_df['step']==step_of_interest]

# %%
concentrations_by_nmolecules

# %%
final_config_df[['nmolecules','uuid']]

# %%
final_config_df[final_config_df['nmolecules'] == 241][['nmolecules','uuid']]

# %%
list_of_tuples = [(row['nmolecules'], row['uuid'][0]) 
    for _, row in final_config_df[['nmolecules','uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[['nmolecules','uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# ## Overview on SDS-passivated substrates (2020/12)

# %%
project_id = "2020-12-14-sds-on-au-111-substrate-passivation"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.surfactant.surface_concentration',
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
res_df.sort_values(by='concentration')

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
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
}

# %%
query

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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                #'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            #'shape': '$_id.shape'
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
res_df[["step"]]

# %%
len(res_df)

# %%
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "SubstratePassivation:HemicylindricalPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool",
    "SubstratePassivation:MonolayerPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool",
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# ## Overview on SDS-passivated substrates (2021/10)

# %%
project_id = "2021-10-06-sds-on-au-111-substrate-passivation"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.surfactant.surface_concentration',
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
res_df.sort_values(by='concentration')

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
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
}

# %%
query

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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                #'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            #'shape': '$_id.shape'
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
res_df[["step"]]

# %%
len(res_df)

# %%
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "SubstratePassivation:HemicylindricalPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool",
    "SubstratePassivation:MonolayerPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool",
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# ## Overview on probe-substrate equilbrated systems

# %% [markdown]
# ## Overview on AFM approach (2021/01/28)

# %%
project_id = "2021-01-28-sds-on-au-111-probe-and-substrate-approach"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    'system.surfactant.surface_concentration': {'$exists': True},
    # 'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.surfactant.surface_concentration',
    # 'shape': 'readme.system.surfactant.aggregates.shape',
    #'x_shift': 'readme.step_specific.merge.x_shift',
    #'y_shift': 'readme.step_specific.merge.y_shift',
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
res_df.sort_values(by='nmolecules')

# %%
distinct_parameter_values = {k: set(res_df[k].unique()) for k in parameters.keys()}

# %%
# filter out unwanted values
immutable_distinct_parameter_values = {k: [e.item() for e in p if (
        isinstance(e, np.float64) and not np.isnan(e)) or (
        not isinstance(e, np.float64) and e is not None)] 
    for k, p in distinct_parameter_values.items()}

# %%
print(immutable_distinct_parameter_values)

# %% [markdown]
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
}

# %%
query

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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                #'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            #'shape': '$_id.shape'
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
res_df[["step"]]

# %%
len(res_df)

# %%
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "ProbeOnSubstrateNormalApproach:ProbeAnalysis:push_dtool"
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# ## Overview on AFM approach (2021/02/05)

# %%
project_id = "2021-02-05-sds-on-au-111-probe-and-substrate-approach"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    'system.surfactant.surface_concentration': {'$exists': True},
    # 'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.surfactant.surface_concentration',
    # 'shape': 'readme.system.surfactant.aggregates.shape',
    #'x_shift': 'readme.step_specific.merge.x_shift',
    #'y_shift': 'readme.step_specific.merge.y_shift',
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
res_df.sort_values(by='nmolecules')

# %%
distinct_parameter_values = {k: set(res_df[k].unique()) for k in parameters.keys()}

# %%
# filter out unwanted values
immutable_distinct_parameter_values = {k: [e.item() for e in p if (
        isinstance(e, np.float64) and not np.isnan(e)) or (
        not isinstance(e, np.float64) and e is not None)] 
    for k, p in distinct_parameter_values.items()}

# %%
print(immutable_distinct_parameter_values)

# %% [markdown]
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
}

# %%
query

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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                #'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            #'shape': '$_id.shape'
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
res_df[["step"]]

# %%
len(res_df)

# %%
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "ProbeOnSubstrateNormalApproach:ProbeAnalysis:push_dtool"
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# ## Overview on merge & AFM approach (2021/10/07)

# %%
project_id = "2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
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
res_df.sort_values(by='nmolecules')

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
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
}

# %%
query

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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                #'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            #'shape': '$_id.shape'
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
res_df[["step"]]

# %%
len(res_df)

# %%
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "ProbeOnSubstrateMergeConversionMinimizationEquilibrationAndApproach:ProbeAnalysis:push_dtool",
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# ### Look at normal approach step

# %%
steps_of_interest = [
    "ProbeOnSubstrateMergeConversionMinimizationEquilibrationAndApproach:LAMMPSProbeNormalApproach:push_dtool",
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# ### Look at DPD equilibration step

# %%
steps_of_interest = [
    "ProbeOnSubstrateMergeConversionMinimizationEquilibrationAndApproach:LAMMPSEquilibrationDPD:push_dtool",
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %%
len(list_of_tuples)

# %% [markdown]
# ## Overview on frame extraction

# %%
project_id = "2021-12-09-sds-on-au-111-probe-and-substrate-approach-frame-extraction"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
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
res_df.sort_values(by='nmolecules')

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
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': [as_std_type(val) for val in values]} for label, values in distinct_parameter_values.items()},
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
set(parameters.keys()) - set(['distance'])

# %%
res_pivot = res_df.pivot_table(
    values='object_count', index=['step', 'distance'], columns=list(set(parameters.keys()) - set(['distance'])), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': [as_std_type(val) for val in values]} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                #'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            #'shape': '$_id.shape'
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
len(res_df)

# %%
res_pivot = res_df.pivot(values='uuid', index=['step', 'distance'], columns=list(set(parameters.keys()) - set(['distance']))) 
#res_pivot.style.apply(highlight_nan)
res_pivot.

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "ForeachPushStub:push_dtool",
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %%
df = pd.DataFrame(final_config_datasets)
df.to_clipboard(index=False,header=False)

# %% [markdown]
# ## Overview on wrap-join

# %%
project_id = "2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
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
res_df.sort_values(by='nmolecules')

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
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': [as_std_type(val) for val in values]} for label, values in distinct_parameter_values.items()},
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
set(parameters.keys()) - set(['distance'])

# %%
res_pivot = res_df.pivot_table(
    values='object_count', index=['step', 'distance'], columns=list(set(parameters.keys()) - set(['distance'])), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': [as_std_type(val) for val in values]} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                #'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            #'shape': '$_id.shape'
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
len(res_df)

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "WrapJoinAndDPDEquilibration:WrapJoinDataFile:push_dtool",
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# ## Overview on wrap-join

# %%
project_id = "2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
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
res_df.sort_values(by='nmolecules')

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
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': [as_std_type(val) for val in values]} for label, values in distinct_parameter_values.items()},
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
set(parameters.keys()) - set(['distance'])

# %%
res_pivot = res_df.pivot_table(
    values='object_count', index=['step', 'distance'], columns=list(set(parameters.keys()) - set(['distance'])), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': [as_std_type(val) for val in values]} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                #'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            #'shape': '$_id.shape'
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
len(res_df)

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "WrapJoinAndDPDEquilibration:WrapJoinDataFile:push_dtool",
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %%
df = pd.DataFrame(final_config_datasets)
df.to_clipboard(index=False,header=False)

# %%
df = pd.DataFrame(final_config_datasets)
df.to_clipboard(index=False,header=False)

# %% [markdown]
# ## Overview on repeated DPD equilibration

# %%
project_id = "2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
len(await dl.query(query))

# %% [markdown]
# ### Overview on steps in project

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

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
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
res_df.sort_values(by='nmolecules')

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
# #### Actual pivot

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': [as_std_type(val) for val in values]} for label, values in distinct_parameter_values.items()},
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
set(parameters.keys()) - set(['distance'])

# %%
res_pivot = res_df.pivot_table(
    values='object_count', index=['step', 'distance'], columns=list(set(parameters.keys()) - set(['distance'])), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': [as_std_type(val) for val in values]} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
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
                #'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            #'shape': '$_id.shape'
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
len(res_df)

# %% [markdown]
# ### Look at equilibrated configurations

# %%
steps_of_interest = [
    "WrapJoinAndDPDEquilibration:LAMMPSEquilibrationDPD:push_dtool",
    "LAMMPSEquilibrationDPD:push_dtool",
]


# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest)]

# %%
final_config_df

# %%
final_config_df[[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %%
#df = pd.DataFrame(final_config_datasets)
#df.to_clipboard(index=False,header=False)

# %% [markdown]
# #### Datasets at x = 25, y = 0 (on hemicylinders)

# %%
# y shift 0: on hemicylinders
selection = (final_config_df['x_shift'] == 25.0) & (final_config_df['y_shift'] == 0.0)

# %%
final_config_df[selection]

# %%
final_config_df[selection][[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[selection][[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[selection][[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# #### Datasets at x = 0, y = -25 (between hemicylinders)

# %%
# y shift -25: between hemicylinders
selection = (final_config_df['x_shift'] == 0.0) & (final_config_df['y_shift'] == -25.0)

# %%
final_config_df[selection]

# %%
final_config_df[selection][[*list(parameters.keys()),'uuid']]

# %%
list_of_tuples = [(*row[parameters.keys()], row['uuid'][0]) 
    for _, row in final_config_df[selection][[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[selection][[*list(parameters.keys()),'uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %% [markdown]
# ## Overview on lateral sliding

# %% [markdown]
# ## Overview on steps in project (SDS on substrate)

# %%
# project_id = "2020-07-29-sds-on-au-111-indenter-passivation"
#project_id = '2020-11-25-au-111-150x150x150-fcc-substrate-creation'
#project_id = '2020-09-14-sds-on-au-111-substrate-passivation'
#project_id = '2020-10-13-ctab-on-au-111-substrate-passivation
#project_id = '2020-12-12-sds-on-au-111-substrate-passivation-trial'
# project_id = '2020-12-23-sds-on-au-111-probe-and-substrate-conversion'
project_id = '2021-02-26-sds-on-au-111-probe-and-substrate-conversion'

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

# %% [markdown]
# <font color="red">Issue: shape parameter lost.</font>

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.query(query)

# %%
len(res)

# %%
readme = await dl.readme(res[-1]['uri'])

# %%
readme

# %%
len(await dl.query(query))

# %%
parameters = { 
    #'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.surfactant.surface_concentration',
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
print(immutable_distinct_parameter_values)

# %% [markdown]
# ### Refined aggregation for hemicylindrical systems

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
    'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
}

# %%
query

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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %%
res_pivot.to_excel('20210226_probe_substrate_merge_hemicylinders.xlsx')

# %% [markdown]
# ### Refined aggregation for monolayer systems

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
    'readme.files_in_info.substrate_data_file.query.uuid': {'$in': monolayer_input_datasets}
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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %%
res_pivot.to_excel('20210226_probe_substrate_merge_monolayer.xlsx')

# %% [markdown]
# ### Overview on UUIDs

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}

# %%
query = {
    'readme.project': project_id,
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
    #'readme.files_in_info.substrate_data_file.query.uuid': {'$in': hemicylinders_input_datasets}
}

# %%
# check files degenerate by 'metadata.type' ad 'metadata.name'
aggregation_pipeline = [
    {
        "$match": query
    },
    {
        '$set': {'readme.shape': { 
            '$cond': [
                {'$in': [
                    '$readme.files_in_info.substrate_data_file.query.uuid',
                    hemicylinders_input_datasets]
                }, 
                'hemicylinders', 
                'monolayer']
            }
        } 
    },
    {  # group by unique project id
        "$group": { 
            "_id": { 
                'step': '$readme.step',
                **{label: '${}'.format(key) for label, key in parameters.items()},
                'shape': '$readme.shape'
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
            'shape': '$_id.shape'
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
len(res_df)

# %%
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=['shape',*list(parameters.keys())])
res_pivot.style.apply(highlight_nan)

# %%

# %%
final_config_df = res_df[res_df['step']=='ProbeOnSubstrateMergeConversionMinimizationAndEquilibration:ProbeOnSubstrateMinimizationAndEquilibration:LAMMPSEquilibrationDPD:push_dtool']

# %%
final_config_df[['concentration','shape','uuid']]

# %%
list_of_tuples = [(row['concentration'], row['shape'], row['uuid'][0]) 
    for _, row in final_config_df[['concentration','shape','uuid']].iterrows()]

# %%
list_of_tuples

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[['concentration','shape','uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %%
