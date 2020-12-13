# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AFM probe on substrate preparation evaluation

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
from fireworks import LaunchPad, Firework, Tracker, Workflow 
from fireworks import FileTransferTask, PyTask, ScriptTask

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


# %%

# %%
from asgiref.sync import async_to_sync


async def get_item_dict(uri):
    manifest = await dl.manifest(uri)
    item_dict = {item['relpath']: item_id for item_id, item in manifest['items'].items()}
    return item_dict


async def read_thermo(uri, file_name='thermo.out'): 
    item_dict = await get_item_dict(uri)
    d = dtoolcore.DataSet.from_uri(uri)
    fpath = d.item_content_abspath(item_dict[file_name])


    df = pd.read_csv(fpath, delim_whitespace=True)
    df.set_index('Step', inplace=True)

    return df

def plot_df(df):
    nplots = len(df.columns)
    ncols = 2
    nrows = nplots // 2

    fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols, 5*nrows))
    for i, col in enumerate(df.columns):
        df[col].plot(ax=axs[i//ncols,i%ncols], title=col)
        
    return fig, axs

async def plot_thermo(uri, file_name='thermo.out'):
    df = await read_thermo(uri, file_name=file_name)
    return plot_df(fg)

# %%
async def get_df_by_query(query):
    res = await dl.query(query)
    res_df = pd.DataFrame(res)
    return res_df

async def get_df_by_filtered_query(query):
    aggregation_pipeline = [
        {
            "$match": query
        },
        {
            "$project": {
                'base_uri': True,
                'uuid': True,
                'uri': True,
                'frozen_at': True,
            }
        },
        {  # sort by earliest date, descending
            "$sort": { 
                "frozen_at": pymongo.DESCENDING,
            }
        }
    ]

    res = await dl.aggregate(aggregation_pipeline)
    res_df = pd.DataFrame(res)
    return res_df

async def get_uri_by_query(query):
    logger = logging.getLogger(__name__)
    res_df = await get_df_by_query(query)
    if len(res_df.uri) > 1:
        logger.warning("Query '%s' yields %d uris %s, return first entry only." % (query, len(res_df.uri), res_df.uri))
    return res_df.uri[0]
    

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
# ## Overview on steps in project

# %%
project_id = '2020-12-08-sds-on-au-111-probe-and-substrate-minimzation-equilibration-approach-not-quite-as-quick-test'

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
# ## Minimization

# %% [markdown]
# ### Overview on UUIDs in step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSMinimization'},
}

# %%
res = await dl.query(query)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
aggregation_pipeline = [
    {
        "$match": query
    },
    {
        "$project": {
            'base_uri': True,
            'uuid': True,
            'uri': True,
            'frozen_at': True,
        }
    },
    {  # sort by earliest date, descending
        "$sort": { 
            "frozen_at": pymongo.DESCENDING,
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
uri = res_df.uri[0]

# %% [markdown]
# ### List items

# %%
manifest = await dl.manifest(uri)

# %%
manifest

# %%
item_dict = {item['relpath']: item_id for item_id, item in manifest['items'].items()}

# %%
item_dict

# %% [markdown]
# ### Evaluate

# %%
import dtoolcore

# %%
d = dtoolcore.DataSet.from_uri(uri)

# %%
uri

# %%
fpath = d.item_content_abspath(item_dict['thermo.out'])

# %%
fpath

# %%
df = pd.read_csv(fpath, delim_whitespace=True)
df.set_index('Step', inplace=True)

# %%
df

# %%
nplots = len(df.columns)
ncols = 2
nrows = nplots // 2

fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols, 5*nrows))
for i, col in enumerate(df.columns):
    df[col].plot(ax=axs[i//ncols,i%ncols], title=col)

# %%

# %% [markdown]
# ## NVT Equilibration

# %% [markdown]
# ### Overview on UUIDs in step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSEquilibrationNVT'},
}

# %%
await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
uri = await get_uri_by_query(query)

# %%
uri

# %% [markdown]
# ### List items

# %%
item_dict = await get_item_dict(uri)

# %%
item_dict

# %% [markdown]
# ### Evaluate

# %%
df = await read_thermo(uri)

# %%
df

# %%
fig, ax = plot_df(df)
fig.show()

# %%

# %% [markdown]
# ## NPT Equilibration

# %% [markdown]
# ### Overview on UUIDs in step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSEquilibrationNPT'},
}

# %%
await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
uri = await get_uri_by_query(query)

# %%
uri

# %% [markdown]
# ### List items

# %%
item_dict = await get_item_dict(uri)

# %%
item_dict

# %% [markdown]
# ### Evaluate

# %%
df = await read_thermo(uri)

# %%
df

# %%
fig, ax = plot_df(df)
fig.show()

# %% [markdown]
# ## DPD Equilibration

# %% [markdown]
# ### Overview on UUIDs in step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSEquilibrationDPD'},
}

# %%
await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
uri = await get_uri_by_query(query)

# %%
uri

# %% [markdown]
# ### List items

# %%
item_dict = await get_item_dict(uri)

# %%
item_dict

# %% [markdown]
# ### Evaluate

# %%
df = await read_thermo(uri)

# %%
df

# %%
fig, ax = plot_df(df)
fig.show()

# %% [markdown]
# ## Normal approach

# %% [markdown]
# ### Overview on UUIDs in step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSProbeNormalApproch'},
}

# %%
await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
uri = await get_uri_by_query(query)

# %%
uri

# %% [markdown]
# ### List items

# %%
item_dict = await get_item_dict(uri)

# %%
item_dict

# %% [markdown]
# ### Evaluate

# %%
df = await read_thermo(uri)

# %%
df

# %%
fig, ax = plot_df(df)
fig.show()

# %%

# %%
