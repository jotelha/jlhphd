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
# # Analyze substrate solvation

# %% [markdown]
# This notebook demonstrates deposition of an SDS adsorption layer on a non-spherical AFM tip model.

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

# dtool functionality
import dtool_lookup_api.asynchronous as dl
import dtoolcore

# dtool related
from asgiref.sync import async_to_sync

# abything else
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
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


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


# %% init_cell=true
def make_query(d:dict={}):
    q = {'creator_username': 'hoermann4'}
    for k, v in d.items():
        q['readme.'+k] = v
    return q

async def get_item_dict(uri):
    manifest = await dl.manifest(uri)
    item_dict = {item['relpath']: item_id for item_id, item in manifest['items'].items()}
    return item_dict

async def fetch_item_by_id(uri, item_id): 
    d = dtoolcore.DataSet.from_uri(uri)
    return d.item_content_abspath(item_id)

async def fetch_item(uri, file_name): 
    item_dict = await get_item_dict(uri)
    return await fetch_item_by_id(uri, item_dict[file_name])

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

# %% init_cell=true
DEFAULT_PARAMETER_DICT = {
        'shape': 'readme.system.surfactant.aggregates.shape',
        'nmolecules': 'readme.system.surfactant.nmolecules'
    }

async def get_df_by_aggregation(
        query,
        parameter_dict=DEFAULT_PARAMETER_DICT,
        id_dict=None):

    if id_dict is None:
        id_dict = {
            'step': '$readme.step',
            **{label: '${}'.format(key) for label, key in parameter_dict.items()},
            "uuid": "$uuid",
            "uri": "$uri",
        }
        
    # check files degenerate by 'metadata.type' ad 'metadata.name'
    aggregation_pipeline = [
        {
            "$match": query
        },
        {  # group by unique project id
            "$group": { 
                "_id":  id_dict,
                "object_count": {"$sum": 1}, # count matching data sets, must always be 1
                "earliest":  {'$min': '$readme.datetime' },
                "latest":  {'$max': '$readme.datetime' },
            },
        },
        {
            "$set": {k: '$_id.{}'.format(k) for k in id_dict.keys()}
        },
        {  # sort by earliest date, descending
            "$sort": { 
                "earliest": pymongo.DESCENDING,
            }
        }
    ]

    res = await dl.aggregate(aggregation_pipeline)
    return pd.DataFrame(res)

# edr evaluation
async def get_edr_df(
        ds_df, filename='default.edr', parameter_dict=DEFAULT_PARAMETER_DICT):

    res_mi_list = []

    for index, row in ds_df.iterrows():
        fpath = await fetch_item(row['uri'], filename)
        em_df = panedr.edr_to_df(fpath)    
        mi = pd.MultiIndex.from_product(
            [*[[row[p]] for p in parameter_dict.keys()], em_df.index],
                names=[*parameter_dict.keys(),'step'])
        em_mi_df = em_df.set_index(mi)        
        res_mi_list.append(em_mi_df)
        # print(row[parameter_dict.keys()].values)
        print('.',end='')


    res_mi_df = pd.concat(res_mi_list)
    return res_mi_df
    # res_df = res_mi_df.reset_index()
    
async def get_mp4_dict(
        ds_df, filename='default.mp4', parameter_dict=DEFAULT_PARAMETER_DICT):
    
    obj_dict = {}
    for index, row in ds_df.iterrows():
        fpath = await fetch_item(row['uri'], filename)
        obj_dict.update({tuple(row[parameter_keys].values): Video.from_file(fpath)})
        print('.',end='')
        
    return obj_dict

# %%
# rdf 
async def get_rdf_dict(
        ds_df, filename='default.txt', parameter_dict=DEFAULT_PARAMETER_DICT):
    
    parameter_keys = list(parameter_dict.keys())
    
    obj_dict = {}
    for index, row in ds_df.iterrows():
        fpath = await fetch_item(row['uri'], filename)
        
        data = np.loadtxt(fpath, comments='#')
        d = data[0] # distance bins
        rdf = data[1:]
                
        obj_dict.update({tuple(row[parameter_keys].values): {'dist': d, 'rdf': rdf}})
        print('.',end='')
        
    return obj_dict

async def get_rdf_df(
        ds_df, filename='default.txt', parameter_dict=DEFAULT_PARAMETER_DICT):
    
    parameter_keys = list(parameter_dict.keys())
    rdf_dict = await get_rdf_dict(ds_df, filename=filename, parameter_dict=parameter_dict)
    
    res_df = pd.DataFrame()
    for parameter_vals, item in rdf_dict.items():
        data = {k: v for k, v in zip(parameter_keys, parameter_vals)}
        for i, rdf in enumerate(item['rdf']):
            data.update({'step': i, 'dist': item['dist'], 'rdf': rdf})
            df = pd.DataFrame(data=data)
            res_df = pd.concat([res_df, df], ignore_index=True)

    res_mi_df = res_df.set_index([*parameter_keys, 'step' , 'dist'])
    return res_mi_df

def plot_rdf_df(
        rdf_df, filename='default.txt', parameter_dict=DEFAULT_PARAMETER_DICT):
    
    parameter_keys = list(parameter_dict.keys())
    
    unique_parameter_tuples = len(rdf_df[parameter_keys].groupby(parameter_keys))
    n = unique_parameter_tuples

    cols = 2 if n > 1 else 1
    rows = round(n/cols)
    if rows > 1:
        positions = [(i,j) for i in range(rows) for j in range(cols)][:n]
    else:
        positions = [i for i in range(cols)][:n]

    fig, ax = plt.subplots(rows,cols,figsize=(5*cols,4*rows))
    if not isinstance(ax, Iterable):
        ax = [ax]


    for pos, (key, grp) in zip(positions, rdf_df.groupby(by=parameter_keys)):
        steps_of_interest = {
            'first': int(grp['step'].unique().min()),
            'intermediate': int(grp['step'].unique().mean().round()),
            'last': int(grp['step'].unique().max())
        }
        for label, step in steps_of_interest.items():
            grp[ grp['step'] == step ].plot('dist', 'rdf', ax=ax[pos], label=label,title=str(key))

    # fig.tight_layout()
    return fig

async def get_rdf_plot_dict(ds_df, filenames, interval=(0.1,40.0)):
    rdf_plot_dict = {}
    for fn in filenames:
        rdf_df = await get_rdf_df(ds_df, fn)
        rdf_df.reset_index(inplace=True)
        rdf_plot_dict.update(
            {
                fn: plot_rdf_df(rdf_df[ 
                    (rdf_df['dist'] >= interval[0]) & (rdf_df['dist'] <= interval[1])
                ])})
        
    return rdf_plot_dict

# %% [markdown]
# ### Global settings

# %% init_cell=true
# pandas settings
pd.options.display.max_rows = 200
pd.options.display.max_columns = 16
pd.options.display.max_colwidth = 256

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
work_prefix = os.path.join( os.path.expanduser("~"), 'sandbox', date_prefix + '_sds_on_au_111_substrate_solvation_postp')

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
fp = FilePad.auto_load()

# %% [markdown]
# ## Conversion from LAMMPS data format to PDB

# %% [markdown]
# The following bash / tcl snippet converts a LAMMPS data file to PDB, assigning the desired names as mapped in a yaml file
# ```bash
# !/bin/bash
# echo "package require jlhvmd; jlh lmp2pdb indenter.lammps indenter.pdb" | vmd -eofexit
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

# %% [markdown]
# ## Overview

# %% [markdown]
# ### Overview on projects in database

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
res_df = pd.DataFrame(res)
res_df

# %% [markdown]
# ### Overview on steps in project

# %%
project_id = '2020-12-14-sds-on-au-111-substrate-passivation'

# %%
query = make_query({
    'project': project_id,
})

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
columns = ['step', 'earliest', 'latest', 'object_count']
res_df = pd.DataFrame(data=res, columns=columns) # pandas Dataframe is just nice for printing in notebook
res_df

# %%
res_df['step'].values

# %% [markdown]
# ## Packing visualization

# %% [markdown]
# ### Surfactant measures

# %%
query = make_query({
    'project': project_id,
    'step': {'$regex': 'SurfactantMoleculeMeasures'}
})

# %%
await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
uri = await get_uri_by_query(query)

# %%
uri

# %%
item_dict = await get_item_dict(uri)

# %%
item_dict

# %%
fpath = await fetch_item(uri, 'default.png')

# %%
Image(fpath)

# %% [markdown]
# ## Energy minimization after solvation analysis

# %% [markdown]
# ### Overview on datasets in step

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
    'step': {'$regex': 'GromacsEnergyMinimizationAfterSolvation'}
})

# %%
len(await dl.query(query))

# %%
ds_df = await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
ds_df = await get_df_by_aggregation(query)

# %%
ds_df

# %%
uri = ds_df.loc[0]['uri']

# %%
uri

# %%
await get_item_dict(uri)

# %% [markdown]
# ### Global observables

# %%
res_mi_df = await get_edr_df(ds_df)

# %%
res_df = res_mi_df.reset_index()

# %%
parameter_dict = DEFAULT_PARAMETER_DICT.copy()

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
for key, grp in res_df.groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=str(key),title=y_quantity)
        
fig.tight_layout()

# %%
fig, ax = plt.subplots(3,2,figsize=(10,12))
for key, grp in res_df[res_df['shape'] == 'monolayer'].groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time', y_quantity, ax=ax[position], label=str(key), title=y_quantity, legend=True)
        
fig.tight_layout()

# %%
fig, ax = plt.subplots(3,2,figsize=(10,12))
for key, grp in res_df[res_df['shape'] == 'hemicylinders'].groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time', y_quantity, ax=ax[position], label=str(key), title=y_quantity, legend=True)
        
fig.tight_layout()

# %% [markdown]
# ###  Visualize trajectory

# %%
obj_dict = await get_mp4_dict(ds_df)

# %%
len(obj_dict)

# %%
obj_dict

# %%
display(obj_dict[('monolayer',338)])

# %%
for key, obj in obj_dict.items():
    print(key)
    display(obj)

# %% [markdown]
# ## NVT equilibration analysis

# %% [markdown]
# ### Overview on datasets in step

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
    'step': {'$regex': 'GromacsNVTEquilibration'}
})

# %%
len(await dl.query(query))

# %%
ds_df = await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
ds_df = await get_df_by_aggregation(query)

# %%
ds_df

# %%
uri = ds_df.loc[0]['uri']

# %%
uri

# %%
await get_item_dict(uri)

# %% [markdown]
# ### Global observables

# %%
res_mi_df = await get_edr_df(ds_df)

# %%
res_df = res_mi_df.reset_index()

# %%
parameter_dict = DEFAULT_PARAMETER_DICT.copy()

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
for key, grp in res_df.groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()

# %%
fig, ax = plt.subplots(3,2,figsize=(10,12))
for key, grp in res_df[res_df['shape'] == 'monolayer'].groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time', y_quantity, ax=ax[position], label=key, title=y_quantity, legend=False)
        
fig.tight_layout()

# %%
fig, ax = plt.subplots(3,2,figsize=(10,12))
for key, grp in res_df[res_df['shape'] == 'hemicylinders'].groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time', y_quantity, ax=ax[position], label=key, title=y_quantity, legend=False)
        
fig.tight_layout()

# %% [markdown]
# ###  Visualize trajectory

# %%
obj_dict = await get_mp4_dict(ds_df)

# %%
len(obj_dict)

# %%
obj_dict

# %%
for key, obj in obj_dict.items():
    print(key)
    display(obj)

# %% [markdown]
# ## NPT equilibration analysis

# %% [markdown]
# ### Overview on datasets in step

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
    'step': {'$regex': 'GromacsNPTEquilibration'}
})

# %%
len(await dl.query(query))

# %%
ds_df = await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
ds_df = await get_df_by_aggregation(query)

# %%
ds_df

# %%
uri = ds_df.loc[0]['uri']

# %%
uri

# %%
await get_item_dict(uri)

# %% [markdown]
# ### Global observables

# %%
res_mi_df = await get_edr_df(ds_df)

# %%
res_df = res_mi_df.reset_index()

# %%
parameter_dict = DEFAULT_PARAMETER_DICT.copy()

# %%
parameter_keys = list(parameter_dict.keys())

# %%
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
    
fig, ax = plt.subplots(rows,cols,figsize=(10,12))
for key, grp in res_df.groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=str(key),title=y_quantity)
        
fig.tight_layout()

# %% [markdown]
# ###  Visualize trajectory

# %%
obj_dict = await get_mp4_dict(ds_df)

# %%
len(obj_dict)

# %%
obj_dict

# %%
for key, obj in obj_dict.items():
    print(key)
    display(obj)

# %% [markdown]
#
# ### Pre-evaluated RDF

# %%
rdf_file_names = sorted([fn for fn in await get_item_dict(uri) if fn.endswith('rdf.txt')])
rdf_file_names

# %%
rdf_plot_dict = await get_rdf_plot_dict(ds_df, rdf_file_names, interval=(0.5,35.0))

# %%
for fn, fig in rdf_plot_dict.items():
    print(fn)
    display(fig)

# %% [markdown]
# ## Relaxation analysis

# %% [markdown]
# ### Overview on datasets in step

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
    'step': {'$regex': 'GromacsRelaxation'}
})

# %%
len(await dl.query(query))

# %%
ds_df = await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
ds_df = await get_df_by_aggregation(query)

# %%
ds_df

# %%
uri = ds_df.loc[0]['uri']

# %%
uri

# %%
await get_item_dict(uri)

# %% [markdown]
# ### Global observables

# %%
res_mi_df = await get_edr_df(ds_df)

# %%
res_df = res_mi_df.reset_index()

# %%
parameter_dict = DEFAULT_PARAMETER_DICT.copy()

# %%
parameter_keys = list(parameter_dict.keys())

# %%
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
    
fig, ax = plt.subplots(rows,cols,figsize=(10,12))
for key, grp in res_df.groupby(parameter_keys):
    for y_quantity, position in zip(y_quantities, positions):
        grp.plot('Time',y_quantity,ax=ax[position],label=key,title=y_quantity)
        
fig.tight_layout()

# %% [markdown]
# ###  Visualize trajectory

# %%
obj_dict = await get_mp4_dict(ds_df)

# %%
len(obj_dict)

# %%
for key, obj in obj_dict.items():
    print(key)
    display(obj)

# %% [markdown]
#
# ### Pre-evaluated RDF

# %%
rdf_file_names

# %%
rdf_plot_dict = await get_rdf_plot_dict(ds_df, rdf_file_names, interval=(0.5,35.0))

# %%
for fn, fig in rdf_plot_dict.items():
    print(fn)
    display(fig)

# %%
