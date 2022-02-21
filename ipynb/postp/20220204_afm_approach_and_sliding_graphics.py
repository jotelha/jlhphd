# -*- coding: utf-8 -*-
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
# # AFM probe on substrate preparation evaluation

# %% [markdown]
# This notebook demonstrates querying of Fireworks workflows and Filepad objects

# %% [markdown]
# ## Initialization

# %% [markdown]
# ### IPython magic

# %% init_cell=true
# %config Completer.use_jedi = False

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

# dtool
import dtool_lookup_api.asynchronous as dl
import dtoolcore

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

import scipy.constants as C

from cycler import cycler
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# for ovito visualization
 # Initialization of Qt event loop in the backend:
from PySide2 import QtCore, QtGui
# %gui qt5

from ovito.io import *
from ovito.vis import *
from ovito.modifiers import *

# Gaussion Process Regression
from SurfaceTopography.Support.Regression import (
    gaussian_process_regression, make_grid, gaussian_kernel)

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


# %% init_cell=true
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

# %% init_cell=true
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
async def read_forces(uri, file_name='default.txt'): 
    item_dict = await get_item_dict(uri)
    d = dtoolcore.DataSet.from_uri(uri)
    fpath = d.item_content_abspath(item_dict[file_name])


    df = pd.read_csv(fpath, delim_whitespace=True)
    # df.set_index('Step', inplace=True)

    return df

# %%
ase.__version__

# %%
import ase.io.lammpsdata.read_lammps_data

# %% init_cell=true
async def read_lammps_config(uri, file_name='default.lammps'): 
    item_dict = await get_item_dict(uri)
    d = dtoolcore.DataSet.from_uri(uri)
    fpath = d.item_content_abspath(item_dict[file_name])

    atoms = ase.io.lammpsdata.read_lammps_data(fpath, units='real')
    return atoms

# %% [markdown]
# ### Global settings

# %% init_cell=true
# plotting# matplotlib settings

# expecially for presentation, larger font settings for plotting are recommendable
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlex

plt.rcParams["figure.figsize"] = (16,10) # the standard figure size

plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 14
plt.rcParams["lines.markeredgewidth"]=1 

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

custom_linestyles = [l[1] for l in [*linestyle_str, *linestyle_tuple]]

dpi = 300

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
#fp = FilePad.from_db_file(
#    os.path.join(os.path.expanduser("~"), '.fireworks', 'fireworks_sandbox_mongodb_auth.yaml'))

# %% [markdown]
# ### Units
#
# $ [F_{LMP}] = \frac{ \mathrm{kcal}}{ \mathrm{mol} \cdot \mathrm{\mathring{A}}} $
#
# $ [F_{PLT}] = \mathrm{nN}$
#
# $ \mathrm{kcal} = 4.184 \mathrm{kJ} = 4.184 \cdot 10^{3} \mathrm{J}$
#
# $ J = N \cdot m$
#
# $ N = J m^{-1} = \frac{ \mathrm{kcal}}{ 4.184 \cdot 10^3 \mathrm{m} }
#     = \frac{ \mathrm{kcal}}{ 4.184 \cdot 10^{13} \mathrm{\mathring{A}} }$
#     
# $ [F_{PLT}] = \mathrm{nN} 
#     = \frac{ 10^9 \cdot N_A^{-1}}{ 4.184 \cdot 10^{13} } 
#         \frac{\mathrm{kcal}}{\mathrm{mol} \cdot \mathrm{\mathring{A}}}
#     = \frac{ 10^{-4} \cdot N_A^{-1}}{ 4.184 } [F_{LMP}]
#     = 0.239 \cdot 10^{-4} N_A^{-1} [F_{LMP}] $
#     
#     
# $ \frac{\mathrm{ kcal }}{ {\mathrm{mol} \mathrm{\mathring{A}}}} = 1.66053892103219 \cdot 10^{-11} \frac{\mathrm{J}}{\mathrm{m}}$

# %% init_cell=true
kCal_to_J = C.calorie * 1e3 # kCal -> J

kCal_per_Ang_to_J_per_m = C.calorie * 1e3 / C.angstrom # kCal / Ang -> J / m

kCal_per_Ang_to_nN =C.calorie * 1e3 / C.angstrom *1e9 # kCal / Ang -> n J / m = n N

force_conversion_factor = C.calorie * 1e3 / C.angstrom *1e9 / C.Avogadro# kCal / (mol * Ang ) -> n N

velocity_conversion_factor = 1.0e5 # from Ang per fs to m per s 
# (1 Ang / 1 fs = 10^15 Ang / s = 10^15 m / 10^10 s)

concentration_conversion_factor = 100 # from Ang^-2 to nm^-2

# velocity_conversion_factor=1


# %% [markdown]
# # Fireworks

# %%
project = '2021-01-28-sds-on-au-111-probe-and-substrate-approach'

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

# %% init_cell=true
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
#
# ## 2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration

# %%
# project_id = "2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach"
project_id = "2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"

# %%
# query = {
#    'readme.project': project_id,
#    'readme.step': {'$regex': 'LAMMPSProbeNormalApproach'},
# }

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'WrapJoinDataFile'},
}

# %%
query

# %% [markdown]
# ### All

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
res_df = pd.DataFrame(res)

# %%
res_df

# %%
res_df.to_json(orient='records')

# %%

# %%
distinct_parameter_values = {k: set(res_df[k].unique()) for k in parameters.keys()}

# distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}
# filter out unwanted values
distinct_parameter_values = {k: [e.item() for e in p if (
        isinstance(e, np.float64) and not np.isnan(e)) or (
        not isinstance(e, np.float64) and e is not None)] 
    for k, p in distinct_parameter_values.items()}
distinct_parameter_values

# filter out unwanted values


# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSProbeNormalApproach'},
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
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
res_df = pd.DataFrame(res)

# %%
res_df

# %%
list_of_tuples = [(row['x_shift'], row['y_shift'], row['uuid'][0]) 
    for _, row in res_df[['x_shift','y_shift','uuid']].iterrows()]

# %%
list_of_tuples

# %%
dict_of_tuples = {(x,y): uuid for x,y,uuid in list_of_tuples}

# %%
dict_of_tuples

# %%
offset_tuple = (0,50)

# %%
uuid = dict_of_tuples[offset_tuple]

# %%
uuid

# %% [markdown]
# ### One

# %%
# manually specify uuid
uuid = 'a3452e3a-3ca8-4795-950f-534223fcf916'

# %%
lookup_res = await dl.lookup(uuid)
assert len(lookup_res) == 1
uri = lookup_res[0]['uri']

# %%
uri

# %%
file_name = 'default.lammps'
item_dict = await get_item_dict(uri)
d = dtoolcore.DataSet.from_uri(uri)
fpath = d.item_content_abspath(item_dict[file_name])

# %%
fpath

# %%
import math
import ovito 
# Boilerplate code generated by OVITO Pro 3.6.0
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *


# %%
def system_info_modifier(frame, data):
    """Print system information."""
    # an additional attribute holding the source directory of the loaded file:
    if "SourceFile" in data.attributes:
        from os.path import dirname
        data.attributes["SourceDir"] = dirname( data.attributes["SourceFile"] )
        from os import chdir
        chdir(data.attributes["SourceDir"])
    
    print("There are %i atrributes with the following values:" % len(data.attributes))

    for attribute_name, attribute_value in data.attributes.items():
        print("  '{:24s}: {}'".format(attribute_name, attribute_value))
    
    print("")
    
    if data.particles != None:
        print("There are %i particles with the following properties:" % data.particles.count)
        for property_name in data.particles.keys():
            print("  '%s'" % property_name)



# %%
# Manual modifications of the imported data objects:
def modify_pipeline_input(frame: int, data: DataCollection):
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(1).color = (1.0, 0.4, 0.4)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(2).color = (0.4, 0.4, 1.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(3).color = (1.0, 1.0, 0.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(4).color = (1.0, 0.4, 1.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(5).color = (0.4, 1.0, 0.2)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(6).color = (0.8, 1.0, 0.7)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(7).color = (0.7, 0.0, 1.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(8).color = (0.2, 1.0, 1.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(9).color = (0.97, 0.97, 0.97)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(10).color = (1.0, 0.4, 0.4)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(11).color = (0.4, 0.4, 1.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(12).color = (1.0, 1.0, 0.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(13).color = (1.0, 0.4, 1.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(14).color = (0.4, 1.0, 0.2)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(15).color = (0.8, 1.0, 0.7)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(16).color = (0.7, 0.0, 1.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(17).color = (0.2, 1.0, 1.0)
    data.particles_.dihedrals_['Dihedral Type_'].type_by_id_(18).color = (0.97, 0.97, 0.97)
    data.particles_.angles_['Angle Type_'].type_by_id_(1).color = (1.0, 0.4, 0.4)
    data.particles_.angles_['Angle Type_'].type_by_id_(2).color = (0.4, 0.4, 1.0)
    data.particles_.angles_['Angle Type_'].type_by_id_(3).color = (1.0, 1.0, 0.0)
    data.particles_.angles_['Angle Type_'].type_by_id_(4).color = (1.0, 0.4, 1.0)
    data.particles_.angles_['Angle Type_'].type_by_id_(5).color = (0.4, 1.0, 0.2)
    data.particles_.angles_['Angle Type_'].type_by_id_(6).color = (0.8, 1.0, 0.7)
    data.particles_.angles_['Angle Type_'].type_by_id_(7).color = (0.7, 0.0, 1.0)
    data.particles_.angles_['Angle Type_'].type_by_id_(8).color = (0.2, 1.0, 1.0)
    data.particles_.angles_['Angle Type_'].type_by_id_(9).color = (0.97, 0.97, 0.97)
    data.particles_.angles_['Angle Type_'].type_by_id_(10).color = (1.0, 0.4, 0.4)
    data.particles_.angles_['Angle Type_'].type_by_id_(11).color = (0.4, 0.4, 1.0)
    data.particles_.angles_['Angle Type_'].type_by_id_(12).color = (1.0, 1.0, 0.0)
    data.particles_.angles_['Angle Type_'].type_by_id_(13).color = (1.0, 0.4, 1.0)
    data.particles_.bonds_.bond_types_.type_by_id_(1).color = (0.7, 0.0, 1.0)
    data.particles_.bonds_.bond_types_.type_by_id_(2).color = (0.2, 1.0, 1.0)
    data.particles_.bonds_.bond_types_.type_by_id_(3).color = (1.0, 0.4, 1.0)
    data.particles_.bonds_.bond_types_.type_by_id_(4).color = (0.4, 1.0, 0.4)
    data.particles_.bonds_.bond_types_.type_by_id_(5).color = (1.0, 0.4, 0.4)
    data.particles_.bonds_.bond_types_.type_by_id_(6).color = (0.4, 0.4, 1.0)
    data.particles_.bonds_.bond_types_.type_by_id_(7).color = (1.0, 1.0, 0.7)
    data.particles_.bonds_.bond_types_.type_by_id_(8).color = (0.0, 0.0, 0.0)
    data.particles_.bonds_.bond_types_.type_by_id_(9).color = (1.0, 1.0, 0.0)
    data.particles_.particle_types_.type_by_id_(1).mass = 1.008
    data.particles_.particle_types_.type_by_id_(1).name = 'HAL2'
    data.particles_.particle_types_.type_by_id_(2).mass = 1.008
    data.particles_.particle_types_.type_by_id_(2).name = 'HAL3'
    # data.particles_.particle_types_.type_by_id_(3).color = (0.5647058823529412, 0.5647058823529412, 0.5647058823529412)
    data.particles_.particle_types_.type_by_id_(3).color = (0.7, 0.7, 0.7)
    data.particles_.particle_types_.type_by_id_(3).mass = 12.011
    data.particles_.particle_types_.type_by_id_(3).name = 'CTL2'
    data.particles_.particle_types_.type_by_id_(3).radius = 0.5
    data.particles_.particle_types_.type_by_id_(3).vdw_radius = 0.1
    # data.particles_.particle_types_.type_by_id_(4).color = (0.5647058823529412, 0.5647058823529412, 0.5647058823529412)
    data.particles_.particle_types_.type_by_id_(4).color = (0.7, 0.7, 0.7)
    data.particles_.particle_types_.type_by_id_(4).mass = 12.011
    data.particles_.particle_types_.type_by_id_(4).name = 'CTL3'
    data.particles_.particle_types_.type_by_id_(4).radius = 0.6
    data.particles_.particle_types_.type_by_id_(5).color = (0.9333333333333333, 0.12549019607843137, 0.06274509803921569)
    data.particles_.particle_types_.type_by_id_(5).mass = 15.9994
    data.particles_.particle_types_.type_by_id_(5).name = 'OSL'
    data.particles_.particle_types_.type_by_id_(5).radius = 0.7
    data.particles_.particle_types_.type_by_id_(6).color = (0.9333333333333333, 0.12549019607843137, 0.06274509803921569)
    data.particles_.particle_types_.type_by_id_(6).mass = 15.9994
    data.particles_.particle_types_.type_by_id_(6).name = 'O2L'
    data.particles_.particle_types_.type_by_id_(6).radius = 0.7
    data.particles_.particle_types_.type_by_id_(7).color = (1.0, 1.0, 0.0)
    data.particles_.particle_types_.type_by_id_(7).mass = 32.060001
    data.particles_.particle_types_.type_by_id_(7).name = 'SL'
    data.particles_.particle_types_.type_by_id_(7).radius = 1.2
    data.particles_.particle_types_.type_by_id_(8).mass = 1.008
    data.particles_.particle_types_.type_by_id_(8).name = 'HT'
    data.particles_.particle_types_.type_by_id_(9).color = (0.9333333333333333, 0.12549019607843137, 0.06274509803921569)
    data.particles_.particle_types_.type_by_id_(9).mass = 15.9994
    data.particles_.particle_types_.type_by_id_(9).name = 'OT'
    data.particles_.particle_types_.type_by_id_(10).color = (0.6705882352941176, 0.3607843137254902, 0.9490196078431372)
    data.particles_.particle_types_.type_by_id_(10).mass = 22.989771
    data.particles_.particle_types_.type_by_id_(10).name = 'SOD'
    data.particles_.particle_types_.type_by_id_(10).radius = 1.2
    data.particles_.particle_types_.type_by_id_(11).color = (0.8549019607843137, 0.6470588235294118, 0.12549019607843137)
    data.particles_.particle_types_.type_by_id_(11).mass = 196.966995
    data.particles_.particle_types_.type_by_id_(11).name = 'AU'
    data.particles_.particle_types_.type_by_id_(11).radius = 1.1


# %%
ovito.scene.pipelines

# %%
try: # https://www.ovito.org/forum/topic/visualization-elements-persist/
    del ovito.scene.pipelines[:]
except:
    pass

# Import a simulation model and set up a data pipeline:
pipeline = import_file(fpath, atom_style = 'full')
pipeline.modifiers.append(modify_pipeline_input)

# Visual element initialization:
# data = pipeline.compute() # Evaluate new pipeline to gain access to visual elements associated with the imported data objects.
# data.cell.vis.enabled = False
# data.particles.vis.radius = 1.0
# del data # Done accessing input DataCollection of pipeline.

pipeline.add_to_scene()
# pipeline.modifiers.clear()


pipeline.modifiers.append(system_info_modifier)

# Slice: remove bulk of gold slab
pipeline.modifiers.append(SliceModifier(
    distance = -21.0, 
    normal = (0.0, 0.0, 1.0), 
    inverse = True, 
    operate_on = {'surfaces', 'dislocations', 'particles'}, 
    gridslice_vis = None))

# remove water and hydrogens

# Select type:
pipeline.modifiers.append(SelectTypeModifier(types = {8, 9, 2, 1}))

# Delete selected:
pipeline.modifiers.append(DeleteSelectedModifier(operate_on = {'bonds', 'particles'}))

# make gold transparent

# Select type:
pipeline.modifiers.append(SelectTypeModifier(types = {11}))

# Compute property:
pipeline.modifiers.append(ComputePropertyModifier(
    expressions = ('0.8',), 
    output_property = 'Transparency', 
    only_selected = True))


# Clear selection:
pipeline.modifiers.append(ClearSelectionModifier())

# only show desired slab

# Slice:
pipeline.modifiers.append(SliceModifier(
    distance = 65.0, 
    normal = (0.0, 1.0, 0.0), 
    inverse = True, 
    operate_on = {'surfaces', 'dislocations', 'particles'}, 
    gridslice_vis = None))

# Slice:
pipeline.modifiers.append(SliceModifier(
    distance = 85.0, 
    normal = (0.0, 1.0, 0.0), 
    operate_on = {'surfaces', 'dislocations', 'particles'}, 
    gridslice_vis = None))

plane = 'xy'

# Simulation cell off
cell_vis = pipeline.source.data.cell.vis
cell_vis.render_cell = False

z_offset = 45

try:
    del vp
except:
    pass

# Viewport setup:
vp = Viewport()
vp.type = Viewport.Type.Front
vp.fov = 100
vp.camera_dir = (1.0, 0.0, -0.0)
vp.camera_pos = (75.45827106078428, 73.1449985, 10.730965867318673+z_offset)

if plane == 'xz':
    vp.overlays.append(CoordinateTripodOverlay(
        size = 0.07, 
        offset_x = 0.01, 
        axis1_enabled = False, 
        axis2_dir = (1.0, 0.0, 0.0)))
elif plane == 'yz':
    vp.overlays.append(CoordinateTripodOverlay(
        size = 0.07, 
        offset_x = 0.01, 
        axis3_enabled = False, 
        axis2_dir = (0.0, 0.0, 1.0)))
elif plane == 'xy':
    vp.overlays.append(CoordinateTripodOverlay(
        size = 0.07, 
        offset_x = 0.01, 
        axis2_enabled = False))

vp.render_image(filename='section.png', size=(900, 900))

# %%
widget = vp.create_jupyter_widget(width=800, height=600, device_pixel_ratio=2.0)

# %%
display(widget)

# %%
try: # https://www.ovito.org/forum/topic/visualization-elements-persist/
    del ovito.scene.pipelines[:]
except:
    pass

# Import a simulation model and set up a data pipeline:
pipeline = import_file(fpath, atom_style = 'full')
pipeline.modifiers.append(modify_pipeline_input)

# Visual element initialization:
# data = pipeline.compute() # Evaluate new pipeline to gain access to visual elements associated with the imported data objects.
# data.cell.vis.enabled = False
# data.particles.vis.radius = 1.0
# del data # Done accessing input DataCollection of pipeline.

pipeline.add_to_scene()
# pipeline.modifiers.clear()


pipeline.modifiers.append(system_info_modifier)

# Slice: remove bulk of gold slab
pipeline.modifiers.append(SliceModifier(
    distance = -21.0, 
    normal = (0.0, 0.0, 1.0), 
    inverse = True, 
    operate_on = {'surfaces', 'dislocations', 'particles'}, 
    gridslice_vis = None))

# remove water and hydrogens

# Select type:
pipeline.modifiers.append(SelectTypeModifier(types = {8, 9, 2, 1}))

# Delete selected:
pipeline.modifiers.append(DeleteSelectedModifier(operate_on = {'bonds', 'particles'}))

# make gold transparent

# Select type:
pipeline.modifiers.append(SelectTypeModifier(types = {11}))
# Compute property:
pipeline.modifiers.append(ComputePropertyModifier(
    expressions = ('0.75',), 
    output_property = 'Transparency', 
    only_selected = True))
# Clear selection:
pipeline.modifiers.append(ClearSelectionModifier())

# make sodium transparent

# Select type:
pipeline.modifiers.append(SelectTypeModifier(types = {10}))
# Compute property:
pipeline.modifiers.append(ComputePropertyModifier(
    expressions = ('0.2',), 
    output_property = 'Transparency', 
    only_selected = True))
# Clear selection:
pipeline.modifiers.append(ClearSelectionModifier())

# make carbon slightly transparent

# Select type:
pipeline.modifiers.append(SelectTypeModifier(types = {3,4}))
# Compute property:
pipeline.modifiers.append(ComputePropertyModifier(
    expressions = ('0.2',), 
    output_property = 'Transparency', 
    only_selected = True))
# Clear selection:
pipeline.modifiers.append(ClearSelectionModifier())


# Simulation cell off
cell_vis = pipeline.source.data.cell.vis
cell_vis.render_cell = False

z_offset = 45

try:
    del vp
except:
    pass

# Viewport setup:
vp = Viewport()
vp.type = Viewport.Type.Perspective


vp.camera_dir = (-18.0, -12.0, -5.0)
#vp.camera_pos = (491.78476554287033, 489.82855104287034, 218.4637027559765)
vp.camera_pos = (581.5954758594607, 411.11767207297385, 151.0213317232804)
vp.fov = math.radians(20)
#vp.zoom_all()

vp.render_image(filename='perspective.png', size=(900, 900))

# %%
vp.camera_pos

# %%
cell_vis = pipeline.source.data.cell.vis

# %%
cell_vis.enabled

# %%
vp.

# %%
tachyon = TachyonRenderer(shadows=False, direct_light_intensity=1.1)
vp.render_image(filename="tachyon.png", size=(900, 900), background=(1,1,1), renderer=tachyon)

tachyon_shadows = TachyonRenderer(shadows=True, direct_light_intensity=1.1)
vp.render_image(filename="tachyon_shadows.png", size=(900, 900), background=(1,1,1), renderer=tachyon_shadows)

tachyon_dof = TachyonRenderer(shadows=False, depth_of_field=True, direct_light_intensity=1.1, focal_length=300.)
vp.render_image(filename="tachyon_dof.png", size=(900, 900), background=(1,1,1), renderer=tachyon_dof)

tachyon_dof_shadows = TachyonRenderer(shadows=True, depth_of_field=True, direct_light_intensity=1.1)
vp.render_image(filename="tachyon_dof_shadows.png", size=(900, 900), background=(1,1,1), renderer=tachyon_dof_shadows)

# %%

# %%

# %%
# select gold indenter, color red
pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {11}))
pipeline.modifiers.append(SliceModifier(distance=0, normal=(0,0,1), 
                          select=True, only_selected=True))
pipeline.modifiers.append(AssignColorModifier(color=(1,0,0)))

# %%
# remove surfactant on top of indenter
pipeline.modifiers.append(ClearSelectionModifier())
pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {1,2,3,4,5,6,7}))
pipeline.modifiers.append(SliceModifier(distance=10, normal=(0,0,1), 
                          select=True, only_selected=True))
pipeline.modifiers.append(DeleteSelectedModifier())


# %%
vp.camera_pos

# %%
vp.overlays.append(CoordinateTripodOverlay(
    size = 0.07, 
    offset_x = 0.325, 
    axis1_enabled = False, 
    axis2_dir = (1.0, 0.0, 0.0)))
vp.overlays.append(CoordinateTripodOverlay(
    size = 0.07, 
    offset_x = 0.635, 
    axis3_enabled = False, 
    axis2_dir = (0.0, 0.0, 1.0)))
vp.overlays.append(CoordinateTripodOverlay(
    size = 0.07, 
    offset_x = 0.01, 
    axis2_enabled = False))

# %%
# Create a virtual viewport:
# vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(0, 0, -1))
# vp.zoom_all(size=(800, 600))

# Create the visual viewport widget:


# %%
widget = vp.create_jupyter_widget(width=800, height=600, device_pixel_ratio=2.0)

# %%
display(widget)

# %%
pipeline.source.data.cell.vis.rendering_color = (1,1,1)

# %%
pipeline.source.data.cell.vis.enabled

# %%
try: # https://www.ovito.org/forum/topic/visualization-elements-persist/
    del ovito.scene.pipelines[:]
except:
    pass

# %%

# %%
display(widget)

# %%
# pipeline.display_color?

# %%
data = pipeline.compute()

# %%
cell_vis.render_cell

# %%
data.cell

# %%
vp.zoom_all()

# %%
try: # https://www.ovito.org/forum/topic/visualization-elements-persist/
    del ovito.scene.pipelines[:]
except:
    pass

# %%

# %%
# remove water
pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {8,9}))
pipeline.modifiers.append(DeleteSelectedModifier())

# %%

# Data import:
pipeline = import_file(fpath, atom_style = 'full')
pipeline.add_to_scene()

# Configuring visual elements associated with imported dataset:
pipeline.compute().particles.vis.radius = 1.0
pipeline.compute().particles.bonds.vis.width = 1.0

# User-defined modifier 'System Information':
# CenterOfMass.py
#
# June 2019, Johannes Hoermann, johannes.hoermann@imtek.uni-freiburg.de
#
# displays system information and changes the working directory to the directory
# containing "SourceFile", usually the last datafile or trajectory loaded in the course of
# pipeline execution (?)
#
from ovito.data import *



# Slice:
pipeline.modifiers.append(SliceModifier(
    distance = -21.0, 
    normal = (0.0, 0.0, 1.0), 
    inverse = True, 
    operate_on = {'dislocations', 'surfaces', 'particles'}, 
    gridslice_vis = None))

# Select type:
pipeline.modifiers.append(SelectTypeModifier(types = {1, 2, 8, 9, 10}))

# Delete selected:
pipeline.modifiers.append(DeleteSelectedModifier(operate_on = {'particles', 'bonds'}))

# Select type:
pipeline.modifiers.append(SelectTypeModifier(types = {11}))

# Compute property:
pipeline.modifiers.append(ComputePropertyModifier(
    expressions = ('0.9',), 
    output_property = 'Transparency', 
    only_selected = True))

# Clear selection:
pipeline.modifiers.append(ClearSelectionModifier())

# Expression selection:
pipeline.modifiers.append(ExpressionSelectionModifier(
    expression = 'ParticleType == 11 && Position.Z > 5', 
    enabled = False))

# Load trajectory:
# mod = LoadTrajectoryModifier()
# mod.source.load('/p/scratch/chfr13/hoermann/fireworks/launchpad/launcher_2019-09-09-13-51-12-345121/default.nc')
# mod.enabled = False
# pipeline.modifiers.append(mod)

# Displacement vectors:
# pipeline.modifiers.append(CalculateDisplacementsModifier(enabled = False))

# Color coding:
pipeline.modifiers.append(ColorCodingModifier(
    property = 'f_storeForcesAve.1', 
    start_value = -0.3222331404685974, 
    end_value = 0.30041250586509705, 
    gradient = ColorCodingModifier.Jet(), 
    only_selected = True, 
    enabled = False))

# Slice:
pipeline.modifiers.append(SliceModifier(
    distance = 70.0, 
    inverse = True, 
    operate_on = {'dislocations', 'surfaces', 'particles'}, 
    gridslice_vis = None))

# Slice:
pipeline.modifiers.append(SliceModifier(
    distance = 90.0, 
    operate_on = {'dislocations', 'surfaces', 'particles'}, 
    gridslice_vis = None))

# Affine transformation:
pipeline.modifiers.append(AffineTransformationModifier(
    transformation = [[6.123233995736766e-17, -1.0, 0.0, 320.0], [1.0, 6.123233995736766e-17, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], 
    target_cell = [[146.949402, 0.0, 0.0, 2.554503], [0.0, 149.724899, 0.0, -0.729928], [0.0, 0.0, 295.96873780610133, -163.217895]], 
    operate_on = {'surfaces', 'vector_properties', 'particles', 'cell', 'voxels'}))

# Viewport setup:
vp = Viewport()
vp.type = Viewport.Type.Front
vp.fov = 50.987614558889206
vp.camera_dir = (-0.0, 1.0, -0.0)
vp.camera_pos = (244.72972652093927, 79.99436202079853, 6.026430513480165)
vp.overlays.append(TextLabelOverlay(
    offset_x = 0.661703887510339, 
    offset_y = -0.0827129859387924, 
    text = '10 m / s probe velocity, dt = 2fs', 
    enabled = False))
vp.overlays.append(TextLabelOverlay(
    offset_x = 0.67, 
    offset_y = 0.0019693568080664837, 
    font_size = 0.03, 
    text = 'time step #[time]', 
    enabled = False))
vp.overlays.append(ColorLegendOverlay(
    offset_x = 0.3150970892906376, 
    offset_y = 0.1221001221001221, 
    legend_size = 0.28, 
    format_string = '%g Å ', 
    enabled = False))
vp.overlays.append(CoordinateTripodOverlay(
    size = 0.07, 
    offset_x = 0.305, 
    axis1_enabled = False, 
    axis2_dir = (1.0, 0.0, 0.0), 
    enabled = False))
vp.overlays.append(CoordinateTripodOverlay(
    size = 0.07, 
    offset_x = 0.615, 
    axis3_enabled = False, 
    axis2_dir = (0.0, 0.0, 1.0), 
    enabled = False))
vp.overlays.append(CoordinateTripodOverlay(
    size = 0.07, 
    offset_x = -0.01, 
    axis2_enabled = False, 
    enabled = False))

# Rendering:
vp.render_image(filename='test.png', size=(4800, 900))

# %%
vp.render_image(filename='test.png', size=(4800, 900))

# %%
# remove all surfactant
# pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {1,2,3,4,5,6,7}))
# pipeline.modifiers.append(DeleteSelectedModifier())

# %%
import ovito
del ovito.scene.pipelines[:]

# %%
# iter

# %%
try: # https://www.ovito.org/forum/topic/visualization-elements-persist/
    del ovito.scene.pipelines[:]
except:
    pass

for offset_tuple, uuid in dict_of_tuples.items():
    x_shift, y_shift = offset_tuple
    
    lookup_res = await dl.lookup(uuid)
    
    if len(lookup_res) != 1:
        logger.warning(
            "Not exactly one lookup result for uuid {}, but {}".format(
                uuid, len(lookup_res)))
        continue
        
    uri = lookup_res[0]['uri']
    logger.info("Process offset ({},{}): {}".format(x_shift, y_shift, uri))

    file_name = 'default.lammps'
    item_dict = await get_item_dict(uri)
    if file_name not in item_dict:
        logger.warning("{} not in dataset!".format(file_name))
        logger.info(item_dict)
        continue
    d = dtoolcore.DataSet.from_uri(uri)
    
    fpath = d.item_content_abspath(item_dict[file_name])
    
    logger.info("Process local {}".format(fpath))


    # Import a simulation model and set up a data pipeline:
    pipeline = import_file(fpath)
    pipeline.add_to_scene()
    # pipeline.modifiers.clear()

    # remove water
    pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {8,9}))
    pipeline.modifiers.append(DeleteSelectedModifier())

    # select gold indenter, color red
    pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {11}))
    pipeline.modifiers.append(SliceModifier(distance=0, normal=(0,0,1), 
                              select=True, only_selected=True))
    pipeline.modifiers.append(AssignColorModifier(color=(1,0,0)))

    # remove surfactant on top of indenter
    pipeline.modifiers.append(ClearSelectionModifier())
    pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {1,2,3,4,5,6,7}))
    pipeline.modifiers.append(SliceModifier(distance=10, normal=(0,0,1), 
                              select=True, only_selected=True))
    pipeline.modifiers.append(DeleteSelectedModifier())

    # remove all surfactant
    # pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {1,2,3,4,5,6,7}))
    # pipeline.modifiers.append(DeleteSelectedModifier())

    vp.zoom_all()
    
    filename = 'final_config_x_{}_y_{}.png'.format(x_shift,y_shift)
    
    logger.info("Render {}".format(filename))

    qimage = vp.render_image(
         size=(640, 480), frame=0, 
         filename=filename, 
         background=(1.0, 1.0, 1.0), 
         alpha=False, renderer=None, crop=False, layout=None)
    
    pipeline.remove_from_scene()

# %%
# y: direction across hemicylinders
# x: direction along hemicylinders


# %% [markdown]
#
# ## 2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration

# %%
# project_id = "2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach"
project_id = "2021-12-30-sds-on-au-111-probe-on-substrate-lateral-sliding"

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSProbeLateralSliding'},
}

# %%
query

# %% [markdown]
# ### All

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
res_df = pd.DataFrame(res)

# %%
res_df

# %%
res_df.to_json(orient='records')

# %%
distinct_parameter_values = {k: set(res_df[k].unique()) for k in parameters.keys()}

# distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}
# filter out unwanted values
distinct_parameter_values = {k: [e.item() for e in p if (
        isinstance(e, np.float64) and not np.isnan(e)) or (
        not isinstance(e, np.float64) and e is not None)] 
    for k, p in distinct_parameter_values.items()}
distinct_parameter_values

# filter out unwanted values


# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSProbeLateralSliding'},
    **{parameters[label]: {'$in': values} for label, values in distinct_parameter_values.items()},
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
            **{k: '$_id.{}'.format(k) for k in parameters.keys()},
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
res_df = pd.DataFrame(res)

# %%
res_df

# %%
list_of_tuples = [(row['x_shift'], row['y_shift'], row['distance'], row['uuid'][0]) 
    for _, row in res_df[['x_shift','y_shift','distance', 'uuid']].iterrows()]

# %%
list_of_tuples

# %%
dict_of_tuples = {(x,y,z): uuid for x,y,z,uuid in list_of_tuples}

# %%
dict_of_tuples

# %%
offset_tuple = (25,0,20)

# %%
uuid = dict_of_tuples[offset_tuple]

# %%
uuid

# %% [markdown]
# ### One

# %%
lookup_res = await dl.lookup(uuid)
assert len(lookup_res) == 1
uri = lookup_res[0]['uri']

# %%
uri

# %%
#file_name = 'default.lammps'
item_dict = await get_item_dict(uri)
# d = dtoolcore.DataSet.from_uri(uri)
# fpath = d.item_content_abspath(item_dict[file_name])

# %%
item_dict

# %%
