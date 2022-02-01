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
# # AFM probe on substrate lateral sliding evaluation

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
from matplotlib.lines import Line2D
from matplotlib.legend import Legend


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

# %% init_cell=true
async def read_lammps_config(uri, file_name='default.lammps'): 
    item_dict = await get_item_dict(uri)
    d = dtoolcore.DataSet.from_uri(uri)
    fpath = d.item_content_abspath(item_dict[file_name])

    atoms = ase.io.lammpsdata.read_lammps_data(fpath, units='real')
    return atoms


# %% init_cell=true
# https://www.titanwolf.org/Network/q/7ec46b59-d932-4fd4-af53-72604f6bf66c/y
def copy_attributes(obj2, obj1, attr_list):
        for i_attribute  in attr_list:
            getattr(obj2, 'set_' + i_attribute)( getattr(obj1, 'get_' + i_attribute)() )

def list_transferable_attributes(obj, except_attributes):
    obj_methods_list = dir(obj)

    obj_get_attr = []
    obj_set_attr = []
    obj_transf_attr =[]

    for name in obj_methods_list:
        if len(name) > 4:
            prefix = name[0:4]
            if prefix == 'get_':
                obj_get_attr.append(name[4:])
            elif prefix == 'set_':
                obj_set_attr.append(name[4:])

    for attribute in obj_set_attr:
        if attribute in obj_get_attr and attribute not in except_attributes:
            obj_transf_attr.append(attribute)

    return obj_transf_attr


# %%
from matplotlib.legend_handler import HandlerLine2D


# %% init_cell=true
class OpaqueSytelHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
    
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        except_attributes = ('transform', 'figure', 'contains', 'picker')
        transferable_attributes = list_transferable_attributes(orig_handle, except_attributes)    
        transform = handlebox.get_transform()
        
        artist_handle = type(orig_handle)([],[])
        copy_attributes(artist_handle, orig_handle, transferable_attributes)
        # artist_handle = copy.copy(orig_handle)
        artist_handle.set_transform(transform)
        # artist_handle.set_xdata([x0, x0+width])
        # artist_handle.set_ydata([y0+height/2, y0+height/2])
        artist_handle.set_xdata([x0+width/2.])
        artist_handle.set_ydata([y0+height/2])
        
        artist_handle.set_alpha(1.)
        handlebox.add_artist(artist_handle)
        
        return artist_handle


# %% init_cell=true
def save_fig(fig, ax, prefix):
    # first with legend
    ax.legend()
    fig.savefig(f'{prefix}_with_legend.svg')
    fig.savefig(f'{prefix}_with_legend.png')
    
    # then only legend
    fig_legend = plt.figure()
    ax_legend = fig_legend.add_subplot(111)
    
    
    legend_handles_labels = ax.get_legend_handles_labels()
    copied_handles = [copy.copy(handle) for handle in legend_handles_labels[0]]
    labels = legend_handles_labels[1]
        
    ax_legend.legend(copied_handles, labels, loc='center')
    ax_legend.axis('off')
    fig_legend.savefig(f'{prefix}_legend.svg')
    fig_legend.savefig(f'{prefix}_legend.png')

    # then without legend
    ax.get_legend().remove()
    fig.savefig(f'{prefix}.svg')
    fig.savefig(f'{prefix}.png')

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

plt.rcParams["lines.linewidth"] = 2
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

default_line_handler = Legend.get_default_handler_map()[Line2D]

Legend.update_default_handler_map({Line2D: OpaqueSytelHandler()})

# for switching back to default line renderer, use 
# Legend.update_default_handler_map({Line2D: default_line_handler})

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
# ## Repeated DPD equilibration

# %% [markdown]
# ### Overview on steps in project

# %% init_cell=true
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
# ### Evaluate one

# %%
df = await read_thermo(uri, file_name="joint.thermo.out")

# %%
df

# %%
fig, ax = plot_df(df)
fig.tight_layout()
fig.show()

# %%
uri

# %% [markdown]
# ### Evaluate all

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'distance': 'readme.step_specific.frame_extraction.distance',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
}

# %%
parameters

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
res

# %%
res = await dl.aggregate(aggregation_pipeline)
res_df = pd.DataFrame(res)

# %%
res_df

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
    'readme.step': {'$regex': 'LAMMPSEquilibrationDPD'},
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
            "uri": {"$addToSet": "$uri"},
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
parameters.keys()

# %%
list_of_tuples = [(*[row[key] for key in parameters.keys()], row['uuid'][0]) 
    for _, row in res_df.iterrows()]

# %%
list_of_tuples

# %%
df_list = []
for columns, row in res_df.iterrows():
    uri = row['uri'][0]
    data_df = await read_thermo(uri, file_name="joint.thermo.out")
    row_df = pd.DataFrame(row).T.drop('step', axis=1)
    df = pd.merge(row_df, data_df.reset_index(), how='cross')
    df_list.append(df)

# %%
len(df_list)

# %%
res_df = pd.concat(df_list, ignore_index=True)

# %%
properties = data_df.columns

# %%
parameter_space_df = res_df[parameters.keys()].drop_duplicates()

# %%
parameter_space_df.iloc[0]

# %%
index = (res_df[parameters.keys()] == parameter_space_df.iloc[0]).T.all()

# %%
res_df[parameter_space_df.iloc[0]]

# %%
df['Step']

# %%
sub

# %%
parameter_set

# %%
df = res_df

nplots = len(properties)
ncols = 2
nrows = nplots // 2

fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols, 5*nrows))

for _, parameter_set in parameter_space_df.iterrows():
    index = (res_df[parameters.keys()] == parameter_set).T.all()
    sub_df = df[index]
    for i, col in enumerate(properties):
        ax = axs[i//ncols,i%ncols]
        ax.plot(sub_df['Step'], sub_df[col], label=str(parameter_set.to_dict()))
        ax.set_title(col)
ax.legend(bbox_to_anchor=(0.5, -0.5))


# %%

# %%
fig.show()

# %%
str(parameter_set.to_dict())

# %%
print(parameter_set.to_dict())

# %%
type(parameter_space_df.iloc[0])

# %%

# %%
res_df[parameters.keys()] == parameter_space_df.iloc[0

# %%
for row in parameter_space_df.itertuples():
    print(row)

# %%
parameter_set

# %%
pd.DataFrame(parameter_set)

# %%
fig, ax = plot_df(df)
fig.tight_layout()
fig.show()

# %%

# %%

# %%
df_list = []
for parameter_set in parameter_space_df.iterrows()
    for (x_shift, y_shift, uuid) in list_of_tuples:
        lookup_res = await dl.lookup(uuid)
        assert len(lookup_res) == 1

        uri = lookup_res[0]['uri']

        readme = await dl.readme(uri)
        velocity = readme['step_specific']['probe_normal_approach']['constant_indenter_velocity']
        if isinstance(velocity, str):
            velocity = float(velocity)

        steps_per_datapoint = readme['step_specific']['probe_normal_approach']['netcdf_frequency']


        df = await read_forces(uri, file_name='default.txt')
        df = df*force_conversion_factor

        df['x_shift'] = x_shift
        df['y_shift'] = y_shift
        df.reset_index(inplace=True)
        df['distance'] = initial_distance + velocity*steps_per_datapoint*fs_per_step*df['index']
        df.drop('index', axis=1, inplace=True)
        df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
legend_pattern = '''$(x,y) = ({x_shift:},{y_shift:}) \mathrm{{\AA}}$'''

# %%
offset_per_curve = 5

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[['y_shift','x_shift']].drop_duplicates().iterrows()):
    sub_df = res_df[ (res_df['x_shift'] == row['x_shift']) & (res_df['y_shift'] == row['y_shift'])]
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
fig, ax = plt.subplots(1,1)

for _, row in res_df[['y_shift','x_shift']].drop_duplicates().iterrows():
    sub_df = res_df[ (res_df['x_shift'] == row['x_shift']) & (res_df['y_shift'] == row['y_shift'])]
    
    sub_df.rolling(window=10,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
from cycler import cycler
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# %%
win = 10
offset_per_curve = 20

unique_parameter_sets = res_df[['y_shift','x_shift']].drop_duplicates()

n = len(unique_parameter_sets) # Number of colors
new_colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]

custom_cycler = cycler(color=new_colors)

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))


for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sub_df = res_df[ (res_df['x_shift'] == row['x_shift']) & (res_df['y_shift'] == row['y_shift'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

ax.legend()
ax.grid(which='major', axis='y')

# %% [markdown]
# ## Forces

# %% init_cell=true
project_id = "2021-12-30-sds-on-au-111-probe-on-substrate-lateral-sliding"

# %% [markdown]
# ### Overview on steps in project

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
# ### Overview on UUIDs in LaterSliding step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSProbeLateralSliding'},
}

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

# %% [markdown]
# ### Overview on UUIDs in ProbeAnalysis step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
}

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

# %% [markdown]
# ## Parametric Evaluation

# %% [markdown]
# ### Sliding velocity = 1 m /s

# %% init_cell=true
fs_per_step = 2  # fs
# initial_distance = 50  # Ang

# %% init_cell=true
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
}

# %% init_cell=true
columns_of_interest = ['direction','distance','x_shift','y_shift']

# %% [markdown]
# #### All

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
    'direction': 'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement',
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
res_df.sort_values(['direction','distance'])

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
    'readme.step': {'$regex': 'ProbeAnalysis'},
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
list_of_tuples = [tuple([row[c] for c in columns_of_interest] + [row['uuid'][0]]) 
    for _, row in res_df[columns_of_interest + ['uuid']].iterrows()]

# %%
list_of_tuples

# %%
df_list = []
for (direction, distance, x_shift, y_shift, uuid) in list_of_tuples:
    lookup_res = await dl.lookup(uuid)
    assert len(lookup_res) == 1
    
    uri = lookup_res[0]['uri']
    
    readme = await dl.readme(uri)
    velocity = readme['step_specific']['probe_lateral_sliding']['constant_indenter_velocity']
    if isinstance(velocity, str):
        velocity = float(velocity)
        
    steps_per_datapoint = readme['step_specific']['probe_lateral_sliding']['netcdf_frequency']
    
    
    dfx = await read_forces(uri, file_name='fx.txt')
    dfy = await read_forces(uri, file_name='fy.txt')
    dfz = await read_forces(uri, file_name='fz.txt')
    
    dfxy = pd.merge(dfx, dfy, how='inner', left_index=True, right_index=True, suffixes=('', '_y'))
    df = pd.merge(dfxy, dfz, how='inner', left_index=True, right_index=True, suffixes=('_x','_z'))
    # df = pd.concat([dfx, dfy, dfz], axis=1, join='outer', )
    df = df*force_conversion_factor
        
    df['x_shift'] = x_shift
    df['y_shift'] = y_shift
    df['distance'] = distance
    df['direction'] = direction
    df.reset_index(inplace=True)
    df['lateral_distance'] = -velocity*steps_per_datapoint*fs_per_step*df['index'] # plot in positive direction
    df.drop('index', axis=1, inplace=True)
    df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
legend_pattern = '''initial $(x,y,z) = ({x_shift:},{y_shift:},{distance:}) \mathrm{{\AA}}$, {direction:} direction'''

# %%
offset_per_curve = 4

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[columns_of_interest].drop_duplicates().iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_x']+i*offset_per_curve,
        label=legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')
ax.grid(which='major', axis='y')
ax.set_title(r'$F_x$')
ax.legend()

# %%
save_fig(fig, ax, f'${project_id}_all')

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[columns_of_interest].drop_duplicates().iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_y']+i*offset_per_curve,
        label=legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')
ax.grid(which='major', axis='y')
ax.set_title(r'$F_y$')
ax.legend()

# %%
save_fig(fig, ax, f'${project_id}all_fy')

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[columns_of_interest].drop_duplicates().iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_z']+i*offset_per_curve,
        label=legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')
ax.grid(which='major', axis='y')
ax.set_title(r'$F_z$')
ax.legend()

# %%
save_fig(fig, ax, f'${project_id}all_fz')

# %%
win = 10
offset_per_curve = 2

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

# n = len(unique_parameter_sets) # Number of colors
#  new_colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) 

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_x']+i*offset_per_curve,
        label=r'$F_x$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_y']+i*offset_per_curve,
        label=r'$F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_z']+i*offset_per_curve,
        label=r'$F_z$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
    
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

ax.legend()
ax.grid(which='major', axis='y')

# %%
save_fig(fig, ax, f'${project_id}all_summary')

# %% [markdown]
# ### y shift = (50.0, 0, -50), direction y (across hemicylinders)

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
    'readme.step_specific.merge.y_shift': {'$in': [50.0,0.0,-50]},
    'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement': 1,
}

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
    'direction': 'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement',
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
    'readme.step': {'$regex': 'ProbeAnalysis'},
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
list_of_tuples = [tuple([row[c] for c in columns_of_interest] + [row['uuid'][0]]) 
    for _, row in res_df[columns_of_interest + ['uuid']].iterrows()]

# %%
list_of_tuples

# %%
print(json.dumps(list_of_tuples, indent=4))

# %%
df_list = []
for (direction, distance, x_shift, y_shift, uuid) in list_of_tuples:
    lookup_res = await dl.lookup(uuid)
    assert len(lookup_res) == 1
    
    uri = lookup_res[0]['uri']
    
    readme = await dl.readme(uri)
    velocity = readme['step_specific']['probe_lateral_sliding']['constant_indenter_velocity']
    if isinstance(velocity, str):
        velocity = float(velocity)
        
    steps_per_datapoint = readme['step_specific']['probe_lateral_sliding']['netcdf_frequency']
    
    
    dfx = await read_forces(uri, file_name='fx.txt')
    dfy = await read_forces(uri, file_name='fy.txt')
    dfz = await read_forces(uri, file_name='fz.txt')
    
    dfxy = pd.merge(dfx, dfy, how='inner', left_index=True, right_index=True, suffixes=('', '_y'))
    df = pd.merge(dfxy, dfz, how='inner', left_index=True, right_index=True, suffixes=('_x','_z'))
    # df = pd.concat([dfx, dfy, dfz], axis=1, join='outer', )
    df = df*force_conversion_factor
        
    df['x_shift'] = x_shift
    df['y_shift'] = y_shift
    df['distance'] = distance
    df['direction'] = direction
    df.reset_index(inplace=True)
    df['lateral_distance'] = -velocity*steps_per_datapoint*fs_per_step*df['index'] # plot in positive direction
    df.drop('index', axis=1, inplace=True)
    df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
columns_of_interest

# %%
res_df.sort_values(columns_of_interest, inplace=True)

# %%
legend_pattern = '''initial $(x,y,z) = ({x_shift:},{y_shift:},{distance:}) \mathrm{{\AA}}$, {direction:} direction'''

# %%
np.count_nonzero((res_df.f_storeUnconstrainedForcesAve_x == 0))

# %%
np.count_nonzero((res_df.f_storeUnconstrainedForcesAve_x == 0) & (res_df.f_storeUnconstrainedForcesAve_z == 0))

# %%
# remove invalid zero forces entries
res_df = res_df[~((res_df.f_storeUnconstrainedForcesAve_x == 0) & (res_df.f_storeUnconstrainedForcesAve_z == 0))]

# %%
win = 10
offset_per_curve = 10

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

# n = len(unique_parameter_sets) # Number of colors
#  new_colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) 

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_x']+i*offset_per_curve,
        label=r'$F_x$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_y']+i*offset_per_curve,
        label=r'$F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_z']+i*offset_per_curve,
        label=r'$F_z$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
    
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
Legend.update_default_handler_map({Line2D: default_line_handler})
save_fig(fig, ax, f'${project_id}_dir_y_y_shift_50_0_-50_summary')
Legend.update_default_handler_map({Line2D: OpaqueSytelHandler()})

# %% [markdown]
#
# #### Force-force data point clouds

# %%
res_df

# %%
# win = 10
# offset_per_curve = 2

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

n = len(unique_parameter_sets) # Number of colors
colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
# colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = ['None']
markers=['x']
# linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) *cycler(marker=markers)

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
# ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

plots = []
labels = []
for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    current_label = r'$F_z : F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y'))
    current_plots = ax.plot(sub_df['f_storeUnconstrainedForcesAve_z'], sub_df['f_storeUnconstrainedForcesAve_y'], 
            alpha=0.1, 
            label=current_label)
    
    plots.extend(current_plots)
    labels.append(current_label)
    
    
ax.set_xlabel(r'lateral force $F_f\, (\mathrm{nN})$')
ax.set_ylabel(r'normal force $F_n\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
ax.legend(plots,labels)
fig

# %%
save_fig(fig, ax, f'${project_id}_force_force_point_cloud_dir_y_y_shift_50_0_-50')

# %% [markdown]
# #### Force-force data point clouds, without zero distance

# %%
# drop distance 0
res_df = res_df[res_df.distance != 0]

# %%
# win = 10
# offset_per_curve = 2

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

n = len(unique_parameter_sets) # Number of colors
colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
# colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = ['None']
markers=['x']
# linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) *cycler(marker=markers)

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
# ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

plots = []
labels = []
for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    current_label = r'$F_z : F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y'))
    current_plots = ax.plot(sub_df['f_storeUnconstrainedForcesAve_z'], sub_df['f_storeUnconstrainedForcesAve_y'], 
            alpha=0.1, 
            label=current_label)
    
    plots.extend(current_plots)
    labels.append(current_label)
    
    
ax.set_xlabel(r'lateral force $F_f\, (\mathrm{nN})$')
ax.set_ylabel(r'normal force $F_n\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
ax.legend(plots,labels)
fig

# %%
save_fig(fig, ax, f'${project_id}force_force_point_cloud_dir_y_y_shift_50_0_-50_no_zero_dist')

# %% [markdown]
# ### y shift = (50.0, 0, -50), direction x (along hemicylinders)

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
    'readme.step_specific.merge.y_shift': {'$in': [50.0,0.0,-50]},
    'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement': 0,
}

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
    'direction': 'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement',
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
    'readme.step': {'$regex': 'ProbeAnalysis'},
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
list_of_tuples = [tuple([row[c] for c in columns_of_interest] + [row['uuid'][0]]) 
    for _, row in res_df[columns_of_interest + ['uuid']].iterrows()]

# %%
list_of_tuples

# %%
print(json.dumps(list_of_tuples, indent=4))

# %%
df_list = []
for (direction, distance, x_shift, y_shift, uuid) in list_of_tuples:
    lookup_res = await dl.lookup(uuid)
    assert len(lookup_res) == 1
    
    uri = lookup_res[0]['uri']
    
    readme = await dl.readme(uri)
    velocity = readme['step_specific']['probe_lateral_sliding']['constant_indenter_velocity']
    if isinstance(velocity, str):
        velocity = float(velocity)
        
    steps_per_datapoint = readme['step_specific']['probe_lateral_sliding']['netcdf_frequency']
    
    
    dfx = await read_forces(uri, file_name='fx.txt')
    dfy = await read_forces(uri, file_name='fy.txt')
    dfz = await read_forces(uri, file_name='fz.txt')
    
    dfxy = pd.merge(dfx, dfy, how='inner', left_index=True, right_index=True, suffixes=('', '_y'))
    df = pd.merge(dfxy, dfz, how='inner', left_index=True, right_index=True, suffixes=('_x','_z'))
    # df = pd.concat([dfx, dfy, dfz], axis=1, join='outer', )
    df = df*force_conversion_factor
        
    df['x_shift'] = x_shift
    df['y_shift'] = y_shift
    df['distance'] = distance
    df['direction'] = direction
    df.reset_index(inplace=True)
    df['lateral_distance'] = -velocity*steps_per_datapoint*fs_per_step*df['index'] # plot in positive direction
    df.drop('index', axis=1, inplace=True)
    df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
res_df.sort_values(columns_of_interest, inplace=True)

# %%
np.count_nonzero((res_df.f_storeUnconstrainedForcesAve_x == 0))

# %%
np.count_nonzero((res_df.f_storeUnconstrainedForcesAve_x == 0) & (res_df.f_storeUnconstrainedForcesAve_z == 0))

# %%
# remove invalid zero forces entries
res_df = res_df[~((res_df.f_storeUnconstrainedForcesAve_x == 0) & (res_df.f_storeUnconstrainedForcesAve_z == 0))]

# %%
legend_pattern = '''initial $(x,y,z) = ({x_shift:},{y_shift:},{distance:}) \mathrm{{\AA}}$, {direction:} direction'''

# %%
win = 10
offset_per_curve = 10

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

# n = len(unique_parameter_sets) # Number of colors
#  new_colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) 

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_x']+i*offset_per_curve,
        label=r'$F_x$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_y']+i*offset_per_curve,
        label=r'$F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_z']+i*offset_per_curve,
        label=r'$F_z$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
    
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
Legend.update_default_handler_map({Line2D: default_line_handler})
save_fig(fig, ax, f'${project_id}_dir_x_y_shift_50_0_-50_summary')
Legend.update_default_handler_map({Line2D: OpaqueSytelHandler()})

# %% [markdown]
#
# #### Force-force data point clouds

# %%
res_df

# %%
unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

n = len(unique_parameter_sets) # Number of colors
colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
linestyles = ['None']
markers=['x']
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) *cycler(marker=markers)

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
# ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

plots = []
labels = []
for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    current_label = r'$F_z : F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y'))
    current_plots = ax.plot(sub_df['f_storeUnconstrainedForcesAve_z'], sub_df['f_storeUnconstrainedForcesAve_x'], 
            alpha=0.1, 
            label=current_label)
    
    plots.extend(current_plots)
    labels.append(current_label)
    
    
ax.set_xlabel(r'lateral force $F_f\, (\mathrm{nN})$')
ax.set_ylabel(r'normal force $F_n\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
save_fig(fig, ax, f'${project_id}_force_force_point_cloud_dir_x_y_shift_50_0_-50')

# %% [markdown]
# #### Force-force data point clouds, without zero distance

# %%
# drop distance 0
res_df = res_df[res_df.distance != 0]

# %%
unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

n = len(unique_parameter_sets) # Number of colors
colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
linestyles = ['None']
markers=['x']
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) *cycler(marker=markers)

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
# ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

plots = []
labels = []
for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    current_label = r'$F_z : F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y'))
    current_plots = ax.plot(sub_df['f_storeUnconstrainedForcesAve_z'], sub_df['f_storeUnconstrainedForcesAve_x'], 
            alpha=0.1, 
            label=current_label)
    
    plots.extend(current_plots)
    labels.append(current_label)
    
    
ax.set_xlabel(r'lateral force $F_f\, (\mathrm{nN})$')
ax.set_ylabel(r'normal force $F_n\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
Legend.update_default_handler_map({Line2D: OpaqueSytelHandler()})
ax.legend(plots,labels)
fig

# %%
save_fig(fig, ax, f'${project_id}_force_force_point_cloud_dir_x_y_shift_50_0_-50_no_zero_dist')

# %% [markdown]
# ## 2022-01-21-sds-on-au-111-probe-on-substrate-lateral-sliding (between hemicylinders)

# %% init_cell=true
project_id = "2022-01-21-sds-on-au-111-probe-on-substrate-lateral-sliding"

# %% [markdown]
# ### Overview on steps in project

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
# ### Overview on UUIDs in LaterSliding step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSProbeLateralSliding'},
}

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

# %% [markdown]
# ### Overview on UUIDs in ProbeAnalysis step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
}

# %%
await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %%
uri = await get_uri_by_query(query)

# %%
uri

# %%
item_dict

# %% [markdown]
# ## Parametric Evaluation

# %% [markdown]
# ### Sliding velocity = 1 m /s

# %% init_cell=true
fs_per_step = 2  # fs
# initial_distance = 50  # Ang

# %% init_cell=true
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
}

# %% init_cell=true
columns_of_interest = ['direction','distance','x_shift','y_shift']

# %% [markdown]
# #### All

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
    'direction': 'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement',
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
res_df.sort_values(['direction','distance'])

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
    'readme.step': {'$regex': 'ProbeAnalysis'},
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
list_of_tuples = [tuple([row[c] for c in columns_of_interest] + [row['uuid'][0]]) 
    for _, row in res_df[columns_of_interest + ['uuid']].iterrows()]

# %%
list_of_tuples

# %%
df_list = []
for (direction, distance, x_shift, y_shift, uuid) in list_of_tuples:
    lookup_res = await dl.lookup(uuid)
    assert len(lookup_res) == 1
    
    uri = lookup_res[0]['uri']
    
    readme = await dl.readme(uri)
    velocity = readme['step_specific']['probe_lateral_sliding']['constant_indenter_velocity']
    if isinstance(velocity, str):
        velocity = float(velocity)
        
    steps_per_datapoint = readme['step_specific']['probe_lateral_sliding']['netcdf_frequency']
    
    
    dfx = await read_forces(uri, file_name='fx.txt')
    dfy = await read_forces(uri, file_name='fy.txt')
    dfz = await read_forces(uri, file_name='fz.txt')
    
    dfxy = pd.merge(dfx, dfy, how='inner', left_index=True, right_index=True, suffixes=('', '_y'))
    df = pd.merge(dfxy, dfz, how='inner', left_index=True, right_index=True, suffixes=('_x','_z'))
    # df = pd.concat([dfx, dfy, dfz], axis=1, join='outer', )
    df = df*force_conversion_factor
        
    df['x_shift'] = x_shift
    df['y_shift'] = y_shift
    df['distance'] = distance
    df['direction'] = direction
    df.reset_index(inplace=True)
    df['lateral_distance'] = -velocity*steps_per_datapoint*fs_per_step*df['index'] # plot in positive direction
    df.drop('index', axis=1, inplace=True)
    df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
legend_pattern = '''initial $(x,y,z) = ({x_shift:},{y_shift:},{distance:}) \mathrm{{\AA}}$, {direction:} direction'''

# %%
offset_per_curve = 4

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[columns_of_interest].drop_duplicates().iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_x']+i*offset_per_curve,
        label=legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')
ax.grid(which='major', axis='y')
ax.set_title(r'$F_x$')
ax.legend()

# %%
save_fig(fig, ax, 'all')

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[columns_of_interest].drop_duplicates().iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_y']+i*offset_per_curve,
        label=legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')
ax.grid(which='major', axis='y')
ax.set_title(r'$F_y$')
ax.legend()

# %%
save_fig(fig, ax, 'all_fy')

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[columns_of_interest].drop_duplicates().iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_z']+i*offset_per_curve,
        label=legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')
ax.grid(which='major', axis='y')
ax.set_title(r'$F_z$')
ax.legend()

# %%
save_fig(fig, ax, 'all_fz')

# %%
win = 10
offset_per_curve = 2

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

# n = len(unique_parameter_sets) # Number of colors
#  new_colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) 

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_x']+i*offset_per_curve,
        label=r'$F_x$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_y']+i*offset_per_curve,
        label=r'$F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_z']+i*offset_per_curve,
        label=r'$F_z$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
    
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

ax.legend()
ax.grid(which='major', axis='y')

# %%
save_fig(fig, ax, 'all_summary')

# %% [markdown]
# ### y shift = (50.0, 0, -50), direction y (across hemicylinders)

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
    'readme.step_specific.merge.y_shift': {'$in': [-25.0,25.0]},
    'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement': 1,
}

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
    'direction': 'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement',
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
    'readme.step': {'$regex': 'ProbeAnalysis'},
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
list_of_tuples = [tuple([row[c] for c in columns_of_interest] + [row['uuid'][0]]) 
    for _, row in res_df[columns_of_interest + ['uuid']].iterrows()]

# %%
list_of_tuples

# %%
print(json.dumps(list_of_tuples, indent=4))

# %%
df_list = []
for (direction, distance, x_shift, y_shift, uuid) in list_of_tuples:
    lookup_res = await dl.lookup(uuid)
    assert len(lookup_res) == 1
    
    uri = lookup_res[0]['uri']
    
    readme = await dl.readme(uri)
    velocity = readme['step_specific']['probe_lateral_sliding']['constant_indenter_velocity']
    if isinstance(velocity, str):
        velocity = float(velocity)
        
    steps_per_datapoint = readme['step_specific']['probe_lateral_sliding']['netcdf_frequency']
    
    
    dfx = await read_forces(uri, file_name='fx.txt')
    dfy = await read_forces(uri, file_name='fy.txt')
    dfz = await read_forces(uri, file_name='fz.txt')
    
    dfxy = pd.merge(dfx, dfy, how='inner', left_index=True, right_index=True, suffixes=('', '_y'))
    df = pd.merge(dfxy, dfz, how='inner', left_index=True, right_index=True, suffixes=('_x','_z'))
    # df = pd.concat([dfx, dfy, dfz], axis=1, join='outer', )
    df = df*force_conversion_factor
        
    df['x_shift'] = x_shift
    df['y_shift'] = y_shift
    df['distance'] = distance
    df['direction'] = direction
    df.reset_index(inplace=True)
    df['lateral_distance'] = -velocity*steps_per_datapoint*fs_per_step*df['index'] # plot in positive direction
    df.drop('index', axis=1, inplace=True)
    df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
columns_of_interest

# %%
res_df.sort_values(columns_of_interest, inplace=True)

# %%
legend_pattern = '''initial $(x,y,z) = ({x_shift:},{y_shift:},{distance:}) \mathrm{{\AA}}$, {direction:} direction'''

# %%
np.count_nonzero((res_df.f_storeUnconstrainedForcesAve_x == 0))

# %%
np.count_nonzero((res_df.f_storeUnconstrainedForcesAve_x == 0) & (res_df.f_storeUnconstrainedForcesAve_z == 0))

# %%
# remove invalid zero forces entries
res_df = res_df[~((res_df.f_storeUnconstrainedForcesAve_x == 0) & (res_df.f_storeUnconstrainedForcesAve_z == 0))]

# %%
win = 10
offset_per_curve = 10

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

# n = len(unique_parameter_sets) # Number of colors
#  new_colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) 

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_x']+i*offset_per_curve,
        label=r'$F_x$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_y']+i*offset_per_curve,
        label=r'$F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_z']+i*offset_per_curve,
        label=r'$F_z$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
    
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
Legend.update_default_handler_map({Line2D: default_line_handler})
save_fig(fig, ax, 'dir_y_y_shift_-25_25_summary')
Legend.update_default_handler_map({Line2D: OpaqueSytelHandler()})

# %% [markdown]
#
# #### Force-force data point clouds

# %%
res_df

# %%
# win = 10
# offset_per_curve = 2

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

n = len(unique_parameter_sets) # Number of colors
colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
# colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = ['None']
markers=['x']
# linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) *cycler(marker=markers)

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
# ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

plots = []
labels = []
for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    current_label = r'$F_z : F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y'))
    current_plots = ax.plot(sub_df['f_storeUnconstrainedForcesAve_z'], sub_df['f_storeUnconstrainedForcesAve_y'], 
            alpha=0.1, 
            label=current_label)
    
    plots.extend(current_plots)
    labels.append(current_label)
    
    
ax.set_xlabel(r'lateral force $F_f\, (\mathrm{nN})$')
ax.set_ylabel(r'normal force $F_n\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
ax.legend(plots,labels)
fig

# %%
save_fig(fig, ax, 'force_force_point_cloud_dir_y_y_shift_-25_25')

# %% [markdown]
# #### Force-force data point clouds, without zero distance

# %%
# drop distance 0
res_df = res_df[res_df.distance != 0]

# %%
# win = 10
# offset_per_curve = 2

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

n = len(unique_parameter_sets) # Number of colors
colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
# colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = ['None']
markers=['x']
# linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) *cycler(marker=markers)

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
# ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

plots = []
labels = []
for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    current_label = r'$F_z : F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y'))
    current_plots = ax.plot(sub_df['f_storeUnconstrainedForcesAve_z'], sub_df['f_storeUnconstrainedForcesAve_y'], 
            alpha=0.1, 
            label=current_label)
    
    plots.extend(current_plots)
    labels.append(current_label)
    
    
ax.set_xlabel(r'lateral force $F_f\, (\mathrm{nN})$')
ax.set_ylabel(r'normal force $F_n\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
ax.legend(plots,labels)
fig

# %%
save_fig(fig, ax, 'force_force_point_cloud_dir_y_y_shift_-25_25_no_zero_dist')

# %% [markdown]
# ### y shift = (25.0, -25.0), direction x (along hemicylinders)

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
    'readme.step_specific.merge.y_shift': {'$in': [25.0, -25.0]},
    'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement': 0,
}

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    #'concentration': 'readme.system.surfactant.surface_concentration',
    #'shape': 'readme.system.surfactant.aggregates.shape',
    'x_shift': 'readme.step_specific.merge.x_shift',
    'y_shift': 'readme.step_specific.merge.y_shift',
    'distance': 'readme.step_specific.frame_extraction.distance',
    'direction': 'readme.step_specific.probe_lateral_sliding.direction_of_linear_movement',
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
    'readme.step': {'$regex': 'ProbeAnalysis'},
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
list_of_tuples = [tuple([row[c] for c in columns_of_interest] + [row['uuid'][0]]) 
    for _, row in res_df[columns_of_interest + ['uuid']].iterrows()]

# %%
list_of_tuples

# %%
print(json.dumps(list_of_tuples, indent=4))

# %%
df_list = []
for (direction, distance, x_shift, y_shift, uuid) in list_of_tuples:
    lookup_res = await dl.lookup(uuid)
    assert len(lookup_res) == 1
    
    uri = lookup_res[0]['uri']
    
    readme = await dl.readme(uri)
    velocity = readme['step_specific']['probe_lateral_sliding']['constant_indenter_velocity']
    if isinstance(velocity, str):
        velocity = float(velocity)
        
    steps_per_datapoint = readme['step_specific']['probe_lateral_sliding']['netcdf_frequency']
    
    
    dfx = await read_forces(uri, file_name='fx.txt')
    dfy = await read_forces(uri, file_name='fy.txt')
    dfz = await read_forces(uri, file_name='fz.txt')
    
    dfxy = pd.merge(dfx, dfy, how='inner', left_index=True, right_index=True, suffixes=('', '_y'))
    df = pd.merge(dfxy, dfz, how='inner', left_index=True, right_index=True, suffixes=('_x','_z'))
    # df = pd.concat([dfx, dfy, dfz], axis=1, join='outer', )
    df = df*force_conversion_factor
        
    df['x_shift'] = x_shift
    df['y_shift'] = y_shift
    df['distance'] = distance
    df['direction'] = direction
    df.reset_index(inplace=True)
    df['lateral_distance'] = -velocity*steps_per_datapoint*fs_per_step*df['index'] # plot in positive direction
    df.drop('index', axis=1, inplace=True)
    df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
res_df.sort_values(columns_of_interest, inplace=True)

# %%
np.count_nonzero((res_df.f_storeUnconstrainedForcesAve_x == 0))

# %%
np.count_nonzero((res_df.f_storeUnconstrainedForcesAve_x == 0) & (res_df.f_storeUnconstrainedForcesAve_z == 0))

# %%
# remove invalid zero forces entries
res_df = res_df[~((res_df.f_storeUnconstrainedForcesAve_x == 0) & (res_df.f_storeUnconstrainedForcesAve_z == 0))]

# %%
legend_pattern = '''initial $(x,y,z) = ({x_shift:},{y_shift:},{distance:}) \mathrm{{\AA}}$, {direction:} direction'''

# %%
win = 10
offset_per_curve = 10

unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

# n = len(unique_parameter_sets) # Number of colors
#  new_colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
colors = ['red', 'green', 'blue'] # for x,y,z
linestyles = [s[1] for s in (linestyle_str + linestyle_tuple)]
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) 

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_x']+i*offset_per_curve,
        label=r'$F_x$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_y']+i*offset_per_curve,
        label=r'$F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    ax.plot(sub_df['lateral_distance'], sub_df['f_storeUnconstrainedForcesAve_z']+i*offset_per_curve,
        label=r'$F_z$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y')))
    
    
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
Legend.update_default_handler_map({Line2D: default_line_handler})
save_fig(fig, ax, 'dir_x_y_shift_-25_25_summary')
Legend.update_default_handler_map({Line2D: OpaqueSytelHandler()})

# %% [markdown]
#
# #### Force-force data point clouds

# %%
res_df

# %%
unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

n = len(unique_parameter_sets) # Number of colors
colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
linestyles = ['None']
markers=['x']
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) *cycler(marker=markers)

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
# ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

plots = []
labels = []
for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    current_label = r'$F_z : F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y'))
    current_plots = ax.plot(sub_df['f_storeUnconstrainedForcesAve_z'], sub_df['f_storeUnconstrainedForcesAve_x'], 
            alpha=0.1, 
            label=current_label)
    
    plots.extend(current_plots)
    labels.append(current_label)
    
    
ax.set_xlabel(r'lateral force $F_f\, (\mathrm{nN})$')
ax.set_ylabel(r'normal force $F_n\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
save_fig(fig, ax, 'force_force_point_cloud_dir_x_y_shift_25_-25')

# %% [markdown]
# #### Force-force data point clouds, without zero distance

# %%
# drop distance 0
res_df = res_df[res_df.distance != 0]

# %%
unique_parameter_sets = res_df[columns_of_interest].drop_duplicates()

n = len(unique_parameter_sets) # Number of colors
colors = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
linestyles = ['None']
markers=['x']
custom_cycler = cycler(linestyle=linestyles)*cycler(color=colors) *cycler(marker=markers)

fig, ax = plt.subplots(1,1)
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
# ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

plots = []
labels = []
for i, (_, row) in enumerate(unique_parameter_sets.iterrows()):
    sel = np.all([res_df[c] == row[c] for c in columns_of_interest], axis=0)
    sub_df = res_df[sel]
    
    current_label = r'$F_z : F_y$, ' + legend_pattern.format(
            x_shift=row['x_shift'],
            y_shift=row['y_shift'],
            distance=row['distance'],
            direction=('x' if row['direction'] == 0 else 'y'))
    current_plots = ax.plot(sub_df['f_storeUnconstrainedForcesAve_z'], sub_df['f_storeUnconstrainedForcesAve_x'], 
            alpha=0.1, 
            label=current_label)
    
    plots.extend(current_plots)
    labels.append(current_label)
    
    
ax.set_xlabel(r'lateral force $F_f\, (\mathrm{nN})$')
ax.set_ylabel(r'normal force $F_n\, (\mathrm{nN})$')

ax.grid(which='major', axis='y')

# %%
Legend.update_default_handler_map({Line2D: OpaqueSytelHandler()})
ax.legend(plots,labels)
fig

# %%
save_fig(fig, ax, 'force_force_point_cloud_dir_x_y_shift_-25_25_no_zero_dist')

# %% [markdown]
# # Other

# %% [markdown]
#
# ### Initial configurations

# %%
project_id = "2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"
# project_id = "2021-12-30-sds-on-au-111-probe-on-substrate-lateral-sliding"

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSEquilibrationDPD'},
}

# %%
columns_of_interest = ['x_shift', 'y_shift', 'distance']

# %% [markdown]
# #### All

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
    'readme.step': {'$regex': 'LAMMPSEquilibrationDPD'},
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
list_of_tuples = [tuple([row[c] for c in columns_of_interest] + [row['uuid'][0]]) 
    for _, row in res_df[columns_of_interest + ['uuid']].iterrows()]

# %%
columns_of_interest

# %%
list_of_tuples

# %%
dict_of_tuples = {(x,y,z): uuid for x,y,z,uuid in list_of_tuples}

# %%
dict_of_tuples

# %%
offset_tuple = (0,-25,20)

# %%
uuid = dict_of_tuples[offset_tuple]

# %%
uuid

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
try: # https://www.ovito.org/forum/topic/visualization-elements-persist/
    del ovito.scene.pipelines[:]
except:
    pass
# Create a virtual viewport:
vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(0, 0, -1))
vp.zoom_all(size=(800, 600))

# Create the visual viewport widget:
widget = vp.create_jupyter_widget(width=800, height=600, device_pixel_ratio=2.0)

# %%
display(widget)

# %%
# Import a simulation model and set up a data pipeline:
pipeline = import_file(fpath)
pipeline.add_to_scene()
pipeline.modifiers.clear()

# remove water
pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {8,9}))
pipeline.modifiers.append(DeleteSelectedModifier())

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
vp.zoom_all()

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
# ## Extracted Frames

# %%
