# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
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

# %%
project_id = '2021-01-28-sds-on-au-111-probe-and-substrate-approach'

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
# ## Normal approach

# %% [markdown]
# ### Overview on UUIDs in step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'LAMMPSProbeNormalApproach'},
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
df = await read_thermo(uri, file_name="joint.thermo.out")

# %%
df

# %%
fig, ax = plot_df(df)
fig.show()

# %%
uri

# %% [markdown]
# ## Normal approach forces

# %% [markdown]
# ### Overview on UUIDs in step

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
}

# %%
await get_df_by_query(query)

# %%
await get_df_by_filtered_query(query)

# %% [markdown]
# ### List items

# %%
item_dict = await get_item_dict(uri)

# %%
item_dict

# %% [markdown]
# ## Parametric Evaluation

# %% [markdown]
# ### Approach velocity = 10 m /s

# %%
fs_per_step = 2  # fs
initial_distance = 50  # Ang

# %%
project_id = '2021-01-28-sds-on-au-111-probe-and-substrate-approach'

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
}

# %% [markdown]
# #### All

# %%
parameters = { 
    'concentration': 'readme.system.concentration',
    'shape': 'readme.system.shape',
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

# filter out unwanted values
immutable_distinct_parameter_values = {k: [e for e in p if (
        isinstance(e, np.float64) and not np.isnan(e)) or (
        not isinstance(e, np.float64) and e is not None)] 
    for k, p in distinct_parameter_values.items()}

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}
distinct_parameter_values

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
list_of_tuples = [(row['concentration'], row['shape'], row['uuid'][0]) 
    for _, row in res_df[['concentration','shape','uuid']].iterrows()]

# %%
list_of_tuples

# %%
await dl.lookup('5b597394-314f-4dcc-a391-969f8bb182ba')

# %%
fs_per_step = 2

# %%
df_list = []
for (concentration, shape, uuid) in list_of_tuples:
    lookup_res = await dl.lookup(uuid)
    assert len(lookup_res) == 1
    
    uri = lookup_res[0]['uri']
    
    readme = await dl.readme(uri)
    velocity = readme['step_specific']['probe_normal_approach']['constant_indenter_velocity']
    steps_per_datapoint = readme['step_specific']['probe_normal_approach']['netcdf_frequency']
    
    
    df = await read_forces(uri, file_name='default.txt')
    df = df*force_conversion_factor
        
    df['concentration'] = concentration
    df['shape'] = shape
    df.reset_index(inplace=True)
    df['distance'] = initial_distance + velocity*steps_per_datapoint*fs_per_step*df['index']
    df.drop('index', axis=1, inplace=True)
    df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
legend_pattern = '''{shape:}, concentration n = {concentration:.2f} $\mathrm{{nm}}^{{-2}}$.'''

# %%
offset_per_curve = 5

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[['shape','concentration']].drop_duplicates().iterrows()):
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
fig, ax = plt.subplots(1,1)

for _, row in res_df[['shape','concentration']].drop_duplicates().iterrows():
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    sub_df.rolling(window=10,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
from cycler import cycler
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# %%
win = 10
offset_per_curve = 20

unique_parameter_sets = res_df[['shape','concentration']].drop_duplicates()

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
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='y')

# %% [markdown]
# #### Monolayers

# %%
parameter_sets_df = res_df[['shape','concentration']].drop_duplicates()

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets_df['shape'] == 'monolayer'].iterrows():
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    sub_df.rolling(window=1,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
win = 10
offset_per_curve = 20
concetration_step = 0.25 / concentration_conversion_factor

previous_concentration = None

parameter_sets_df = res_df[['shape','concentration']].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    parameter_sets_df['shape'] == 'monolayer'].sort_values(by='concentration', ascending=True)


n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('summer')(1-(i+2)/n) for i in range(n)]
# new_colors.reverse()

custom_cycler = cycler(color=new_colors)

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

offset = 0
for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    if not previous_concentration:
        previous_concentration = row['concentration']
        
    offset += (row['concentration'] - previous_concentration)/concetration_step * offset_per_curve
    previous_concentration = row['concentration']

    
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+offset,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('monolayer_v_10_m_per_s.svg', dpi=dpi)

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), framealpha=1, labelcolor='linecolor')

plt.savefig('monolayer_v_10_m_per_s_legended.svg', dpi=dpi)

# %% [markdown]
# #### Hemicylinders

# %%
win = 10
offset_per_curve = 20
concetration_step = 0.25 / concentration_conversion_factor
offset = 0
previous_concentration = None

parameter_sets_df = res_df[['shape','concentration']].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    parameter_sets_df['shape'] == 'hemicylinders'].sort_values(by='concentration', ascending=True)


n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('summer')(1-(i+2)/n) for i in range(n)]

custom_cycler = cycler(color=new_colors)

# custom_cycler = cycler(color=new_colors) + cycler(linestyle=custom_linestyles[:n])

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    if not previous_concentration:
        previous_concentration = row['concentration']
        
    offset += (row['concentration'] - previous_concentration)/concetration_step * offset_per_curve
    previous_concentration = row['concentration']

    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+offset,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('hemicylinders_v_10_m_per_s.svg', dpi=dpi)

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), framealpha=1, labelcolor='linecolor')

plt.savefig('hemicylinders_v_10_m_per_s_legended.svg', dpi=dpi)

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets_df['shape'] == 'hemicylinders'].iterrows():
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    sub_df.rolling(window=50,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %% [markdown]
# #### c = 0.5 nm^-2 monolayer vs hemicylinders

# %%
selected_cocentration = 0.5 / concentration_conversion_factor

# %%
win = 10
offset_per_curve = 0
concetration_step = 0.25 / concentration_conversion_factor
offset = 0
previous_concentration = None

parameter_sets_df = res_df[['shape','concentration']].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    parameter_sets_df['concentration'] == selected_cocentration]

n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('hot')(1. * i/n) for i in range(n)]

custom_cycler = cycler(color=new_colors)

# custom_cycler = cycler(color=new_colors) + cycler(linestyle=custom_linestyles[:n])

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
if offset_per_curve != 0:
    ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    if not previous_concentration:
        previous_concentration = row['concentration']
        
    offset += (row['concentration'] - previous_concentration)/concetration_step * offset_per_curve
    previous_concentration = row['concentration']

    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+offset,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('v_10_m_per_s_c_0.5_per_nm_sq.svg', dpi=dpi)
ax.legend(framealpha=1, labelcolor='linecolor')
plt.savefig('v_10_m_per_s_c_0.5_per_nm_sq_legended.svg', dpi=dpi)

# %% [markdown]
# #### c = 1 nm^-2 monolayer vs hemicylinders

# %%
selected_cocentration = 1 / concentration_conversion_factor

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets_df['concentration'] == selected_cocentration].iterrows():
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    sub_df.rolling(window=10,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
win = 10
offset_per_curve = 0
concetration_step = 0.25 / concentration_conversion_factor
offset = 0
previous_concentration = None

parameter_sets_df = res_df[['shape','concentration']].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    parameter_sets_df['concentration'] == selected_cocentration]

n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('hot')(1. * i/n) for i in range(n)]

custom_cycler = cycler(color=new_colors)

# custom_cycler = cycler(color=new_colors) + cycler(linestyle=custom_linestyles[:n])

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
if offset_per_curve != 0:
    ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    if not previous_concentration:
        previous_concentration = row['concentration']
        
    offset += (row['concentration'] - previous_concentration)/concetration_step * offset_per_curve
    previous_concentration = row['concentration']

    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+offset,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('v_10_m_per_s_c_1_per_nm_sq.svg', dpi=dpi)
ax.legend(framealpha=1, labelcolor='linecolor')
plt.savefig('v_10_m_per_s_c_1_per_nm_sq_legended.svg', dpi=dpi)

# %% [markdown]
# #### c = 1.25 nm^-2 monolayer vs hemicylinders

# %%
selected_cocentration = 1.25 / concentration_conversion_factor

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets_df['concentration'] == selected_cocentration].iterrows():
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    sub_df.rolling(window=10,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
win = 10
offset_per_curve = 0
concetration_step = 0.25 / concentration_conversion_factor
offset = 0
previous_concentration = None

parameter_sets_df = res_df[['shape','concentration']].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    parameter_sets_df['concentration'] == selected_cocentration]

n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('hot')(1. * i/n) for i in range(n)]

custom_cycler = cycler(color=new_colors)

# custom_cycler = cycler(color=new_colors) + cycler(linestyle=custom_linestyles[:n])

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
if offset_per_curve != 0:
    ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    if not previous_concentration:
        previous_concentration = row['concentration']
        
    offset += (row['concentration'] - previous_concentration)/concetration_step * offset_per_curve
    previous_concentration = row['concentration']

    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+offset,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('v_10_m_per_s_c_1.25_per_nm_sq.svg', dpi=dpi)
ax.legend(framealpha=1, labelcolor='linecolor')
plt.savefig('v_10_m_per_s_c_1.25_per_nm_sq_legended.svg', dpi=dpi)

# %% [markdown]
# ####  c = 0.5, 1.0, 1.25 nm^-2

# %%
selected_velocity = 10 * velocity_conversion_factor

# %%
selected_concentration_range = np.array([0.5, 1.25]) / concentration_conversion_factor

# %%
parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()

# %%
win = 10
offset_per_curve = 20
concentration_step = 0.25 / concentration_conversion_factor
offset = 0
previous_concentration = None

parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    (parameter_sets_df['concentration'] >= selected_cocentration_range[0]) &
    (parameter_sets_df['concentration'] <= selected_cocentration_range[1])
].sort_values(by=['concentration','shape'], ascending=[True, False])

n1 = len(restricted_parameter_sets_df['concentration'].drop_duplicates())
n2 = len(restricted_parameter_sets_df['shape'].drop_duplicates())

# select only datasets where both aggregate shapes are available

restricted_parameter_sets_df = restricted_parameter_sets_df.groupby(
    'concentration').filter(lambda x: len(x) == n2)


# n = len(restricted_parameter_sets_df) # Number of colors
ranges = [0.5 + 0.9*(i+1)/n1*(j/(n2-1)-0.5) for i in range(n1) for j in range(n2)]


new_colors = [plt.get_cmap('PuOr')(r) for r in ranges]

custom_cycler = cycler(color=new_colors) # + cycler(linestyle=custom_linestyles[:n2])*n1

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
if offset_per_curve != 0:
    ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))


for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    if not previous_concentration:
        previous_concentration = row['concentration']
        
    offset += (row['concentration'] - previous_concentration)/concentration_step * offset_per_curve
    previous_concentration = row['concentration']

    
    sub_df = res_df[
        (res_df['concentration'] == row['concentration']) & 
        (res_df['shape'] == row['shape'])
    ].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+offset,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
            
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('v_10_m_per_s_c_0.5_to_1.25_per_sq_nm.svg', dpi=dpi)

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), framealpha=1, labelcolor='linecolor')

plt.savefig('v_10_m_per_s_c_0.5_to_1.25_per_sq_nm_legended.svg', dpi=dpi)

# %% [markdown]
# #### c = 2.75 nm^-2 monolayer

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets['concentration'] == 0.0275].iterrows():
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    sub_df.rolling(window=10,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
win = 10
offset_per_curve = 0
concetration_step = 0.25 / concentration_conversion_factor
offset = 0
previous_concentration = None

parameter_sets_df = res_df[['shape','concentration']].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    (parameter_sets_df['shape'] == 'monolayer') &
    (parameter_sets_df['concentration'] == selected_cocentration)]

n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('hot')(1. * i/n) for i in range(n)]

custom_cycler = cycler(color=new_colors)

# custom_cycler = cycler(color=new_colors) + cycler(linestyle=custom_linestyles[:n])

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
if offset_per_curve != 0:
    ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))

for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    if not previous_concentration:
        previous_concentration = row['concentration']
        
    offset += (row['concentration'] - previous_concentration)/concetration_step * offset_per_curve
    previous_concentration = row['concentration']

    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+offset,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('monolayer_v_10_m_per_s_c_2.75_per_nm_sq.svg', dpi=dpi)
ax.legend(framealpha=1, labelcolor='linecolor')
plt.savefig('monolayer_v_10_m_per_s_c_2.75_per_nm_sq_legended.svg', dpi=dpi)

# %% [markdown]
# ### Approach velocity = 1 m /s

# %%
fs_per_step = 2  # fs
initial_distance = 50  # Ang

# %%
project_id = '2021-02-05-sds-on-au-111-probe-and-substrate-approach'

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
}

# %% [markdown]
# #### All

# %%
parameters = { 
    'concentration': 'readme.system.concentration',
    'shape': 'readme.system.shape',
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

# filter out unwanted values
immutable_distinct_parameter_values = {k: [e for e in p if (
        isinstance(e, np.float64) and not np.isnan(e)) or (
        not isinstance(e, np.float64) and e is not None)] 
    for k, p in distinct_parameter_values.items()}

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}
distinct_parameter_values

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
list_of_tuples = [(row['concentration'], row['shape'], row['uuid'][0]) 
    for _, row in res_df[['concentration','shape','uuid']].iterrows()]

# %%
list_of_tuples

# %%
df_list = []
for (concentration, shape, uuid) in list_of_tuples:
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
        
    df['concentration'] = concentration
    df['shape'] = shape
    df.reset_index(inplace=True)
    df['distance'] = initial_distance + velocity*steps_per_datapoint*fs_per_step*df['index']
    df.drop('index', axis=1, inplace=True)
    df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
legend_pattern = '''{shape:}, concentration n = {concentration:} $\mathrm{{nm}}^{{-2}}$.'''

# %%
offset_per_curve = 5

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[['shape','concentration']].drop_duplicates().iterrows()):
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
fig, ax = plt.subplots(1,1)

for _, row in res_df[['shape','concentration']].drop_duplicates().iterrows():
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    sub_df.rolling(window=10,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
from cycler import cycler
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# %%
win = 10
offset_per_curve = 20

unique_parameter_sets = res_df[['shape','concentration']].drop_duplicates()

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
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='y')

# %% [markdown]
# #### Monolayers

# %%
parameter_sets_df = res_df[['shape','concentration']].drop_duplicates()

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets_df['shape'] == 'monolayer'].iterrows():
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    sub_df.rolling(window=1,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
win = 10
offset_per_curve = 20
concetration_step = 0.25 / concentration_conversion_factor
offset = 0
previous_concentration = None


parameter_sets_df = res_df[['shape','concentration']].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    parameter_sets_df['shape'] == 'monolayer'].sort_values(by='concentration', ascending=True)


n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('summer')(1-(i+1)/n) for i in range(n)]

custom_cycler = cycler(color=new_colors)

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))


for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    if not previous_concentration:
        previous_concentration = row['concentration']
        
    offset += (row['concentration'] - previous_concentration)/concetration_step * offset_per_curve
    previous_concentration = row['concentration']
    
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+offset,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('monolayer_v_1_m_per_s.svg', dpi=dpi)

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), framealpha=1, labelcolor='linecolor')

plt.savefig('monolayer_v_1_m_per_s_legended.svg', dpi=dpi)


# %%
restricted_parameter_sets_df

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets_df['shape'] == 'monolayer'].iterrows():
    sub_df = res_df[ (res_df['concentration'] == row['concentration']) & (res_df['shape'] == row['shape'])]
    
    sub_df.rolling(window=50,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %% [markdown]
# ### Monolayers

# %%
fs_per_step = 2  # fs
initial_distance = 50  # Ang

# %%
project_id = {
    '$in': [
        '2021-01-28-sds-on-au-111-probe-and-substrate-approach',  # 10 m / s
        '2021-02-05-sds-on-au-111-probe-and-substrate-approach',  # 1 m / s
    ]
}

# %%
query = {
    'readme.project': project_id,
    'readme.step': {'$regex': 'ProbeAnalysis'},
    'readme.system.shape': 'monolayer',
}

# %% [markdown]
# #### All

# %%
parameters = { 
    'concentration': 'readme.system.concentration',
    'shape': 'readme.system.shape',
    'velocity': 'readme.step_specific.probe_normal_approach.constant_indenter_velocity',
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

# filter out unwanted values
immutable_distinct_parameter_values = {k: [e for e in p if (
        isinstance(e, np.float64) and not np.isnan(e)) or (
        not isinstance(e, np.float64) and e is not None)] 
    for k, p in distinct_parameter_values.items()}

# %%
distinct_parameter_values = {k: v.copy() for k,v in immutable_distinct_parameter_values.items()}
distinct_parameter_values

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
parameters.keys()

# %%
keys = [*list()]

# %%
row

# %%
list_of_tuples = [(*[row[k] for k in parameters.keys()], row['uuid'][0]) 
    for _, row in res_df[[*list(parameters.keys()),'uuid']].iterrows()]

# %%
list_of_tuples

# %%
df_list = []
for (concentration, shape, velocity, uuid) in list_of_tuples:
    lookup_res = await dl.lookup(uuid)
    assert len(lookup_res) == 1
    
    uri = lookup_res[0]['uri']
    
    readme = await dl.readme(uri)
    # velocity = readme['step_specific']['probe_normal_approach']['constant_indenter_velocity']
    if isinstance(velocity, str):
        velocity = float(velocity)
        
    steps_per_datapoint = readme['step_specific']['probe_normal_approach']['netcdf_frequency']
    
    
    df = await read_forces(uri, file_name='default.txt')
    df = df*force_conversion_factor
        
    df['concentration'] = concentration
    df['shape'] = shape
    df['velocity'] = velocity
    df.reset_index(inplace=True)
    df['distance'] = initial_distance + velocity*steps_per_datapoint*fs_per_step*df['index']
    df.drop('index', axis=1, inplace=True)
    df_list.append(df)


# %%
res_df = pd.concat(df_list, axis=0, join="outer", ignore_index=True)

# %%
res_df

# %%
legend_pattern = '''{shape:}, concentration n = {concentration:} $\mathrm{{nm}}^{{-2}}$, velocity v = {velocity:} $\mathrm{{m}} \mathrm{{s}}^{{-1}}$.'''

# %%
offset_per_curve = 5

# %%
fig, ax = plt.subplots(1,1)

for i, (_, row) in enumerate(res_df[list(parameters.keys())].drop_duplicates().iterrows()):
    sub_df = res_df[ 
        (res_df['concentration'] == row['concentration']) & 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity'])
    ]
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
fig, ax = plt.subplots(1,1)

for _, row in res_df[list(parameters.keys())].drop_duplicates().iterrows():
    sub_df = res_df[ 
        (res_df['concentration'] == row['concentration']) & 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity'])
    ]
    
    sub_df.rolling(window=10,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))

    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
win = 10
offset_per_curve = 20

unique_parameter_sets = res_df[list(parameters.keys())].drop_duplicates()

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
    sub_df = res_df[ 
        (res_df['concentration'] == row['concentration']) & 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity'])
    ].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))

    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='y')

ax.legend()

# %% [markdown]
# ####  c = 1.25 nm^-2

# %%
selected_cocentration = 1.25 / concentration_conversion_factor

# %%
parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets_df['concentration'] == selected_cocentration].iterrows():
    sub_df = res_df[ 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity']) &
        (res_df['concentration'] == row['concentration'])
    ]
    
    sub_df.rolling(window=1,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
win = 10
offset_per_curve = 0

parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    (parameter_sets_df['shape'] == 'monolayer') &
    (parameter_sets_df['concentration'] == selected_cocentration)
].sort_values(by='velocity', ascending=False)


n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('bwr')(1. * i/(n-1)) for i in range(n)]

custom_cycler = cycler(color=new_colors) + cycler(linestyle=custom_linestyles[:n])

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
if offset_per_curve != 0:
    ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))


for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    sub_df = res_df[
        (res_df['concentration'] == row['concentration']) & 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity'])
    ].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('monolayer_c_1.25_per_sq_nm.svg', dpi=dpi)

ax.legend(framealpha=1)

plt.savefig('monolayer_c_1.25_per_sq_nm_legended.svg', dpi=dpi)

# %% [markdown]
# ####  c = 1.0 nm^-2

# %%
selected_cocentration = 1. / concentration_conversion_factor

# %%
parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets_df['concentration'] == selected_cocentration].iterrows():
    sub_df = res_df[ 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity']) &
        (res_df['concentration'] == row['concentration'])
    ]
    
    sub_df.rolling(window=1,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
win = 10
offset_per_curve = 0

parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    (parameter_sets_df['shape'] == 'monolayer') &
    (parameter_sets_df['concentration'] == selected_cocentration)
].sort_values(by='velocity', ascending=False)


n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('bwr')(1. * i/(n-1)) for i in range(n)]

custom_cycler = cycler(color=new_colors) + cycler(linestyle=custom_linestyles[:n])

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
if offset_per_curve != 0:
    ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))


for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    sub_df = res_df[
        (res_df['concentration'] == row['concentration']) & 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity'])
    ].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('monolayer_c_1.0_per_sq_nm.svg', dpi=dpi)

ax.legend(framealpha=1)

plt.savefig('monolayer_c_1.0_per_sq_nm_legended.svg', dpi=dpi)

# %% [markdown]
# ####  c = 0.5 nm^-2

# %%
selected_cocentration = 0.5 / concentration_conversion_factor

# %%
parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()

# %%
fig, ax = plt.subplots(1,1)

for _, row in parameter_sets_df[parameter_sets_df['concentration'] == selected_cocentration].iterrows():
    sub_df = res_df[ 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity']) &
        (res_df['concentration'] == row['concentration'])
    ]
    
    sub_df.rolling(window=1,center=True).mean().plot(
        x='distance', y='f_storeUnconstrainedForcesAve', ax=ax,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

# %%
win = 10
offset_per_curve = 0

parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    (parameter_sets_df['shape'] == 'monolayer') &
    (parameter_sets_df['concentration'] == selected_cocentration)
].sort_values(by='velocity', ascending=False)


n = len(restricted_parameter_sets_df) # Number of colors
new_colors = [plt.get_cmap('bwr')(1. * i/(n-1)) for i in range(n)]

custom_cycler = cycler(color=new_colors) + cycler(linestyle=custom_linestyles[:n])

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
if offset_per_curve != 0:
    ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))


for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    sub_df = res_df[
        (res_df['concentration'] == row['concentration']) & 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity'])
    ].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+i*offset_per_curve,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('monolayer_c_0.5_per_sq_nm.svg', dpi=dpi)

ax.legend(framealpha=1)

plt.savefig('monolayer_c_0.5_per_sq_nm_legended.svg', dpi=dpi)

# %% [markdown]
# ####  c = 0.5, 1.0, 1.25 nm^-2

# %%
selected_concentration_range = np.array([0.5, 1.25]) / concentration_conversion_factor

# %%
parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()

# %%
win = 10
offset_per_curve = 20
concentration_step = 0.25 / concentration_conversion_factor
offset = 0
previous_concentration = None

parameter_sets_df = res_df[list(parameters.keys())].drop_duplicates()
restricted_parameter_sets_df = parameter_sets_df[
    (parameter_sets_df['shape'] == 'monolayer') &
    (parameter_sets_df['concentration'] >= selected_concentration_range[0]) &
    (parameter_sets_df['concentration'] <= selected_concentration_range[1])
].sort_values(by=['concentration','velocity'], ascending=[True, False])

n1 = len(restricted_parameter_sets_df['concentration'].drop_duplicates())
n2 = len(restricted_parameter_sets_df['velocity'].drop_duplicates())

# n = len(restricted_parameter_sets_df) # Number of colors
ranges = [0.5 + (i+1)/n1*(j/(n2-1)-0.5) for i in range(n1) for j in range(n2)]


new_colors = [plt.get_cmap('coolwarm')(r) for r in ranges]

custom_cycler = cycler(color=new_colors) + cycler(linestyle=custom_linestyles[:n2])*n1

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_prop_cycle(custom_cycler)

# Make a plot with major ticks that are multiples of 20 and minor ticks that
# are multiples of 5.  Label major ticks with '%d' formatting but don't label
# minor ticks.
if offset_per_curve != 0:
    ax.yaxis.set_major_locator(MultipleLocator(offset_per_curve))
ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.yaxis.set_minor_locator(MultipleLocator(5))


ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

# For the minor ticks, use no labels; default NullFormatter.
ax.xaxis.set_minor_locator(MultipleLocator(1))


for i, (_, row) in enumerate(restricted_parameter_sets_df.iterrows()):
    if not previous_concentration:
        previous_concentration = row['concentration']
        
    offset += (row['concentration'] - previous_concentration)/concentration_step * offset_per_curve
    previous_concentration = row['concentration']

    
    sub_df = res_df[
        (res_df['concentration'] == row['concentration']) & 
        (res_df['shape'] == row['shape']) &
        (res_df['velocity'] == row['velocity'])
    ].rolling(window=win,center=True).mean()
    
    ax.plot(sub_df['distance'], sub_df['f_storeUnconstrainedForcesAve']+offset,
        label=legend_pattern.format(
            concentration=concentration_conversion_factor*row['concentration'],
            shape=row['shape'],
            velocity=velocity_conversion_factor*row['velocity']))
    
ax.set_xlabel(r'distance $d\, (\mathrm{\AA})$')
ax.set_ylabel(r'normal force $F_N\, (\mathrm{nN})$')

#ax.legend()
ax.grid(which='major', axis='both')

plt.savefig('monolayer_c_0.5_to_1.25_per_sq_nm.svg', dpi=dpi)

ax.legend(framealpha=1, labelcolor='linecolor')

plt.savefig('monolayer_c_0.5_to_1.25_per_sq_nm_legended.svg', dpi=dpi)

# %% [markdown]
# ### Reverse query UUIDs

# %%
project_id = {
    '$in': [
        '2021-01-28-sds-on-au-111-probe-and-substrate-approach',  # 10 m / s
        '2021-02-05-sds-on-au-111-probe-and-substrate-approach',  # 1 m / s
    ]
}

# %%
query = {
    'readme.project': project_id,
    #'readme.step': {'$regex': 'ProbeAnalysis'},
    'readme.system.shape': 'monolayer', 
    'readme.system.concentration': 1.25 / concentration_conversion_factor,
    #'readme.step_specific.probe_normal_approach.constant_indenter_velocity': -1. / velocity_conversion_factor,
    'readme.step_specific.probe_normal_approach.constant_indenter_velocity': '-1e-05'
}

# %%
parameters = { 
    'concentration': 'readme.system.concentration',
    'shape': 'readme.system.shape',
    'velocity': 'readme.step_specific.probe_normal_approach.constant_indenter_velocity',
}

# %%
await get_df_by_query(query)

# %%

# %%
query = {
    'readme.project': project_id,
    'readme.system.shape': 'monolayer', 
    'readme.system.concentration': 1. / concentration_conversion_factor,
    'readme.step_specific.probe_normal_approach.constant_indenter_velocity': '-1e-05'
}

# %%
parameters = { 
    'concentration': 'readme.system.concentration',
    'shape': 'readme.system.shape',
    'velocity': 'readme.step_specific.probe_normal_approach.constant_indenter_velocity',
}

# %%
await get_df_by_query(query)

# %%
