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
# # Project overview

# %% [markdown]
# Evaluation of datasets on dtool lookup server.

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


# %% init_cell=true
def make_query(d:dict={}):
    q = {'creator_username': 'hoermann4'}
    for k, v in d.items():
        q['readme.'+k] = v
    return q


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

# %%
iso_date_prefix = datetime.datetime.now().date().isoformat()

# %% init_cell=true
work_prefix = os.path.join( os.path.expanduser("~"), 'sandbox', date_prefix + '_fireworks_project_overview')

# %% init_cell=true
try:
    os.mkdir(work_prefix)
except FileExistsError as exc:
    print(exc)

# %% init_cell=true
os.chdir(work_prefix)

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

# %%
res_df.to_html(f'{iso_date_prefix}-project-overview.html')
res_df.to_excel(f'{iso_date_prefix}-project-overview.xlsx')
res_df.to_json(f'{iso_date_prefix}-project-overview.json', indent=4, orient='records')

# %% [markdown]
# ## Overview on SDS-passivated indenters (2020/07/29)

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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

# %%
final_config_datasets = [row.to_dict() for _, row in final_config_df[['nmolecules','uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %%
# reconstruct cncentrations:
r = 2.5 # 2.5 nm
A = 4*np.pi*r**2

# %%
525 / A


# %%
def round_to_multiple_of_base(x, prec=2, base=0.25):     
    return (base * (np.array(x) / base).round()).round(prec)


# %%
for c in final_config_datasets:
    c['concentration'] = round_to_multiple_of_base(c['nmolecules'] / A, base=0.25)

# %%
final_config_datasets

# %%

# %%
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on SDS-passivated substrates (2020/12/14)

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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

# %%
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on SDS-passivated substrates (2021/10/06)

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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

# %%
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ### Look at last step, monolayer only

# %%
steps_of_interest = [
    #"SubstratePassivation:HemicylindricalPackingAndEquilibartion:GromacsMinimizationEquilibrationRelaxation:GromacsRelaxation:push_dtool",
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

# %%
with open(f"{project_id}_final_configs_monolayer_only.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %%

# %% [markdown]
# ## Overview on LAMMPS equlibration of passivated substrate-probe systems (2020/12/23)

# %%
project_id = "2020-12-23-sds-on-au-111-probe-and-substrate-conversion"

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "ProbeOnSubstrateMergeConversionMinimizationAndEquilibration:ProbeOnSubstrateMinimizationAndEquilibration:LAMMPSEquilibrationDPD:push_dtool"
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
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

# %% [markdown]
# ### Look at approach step

# %%
steps_of_interest = [
    "ProbeOnSubstrateNormalApproach:LAMMPSProbeNormalApproach:push_dtool"
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
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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

# %%
with open(f"{project_id}_ProbeAnalysis.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ### Look at normal approach step

# %%
steps_of_interest = [
     "ProbeOnSubstrateNormalApproach:LAMMPSProbeNormalApproach:push_dtool",
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
with open(f"{project_id}_LAMMPSProbeNormalApproach.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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

# %%
with open(f"{project_id}_LAMMPSEquilibrationDPD.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on AFM approach (2021/01/28)
# fast appaorach on monolayer, 10 m /s 20210128_au_probe_substrate_normal_approach.py

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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

# %%
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

# %% [markdown]
# ### Look at approach step

# %%
steps_of_interest = [
    "ProbeOnSubstrateNormalApproach:LAMMPSProbeNormalApproach:push_dtool"
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
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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

# %%
with open(f"{project_id}_ProbeAnalysis.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ### Look at normal approach step

# %%
steps_of_interest = [
     "ProbeOnSubstrateNormalApproach:LAMMPSProbeNormalApproach:push_dtool",
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
with open(f"{project_id}_LAMMPSProbeNormalApproach.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on frame extraction (2022-02-10)
# from monolayer probing, '2021-02-05-sds-on-au-111-probe-and-substrate-approach'

# %%
project_id = "2022-02-10-sds-on-au-111-probe-and-substrate-approach-frame-extraction"

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

# %% [markdown]
# #### Filter only values of interest

# %%
immutable_distinct_parameter_values['distance'] = [
    d for d in immutable_distinct_parameter_values['distance'] if d < 10 or d % 5. == 0]

# %%
immutable_distinct_parameter_values

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


# %%
res_pivot.to_excel(f"{project_id}_filtered_steps.xlsx")

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

# %%
res_pivot

# %%
res_pivot.to_excel(f"{project_id}_filtered_uuids.xlsx")

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
# df = pd.DataFrame(final_config_datasets)
# df.to_clipboard(index=False,header=False)

# %%
with open(f"{project_id}_filtered_final_config.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on wrap-join and on repeated DPD equilibration (2022-02-11)

# %%
project_id = "2022-02-11-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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
# ### Look at wrap-join step

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
with open(f"{project_id}_WrapJoinDataFile.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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
with open(f"{project_id}_LAMMPSEquilibrationDPD.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on merge & AFM approach, on & between hemicylinders (2021-10-07)

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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

# %%
with open(f"{project_id}_analysis.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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

# %%
with open(f"{project_id}_LAMMPSProbeNormalApproach.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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

# %%
with open(f"{project_id}_LAMMPSEquilibrationDPD.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on frame extraction (2021-12-09)

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %%
# df = pd.DataFrame(final_config_datasets)
# df.to_clipboard(index=False,header=False)

# %% [markdown]
# ## Overview on frame extraction (2022-02-11)
# from '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach' on & between hemicylinders approach at -1e-5 m / s

# %%
project_id = "2022-02-11-sds-on-au-111-probe-and-substrate-approach-frame-extraction"

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ### Filter by parameters

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

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
# #### Filter only values of interest

# %%
immutable_distinct_parameter_values['distance'] = [
    d for d in immutable_distinct_parameter_values['distance'] if d < 10 or d % 5. == 0]

# %%
immutable_distinct_parameter_values['x_shift'] = [0]
immutable_distinct_parameter_values['y_shift'] = [0,-25.0]


# %%
immutable_distinct_parameter_values

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


# %%
res_pivot.to_excel(f"{project_id}_filtered_steps.xlsx")

# %% [markdown]
# #### Overview on UUIDs

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

# %%
res_pivot

# %%
res_pivot.to_excel(f"{project_id}_filtered_uuids.xlsx")

# %% [markdown]
# #### Look at last step

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
with open(f"{project_id}_filtered_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on wrap-join and on repeated DPD equilibration (2022-02-18)

# %%
project_id = "2022-02-18-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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
# ### Look at wrap-join step

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
with open(f"{project_id}_WrapJoinDataFile.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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
with open(f"{project_id}_LAMMPSEquilibrationDPD.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# #### Datasets at x = 25, y = 0 (on hemicylinders)

# %%
x_shift, y_shift = (25.0, 0.0)

# %%
# y shift 0: on hemicylinders
selection = (final_config_df['x_shift'] == x_shift) & (final_config_df['y_shift'] == y_shift)

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

# %%
with open(f"{project_id}_LAMMPSEquilibrationDPD_x_{x_shift}_y_{y_shift}.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# #### Datasets at x = 0, y = 0 (on hemicylinders)

# %%
x_shift, y_shift = (0.0, 0.0)

# %%
# y shift 0: on hemicylinders
selection = (final_config_df['x_shift'] == x_shift) & (final_config_df['y_shift'] == y_shift)

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

# %%
with open(f"{project_id}_LAMMPSEquilibrationDPD_x_{x_shift}_y_{y_shift}.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# #### Datasets at x = 0, y = -25 (between hemicylinders)

# %%
x_shift, y_shift = (0.0, -25.0)

# %%
# y shift -25: between hemicylinders
selection = (final_config_df['x_shift'] == x_shift) & (final_config_df['y_shift'] == y_shift)

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

# %%
with open(f"{project_id}_LAMMPSEquilibrationDPD_x_{x_shift}_y_{y_shift}.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on merge & AFM approach, on hemicylinder flanks (2021-12-09)

# %%
project_id = "2021-12-09-sds-on-au-111-probe-and-substrate-merge-and-approach"

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

# %% [markdown]
# ### Look at last step

# %%
steps_of_interest = [
    "ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction:ProbeAnalysis3D:push_dtool",
    "ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction:ProbeAnalysis3DAndFrameExtraction:ProbeAnalysis3D:push_dtool",
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
with open(f"{project_id}_analysis.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ### Look at normal approach step

# %%
steps_of_interest = [
    "ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction:LAMMPSProbeNormalApproach:push_dtool",
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
with open(f"{project_id}_LAMMPSProbeNormalApproach.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ### Look at DPD equilibration step

# %%
steps_of_interest = [
    "ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction:LAMMPSEquilibrationDPD:push_dtool"
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

# %%
with open(f"{project_id}_LAMMPSEquilibrationDPD.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on frame extraction (2022-02-12)
# from 2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach on & between hemicylinders approach at -1e-5 m / s
#

# %%
project_id = "2022-02-12-sds-on-au-111-probe-and-substrate-approach-frame-extraction"

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
# y-shift 12.5, -37.5: "upper" flank, 37.5, "lower" flank

# %%
res_pivot

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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
with open(f"{project_id}_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ### Filter by parameters

# %% [markdown]
# #### Identify distinct parameter values

# %%
query = make_query({
    'project': project_id,
    'system.surfactant.nmolecules': {'$exists': True},
    #'system.surfactant.aggregates.shape': {'$exists': True},
})

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
# #### Filter only values of interest

# %%
immutable_distinct_parameter_values['distance'] = [
    d for d in immutable_distinct_parameter_values['distance'] if d < 10 or d % 5. == 0]

# %%
immutable_distinct_parameter_values['x_shift'] = [0]
immutable_distinct_parameter_values['y_shift'] = [12.5,37.5] # former on upper, latter on lower flank


# %%
immutable_distinct_parameter_values

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


# %%
res_pivot.to_excel(f"{project_id}_filtered_steps.xlsx")

# %% [markdown]
# #### Overview on UUIDs

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

# %%
res_pivot

# %%
res_pivot.to_excel(f"{project_id}_filtered_uuids.xlsx")

# %% [markdown]
# #### Look at last step

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
with open(f"{project_id}_filtered_final_configs.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on frame extraction (2021-12-09)

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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

# %%
res_pivot

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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
# df = pd.DataFrame(final_config_datasets)
# df.to_clipboard(index=False,header=False)

# %%
with open(f"{project_id}_final_config.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on wrap-join and on repeated DPD equilibration (2022-02-19)

# %% [markdown]
# DPD equlibration from 2021-12-09-sds-on-au-111-probe-and-substrate-merge-and-approach, configs from 2022-02-12-sds-on-au-111-probe-and-substrate-approach-frame-extraction on hemicylinder flanks

# %%
project_id = "2022-02-19-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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
# ### Look at wrap-join step

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
with open(f"{project_id}_WrapJoinDataFile.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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
with open(f"{project_id}_LAMMPSEquilibrationDPD.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on wrap-join and on repeated DPD equilibration (2021-12-27)

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


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

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
# ### Look at wrap-join step

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
with open(f"{project_id}_WrapJoinDataFile.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

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
with open(f"{project_id}_LAMMPSEquilibrationDPD.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %%
#df = pd.DataFrame(final_config_datasets)
#df.to_clipboard(index=False,header=False)

# %% [markdown]
# #### Datasets at x = 25, y = 0 (on hemicylinders)

# %%
x_shift, y_shift = (25.0, 0.0)

# %%
# y shift 0: on hemicylinders
selection = (final_config_df['x_shift'] == x_shift) & (final_config_df['y_shift'] == y_shift)

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

# %%
with open(f"{project_id}_LAMMPSEquilibrationDPD_x_{x_shift}_y_{y_shift}.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# #### Datasets at x = 0, y = 0 (on hemicylinders)

# %%
x_shift, y_shift = (0.0, 0.0)

# %%
# y shift 0: on hemicylinders
selection = (final_config_df['x_shift'] == x_shift) & (final_config_df['y_shift'] == y_shift)

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

# %%
with open(f"{project_id}_LAMMPSEquilibrationDPD_x_{x_shift}_y_{y_shift}.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# #### Datasets at x = 0, y = -25 (between hemicylinders)

# %%
x_shift, y_shift = (0.0, -25.0)

# %%
# y shift -25: between hemicylinders
selection = (final_config_df['x_shift'] == x_shift) & (final_config_df['y_shift'] == y_shift)

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

# %%
with open(f"{project_id}_LAMMPSEquilibrationDPD_x_{x_shift}_y_{y_shift}.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on lateral sliding (2021-12-30)

# %%
project_id = "2021-12-30-sds-on-au-111-probe-on-substrate-lateral-sliding"

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

# %%
res_df = pd.DataFrame(res)

# %%
sorted(list(parameters.keys()))

# %%
res_df.sort_values(by=sorted(list(parameters.keys())))

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
distinct_parameter_values

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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

# %% [markdown]
# ### Overview on UUIDs

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
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

# %% [markdown]
# ### Look at last step (analysis)

# %%
steps_of_interest = [
    "ProbeOnSubstrateLateralSliding:ProbeAnalysis3D:push_dtool"
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
with open(f"{project_id}_analysis.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on lateral sliding (2022-01-21)

# %%
project_id = "2022-01-21-sds-on-au-111-probe-on-substrate-lateral-sliding"

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

# %%
res_df = pd.DataFrame(res)

# %%
sorted(list(parameters.keys()))

# %%
res_df.sort_values(by=sorted(list(parameters.keys())))

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
distinct_parameter_values

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
res_pivot = res_df.pivot_table(
    values='object_count', index=['step'], columns=list(parameters.keys()), 
    aggfunc=pd.notna, fill_value=False)
res_pivot = res_pivot.reindex(
    res_df.groupby(by='step')['earliest'].min().sort_values(ascending=True).index)
res_pivot = res_pivot.style.apply(highlight_bool)
res_pivot


# %%
res_pivot.to_excel(f"{project_id}_steps.xlsx")

# %% [markdown]
# ### Overview on UUIDs

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
res_pivot = res_df.pivot(values='uuid', index=['step'], columns=list(parameters.keys()))
res_pivot.style.apply(highlight_nan)

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

# %% [markdown]
# ### Look at last step (analysis)

# %%
steps_of_interest = [
    "ProbeOnSubstrateLateralSliding:ProbeAnalysis3D:push_dtool"
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
with open(f"{project_id}_analysis.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %%

# %%
