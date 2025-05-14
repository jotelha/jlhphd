# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

# %%
import os
import datetime
import logging
import pymongo
import pandas as pd
import numpy as np
import json
import yaml
import dtool_lookup_api.asynchronous as dla

# %% [markdown]
# ### Logging

# %% init_cell=true
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
date_prefix = datetime.datetime.now().strftime("%Y%m%d")

# %%
iso_date_prefix = datetime.datetime.now().date().isoformat()

# %% init_cell=true
work_prefix = os.path.join( os.path.expanduser("~"), 'sandbox', date_prefix + '_fireworks_project_overview')

# %% init_cell=true
try:
    os.makedirs(work_prefix, exist_ok=True)
except FileExistsError as exc:
    print(exc)

# %% init_cell=true
os.chdir(work_prefix)

# %% [markdown]
# ## Overview on recent projects

# %%
import dtool_lookup_api.asynchronous as dl

# %%
pagination = {}

# %%
res = await dl.get_datasets_by_mongo_query({'readme.owners.name': {'$regex': 'Johannes'}}, pagination=pagination)

# %%
pagination

# %%
pagination = {}

# %%
res = await dl.get_datasets_by_mongo_query({'creator_username': 'hoermann4'}, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[0]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

# %%
len(res)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
res_df[['earliest','object_count','project']].iloc[:5]

# %% [markdown]
# ## Overview on recent production projects

# %%
project_overview = {
    '2022-08-26-sds-on-au-111-probe-on-substrate-lateral-sliding': {
         'pyfile': '20220826_au_probe_on_substrate_lateral_sliding.py',
         'group': '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach',
         'system': '2022-08-24-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration',
         'comment': 'lateral sliding on monolayers, one run per Ang between 10 and 20 Ang normal dist, above each 5 Ang',
     },
     '2022-08-24-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration': {
         'pyfile': '20220824_au_probe_on_substrate_wrap_join_and_equilibration.py',
         'group': '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach',
         'system': '2022-05-17-sds-on-au-111-probe-and-substrate-approach-frame-extraction',
     },  
     '2022-05-17-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration': {
         'pyfile': '20220517_au_probe_on_substrate_wrap_join_and_equilibration.py',
         'group': '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach',
         'system': '2022-05-17-sds-on-au-111-probe-and-substrate-approach-frame-extraction',
     },  
     '2022-05-17-sds-on-au-111-probe-and-substrate-approach-frame-extraction': {
         'pyfile': '20220517_au_probe_substrate_normal_approach_frame_extraction.py',
         'system': '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach',
     },
     '2022-05-17-sds-on-au-111-probe-on-substrate-lateral-sliding': {
         'pyfile': '20220517_au_probe_on_substrate_lateral_sliding.py',
         'group': '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach',
         'system': '2022-05-17-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration',
         'analysis': [
             '048_probe_on_monolayers_lateral_sliding',
         ]
     },
     '2022-04-19-sds-on-au-111-probe-on-substrate-lateral-sliding': { 
         'pyfile': '20220419_au_probe_on_substrate_lateral_sliding.py',
         'group': '2021-02-05-sds-on-au-111-probe-and-substrate-approach',
         'system': '2022-02-11-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration',
         'comment': 'lateral sliding on monolayers, except concentrations 1.25 and 2 per nm^2',
         'analysis': '048_probe_on_monolayers_lateral_sliding',
     },
     '2022-03-31-sds-on-au-111-probe-on-substrate-lateral-sliding': {
        'pyfile': '20220331_au_probe_on_substrate_lateral_sliding.py',
        'group': '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach',
        'system': '2022-02-18-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration',
        'comment': 'lateral sliding on and between hemicylinders at offset x = 0, y = 0 and y = -25, high resolution of 1 Ang below distance 10 Ang',
     },
     '2022-02-21-sds-on-au-111-probe-on-substrate-lateral-sliding': { 
         'pyfile': '20220221_au_probe_on_substrate_lateral_sliding.py',
         'group': '2021-02-05-sds-on-au-111-probe-and-substrate-approach',
         'system': '2022-02-11-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration',
         'comment': 'nmolecules 390 and 625 only'
     },
     '2022-02-19-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration': {  # hemicylinders, flanks
         'pyfile': '20220219_au_probe_on_substrate_wrap_join_and_equilibration.py',
         'group': '2021-12-09-sds-on-au-111-probe-and-substrate-merge-and-approach',
         'system': '2022-02-12-sds-on-au-111-probe-and-substrate-approach-frame-extraction',
     },
     '2022-02-18-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration': {  # hemicylinders
         'pyfile': '20220218_au_probe_on_substrate_wrap_join_and_equilibration.py', 
         'group': '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach',
         'system': '2022-02-11-sds-on-au-111-probe-and-substrate-approach-frame-extraction',
     },
     '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach': {
         'pyfile': '20220212_au_probe_substrate_merge_and_approach_monolayer_only.py',
         'probe': '2020-07-29-sds-on-au-111-indenter-passivation',
         'substrate': '2021-10-06-sds-on-au-111-substrate-passivation',
     },
     '2022-02-11-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration': {
         'pyfile': '20220211_au_probe_on_substrate_wrap_join_and_equilibration.py',
         'group': '2020-12-23-sds-on-au-111-probe-and-substrate-conversion',  # atom groups as defined for dpd equilibration, WRONG NOTE IN FILE
         'system': '2022-02-10-sds-on-au-111-probe-and-substrate-approach-frame-extraction', 
     },
     '2022-02-10-sds-on-au-111-probe-and-substrate-approach-frame-extraction': {  # frame extraction
         'pyfile': '20220210_au_probe_substrate_normal_approach_frame_extraction.py',
         'system': '2021-02-05-sds-on-au-111-probe-and-substrate-approach',
     },  
     '2022-02-12-sds-on-au-111-probe-and-substrate-approach-frame-extraction': {  # hemicylinders, flanks
         'pyfile': '20220212_au_probe_substrate_normal_approach_frame_extraction.py',
         'system': '2021-12-09-sds-on-au-111-probe-and-substrate-merge-and-approach',
     },
     '2022-02-11-sds-on-au-111-probe-and-substrate-approach-frame-extraction': {  # hemicylinders
         'pyfile': '20220211_au_probe_substrate_normal_approach_frame_extraction.py',
         'system': '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach',
     },
     '2022-02-01-sds-on-au-111-probe-and-substrate-merge-and-approach': {
         'pyfile': '20220201_au_probe_substrate_merge_and_approach.py',
         'probe': '2020-07-29-sds-on-au-111-indenter-passivation',
         'substrate': ' 2020-09-29-sds-on-au-111-substrate-passivation-trial',
     },
     '2022-02-01-sds-on-au-111-substrate-passivation': {  # sds-passivated substrates
         'pyfile': '20220201_au_substrate_solvation_sds.py',
         'substrate': '2020-11-25-au-111-150x150x150-fcc-substrate-creation',  # uuid: b5774404-e151-4398-bda9-36eb523a0ae7
     },
     '2021-01-31-sds-on-au-111-probe-on-substrate-lateral-sliding': {  # wrong year in project ID
         'pyfile': '20220131_au_probe_on_substrate_lateral_sliding.py',
         'group': '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach',
         'system': '2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration',
     },
     '2022-01-21-sds-on-au-111-probe-on-substrate-lateral-sliding': {
         'pyfile': '20220121_au_probe_on_substrate_lateral_sliding.py',
         'group': '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach',
         'system': '2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration',
     },
     #'2021-12-30-sds-on-au-111-probe-on-substrate-lateral-sliding',
     '2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration': {  # 
         'pyfile': [
             '20211227_au_probe_on_substrate_wrap_join_and_equilibration.py',
             '20211228_au_probe_on_substrate_dpd_equilibration.py',  # ?
         ],
         'group': '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach', 
         'system': '2021-12-09-sds-on-au-111-probe-and-substrate-approach-frame-extraction',
     },
     '2021-12-09-sds-on-au-111-probe-and-substrate-merge-and-approach': {  # hemicylinders, flank
         'pyfile': '20211209_au_probe_substrate_merge_and_approach.py',
         'probe': '2020-07-29-sds-on-au-111-indenter-passivation',   # uuid: b789ebc7-daec-488b-ba8f-e1c9b2d8fb47
         'substrate': '2020-09-29-sds-on-au-111-substrate-passivation-trial',  # uuid: a5582146-ac99-422b-91b4-dd12676b82a4     
     },
     '2021-12-09-sds-on-au-111-probe-and-substrate-approach-frame-extraction': {  # hemicylinders
         'pyfile': '20211209_au_probe_substrate_normal_approach_frame_extraction.py',
         'system': '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach',
     },  
     '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach': {  # hemicylinders
         'pyfile': '20211007_au_probe_substrate_merge_and_approach.py',
         'probe': '2020-07-29-sds-on-au-111-indenter-passivation',   # uuid: b789ebc7-daec-488b-ba8f-e1c9b2d8fb47
         'substrate': '2020-09-29-sds-on-au-111-substrate-passivation-trial',  # uuid: a5582146-ac99-422b-91b4-dd12676b82a4
     },
     '2021-10-06-sds-on-au-111-substrate-passivation': {  # sds-passivated substrates
         'pyfile': '20211006_au_substrate_solvation_sds.py',
         'substrate': '2020-11-25-au-111-150x150x150-fcc-substrate-creation',  # uuid: b5774404-e151-4398-bda9-36eb523a0ae7
     },         
     '2021-02-05-sds-on-au-111-probe-and-substrate-approach': {  # normal approach on monolayer
         'pyfile': [
             '20210205_au_probe_substrate_normal_approach.py',
             '20210224_au_probe_substrate_normal_approach.py'
         ],
         'system': '2020-12-23-sds-on-au-111-probe-and-substrate-conversion', 
     },    
     '2021-01-28-sds-on-au-111-probe-and-substrate-approach': {  # normal approach on monolayer
         'pyfile': '20210128_au_probe_substrate_normal_approach.py',
         'system': '2020-12-23-sds-on-au-111-probe-and-substrate-conversion',
     },
     '2020-12-23-sds-on-au-111-probe-and-substrate-conversion': {  # probe on substrate merged
         'pyfile': '20201222_au_pqrobe_substrate_merge.py',
         'probe': '2020-07-29-sds-on-au-111-indenter-passivation',
         'substrate': '2020-12-14-sds-on-au-111-substrate-passivation'
     },
     '2020-12-14-sds-on-au-111-substrate-passivation': {  # sds-passivated substrates
         'pyfile': '20201214_au_substrate_solvation_sds.py',
         'substrate': '2020-11-25-au-111-150x150x150-fcc-substrate-creation'  # uuid: b5774404-e151-4398-bda9-36eb523a0ae7
     },
     '2020-11-25-au-111-150x150x150-fcc-substrate-creation': {
         'pyfile': '20201019_au_111_150x150x150Ang_substrate_preparation_production.py',
     },
     '2020-09-29-sds-on-au-111-substrate-passivation-trial': {  # wrong year in label (?)
         'pyfile': '20210929_au_substrate_solvation_sds.py',
         'substrate': '2020-11-25-au-111-150x150x150-fcc-substrate-creation'  # uuid: b5774404-e151-4398-bda9-36eb523a0ae7
     },
     '2020-07-29-sds-on-au-111-indenter-passivation': {
         'pyfile': '20200728_au_cluster_solvation.py'
     },  # sds-passivated probes
}

# %%
projects = list(project_overview.keys())

# %%
query = make_query({'project': {'$in': projects}})

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_size=50)

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
for v in res_df["project"].values:
    print(f'* {v}')

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_query(query, page_number=i))

# %%
len(res)

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
res = await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_query(query, page_number=i))

# %%
len(res)

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]



# %%
res = await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
final_config_datasets = [row.to_dict() for _, row in final_config_df[['nmolecules','uuid']].iterrows()]

# %%
for d in final_config_datasets:
    d['uuid'] = d['uuid'][0]

# %%
final_config_datasets

# %%
# reconstruct concentrations:
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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
res = await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dla.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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

# %% [markdown]
# ## Overview on SDS-passivated substrates (2022/02/01)

# %%
project_id = "2022-02-01-sds-on-au-111-substrate-passivation"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# # Monolayers

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# fast approach on monolayer, 10 m /s 20210128_au_probe_substrate_normal_approach.py

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
with open(f"{project_id}_filtered_final_config.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on AFM approach and frame extraction (2022/02/12)
# Extraction 1 Ang-spaced below < 10 Ang

# %%
project_id = "2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
    #'system.surfactant.nmolecules': {'$exists': True},
    'system.concentration': {'$exists': True},
    # 'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    #'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.concentration',
    # 'shape': 'readme.system.surfactant.aggregates.shape',
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

# %%
res_df = pd.DataFrame(res)

# %%
res_df.sort_values(by='concentration')

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# ### Look at approach step

# %%
steps_of_interest = [
    "ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction:LAMMPSProbeNormalApproach:push_dtool"
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
# ### Overview on frame extraction

# %%
query = make_query({
    'project': project_id,
    #'system.surfactant.nmolecules': {'$exists': True},
    'system.concentration': {'$exists': True},
    'step_specific.frame_extraction.distance': {'$exists': True},
    # 'system.surfactant.aggregates.shape': {'$exists': True},
})

# %%
query

# %%
parameters = { 
    #'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.concentration',
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination, page_size=50)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i, page_size=50))

# %%
res

# %%
len(res)

# %%
res_df = pd.DataFrame(res)

# %%
res_df[res_df["distance"] == 4.0]

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination, page_size=50)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i, page_size=50))

# %%
res_df = pd.DataFrame(res)

# %%
res_df.sort_values(['concentration','distance'])

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
    "ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction:ProbeAnalysis3DAndFrameExtraction:ForeachPushStub:push_dtool",
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
with open(f"{project_id}_filtered_final_config.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on frame extraction (2022-05-17)
# from monolayer probing, '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach'

# %%
project_id = "2022-05-17-sds-on-au-111-probe-and-substrate-approach-frame-extraction"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.concentration',
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
    d for d in immutable_distinct_parameter_values['distance'] if (d > 0) and (d < 10 or d % 5. == 0) and (d < 25)]

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination, page_size=50)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i, page_size=50))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination, page_size=51)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i, page_size=51))

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
duplicated = res_df[["step","distance", 'concentration', 'nmolecules', 'x_shift', 'y_shift']].duplicated()

# %%
res_df[duplicated]

# %%
len(res_df)

# %%
set(parameters.keys())

# %%
res_pivot = res_df.pivot(values='uuid', index=['step', 'distance'], columns=list(set(parameters.keys()) - set(['distance']))) 
res_pivot.style.apply(highlight_nan)

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# ### Look at subset of equilibrated configurations

# %%
steps_of_interest = [
    "WrapJoinAndDPDEquilibration:LAMMPSEquilibrationDPD:push_dtool",
    "LAMMPSEquilibrationDPD:push_dtool",
]


# %%
# nmolecules_of_interest = [390, 625]
nmolecules_of_interest = [156, 235, 313, 703]

# %%
final_config_df = res_df[res_df['step'].isin(steps_of_interest) & res_df["nmolecules"].isin(nmolecules_of_interest)]

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
with open(f"{project_id}_filtered_LAMMPSEquilibrationDPD.json", 'w') as f:
    json.dump(final_config_datasets, f, indent=4)

# %% [markdown]
# ## Overview on wrap-join and on repeated DPD equilibration (2022-04-20) INVALID
#
# INVALID. Extracted same frame at zero dist for all trajectories.

# %%
project_id = "2022-04-20-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.concentration',
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# ## Overview on wrap-join and on repeated DPD equilibration (2022-05-17)

# %%
project_id = "2022-05-17-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.concentration',
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# ## Overview on lateral sliding (2022-02-21)

# %%
project_id = "2022-02-21-sds-on-au-111-probe-on-substrate-lateral-sliding"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    #'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.surfactant.surface_concentration',
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# ## Overview on lateral sliding (2022-04-19)

# %%
project_id = "2022-04-19-sds-on-au-111-probe-on-substrate-lateral-sliding"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    #'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.surfactant.surface_concentration',
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# ## Overview on lateral sliding (2022-05-17)

# %%
project_id = "2022-05-17-sds-on-au-111-probe-on-substrate-lateral-sliding"

# %%
# queries to the data base are simple dictionaries
query = make_query({
    'project': project_id,
})

# %%
query

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

# %%
readme

# %%
# no concentration, only molecules

# %%
parameters = { 
    #'nmolecules': 'readme.system.surfactant.nmolecules',
    'concentration': 'readme.system.concentration',
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# # Hemicylinders

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

# %%
res_df = pd.DataFrame(res)

# %%
res_df

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

# %%
res_df = pd.DataFrame(res)

# %%
res_df

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

# %%
res_df = pd.DataFrame(res)

# %%
res_df

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
query

# %%
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# on and between hemicylinders, at narrow intervals of 1 Ang for Au-Au distance y 10 Ang

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

# %%
res_df = pd.DataFrame(res)

# %%
res_df

# %%
len(res_df)

# %%
res_pivot.to_excel(f"{project_id}_uuids.xlsx")

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
# from 2021-12-09-sds-on-au-111-probe-and-substrate-merge-and-approachon hemicylinder flanks approach at -1e-5 m / s
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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

# %%
res_df = pd.DataFrame(res)

# %%
res_df

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
datasets = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
res = await dl.get_datasets_by_mongo_query(query=query, pagination=pagination)

# %%
pagination

# %%
readme_str = await dl.get_readme(res[-1]['uri'])

# %%
readme = yaml.safe_load(readme_str)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]

# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
            "earliest": pymongo.DESCENDING,
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
        }
    }
]



# %%
res = await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, pagination=pagination)

# %%
pagination

# %%
res = []
for i in range(1, pagination['total_pages']+1):
    res.extend(await dl.get_datasets_by_mongo_aggregation(aggregation_pipeline, page_number=i))

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
# on hemicylinders only, lateral offset only x, y = (25,0)

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
# between hemicylinders, offset x, y = (0, -25) only

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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

# %% [markdown]
# ## Overview on lateral sliding (2022-01-31)
# on hemicylinders, only at lateral offset 0,0

# %%
project_id = "2021-01-31-sds-on-au-111-probe-on-substrate-lateral-sliding" # wrong date in project id

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
# ## Overview on lateral sliding (2022-03-31)
# on hemicylinders

# %%
project_id = "2022-03-31-sds-on-au-111-probe-on-substrate-lateral-sliding"

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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
            **{label: pymongo.DESCENDING for label in parameters.keys()},
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
