#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:30:18 2020

@author: jotelha
"""


# In[20]:
import os, os.path
import datetime
# FireWorks functionality 
from fireworks.utilities.dagflow import DAGFlow, plot_wf
from fireworks import Firework, LaunchPad, Workflow
from fireworks.utilities.filepad import FilePad


timestamp = datetime.datetime.now()
yyyymmdd = timestamp.strftime('%Y%m%d')
yyyy_mm_dd = timestamp.strftime('%Y-%m-%d')

# In[22]:

# prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'
prefix = '/home/jotelha/git/jlhphd'
work_prefix = '/home/jotelha/tmp/{date:s}_fw/'.format(date=yyyymmdd)
try:
    os.makedirs(work_prefix)
except:
    pass
os.chdir(work_prefix)

# the FireWorks LaunchPad
lp = LaunchPad.auto_load() #Define the server and database
# FilePad behaves analogous to LaunchPad
fp = FilePad.auto_load()

# In[30]:
    
from jlhpy.utilities.wf.substrate.chain_wf_fcc_substrate_creation import FCCSubstrateCreationChainWorkflowGenerator
# from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

project_id = '2020-09-25-au-111-fcc-substrate-creation-trial'

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': parameter_values})
wfg = FCCSubstrateCreationChainWorkflowGenerator(
    project_id=project_id, 
    integrate_push=True,
    description="Au(111) 180x180x180 Ang FCC substrate creation",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    infile_prefix=prefix,
    machine='juwels_devel',
    mode='trial',
    system = { 
        'substrate': {
            'element': 'Au',
            'lattice_constant': 4.075,
            'approximate_measures': [180., 180., 180.],
            'lmp': {  # LAMMPS-specific
                'type': 11,
            }
        },
    },
    step_specific={
        'minimization' : {
            'fixed_box': {
                'ftol': 1.e-6,
                'maxiter': 10000,
                'maxeval': 100000,
            },
            'relaxed_box': {
                'ftol': 1.e-6,
                'maxiter': 10000,
                'maxeval': 100000,
                'pressure': 1.0,
            }
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/TRIAL_DATASETS',
            'remote_dataset': None,  # initial source dataset
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()