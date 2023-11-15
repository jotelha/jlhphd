#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:03:43 2020

@author: jotelha
"""


# In[20]:
import os, os.path
import datetime
# FireWorks functionality 
# from fireworks.utilities.dagflow import DAGFlow, plot_wf
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
    
from jlhpy.utilities.wf.packing.sub_wf_195_gromacs_relax_recover import GromacsRelaxationRecoverMain
# from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

project_id = '2020-07-29-ctab-on-au-111-indenter-passivation'
wfglp = GromacsRelaxationRecoverMain(
    project_id=project_id, 
    machine='juwels_devel',
    mode='trial',
)

wf = wfg.build_wf()