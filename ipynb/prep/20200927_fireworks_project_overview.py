#!/usr/bin/env python
# coding: utf-8

# # Prepare AFM tip solvation

# This notebook demonstrates deposition of an SDS adsorption layer on a non-spherical AFM tip model.

# ## Initialization

# ### IPython magic

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '3')


# ### Imports

# In[2]:


import ase.io # here used for reading pdb files
from ase.visualize import view
from ase.visualize.plot import plot_atoms # has nasty offset issues
from cycler import cycler # here used for cycling through colors in plots
import datetime
import fabric # for pythonic ssh connections
from fireworks import LaunchPad, Firework, Tracker, Workflow 
from fireworks import FileTransferTask, PyTask, ScriptTask

# FireWorks functionality 
from fireworks import Firework, LaunchPad, ScriptTask, Workflow
from fireworks.user_objects.firetasks.templatewriter_task import TemplateWriterTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask, GetFilesTask, GetFilesByQueryTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask
from fireworks.utilities.filepad import FilePad # direct FilePad access, similar to the familiar LaunchPad

import glob
import gc # manually clean up memory with gc.collect()
import gromacs # GromacsWrapper, here used for evoking gmc commands, reading and writing .ndx files
# from io import StringIO, TextIOWrapper
import io
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

# ### Logging

# In[3]:


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ParmEd needs to know the GROMACS topology folder, usually get this from 
# envionment variable `GMXLIB`:

# ### Function definitions

# In[4]:


def find_undeclared_variables(infile):
    """identify all variables evaluated in a jinja 2 template file"""
    env = jinja2.Environment()
    with open(infile) as template_file:
        parsed = env.parse(template_file.read())

    undefined = jinja2.meta.find_undeclared_variables(parsed)
    return undefined


# In[5]:


def plot_side_views_with_spheres(atoms, cc, R, figsize=(12,4), fig=None, ax=None):
    """
    Plots xy, yz and zx projections of atoms and sphere(s) 
    
    Parameters
    ----------
    atoms: ase.atoms
        
    cc: (N,3) ndarray
        centers of spheres
    R:  (N,) ndarray
        radii of spheres
    figsize: 2-tuple, default (12,4)
    fig: matplotlib.figure, default None
    ax:  list of three matploblib.axes objects
    """
    
    logger = logging.getLogger(__name__)
    
    atom_radii = 0.5
    
    cc = np.array(cc,ndmin=2)
    logger.info("C({}) = {}".format(cc.shape,cc))
    R = np.array(R,ndmin=1)
    logger.info("R({}) = {}".format(R.shape,R))
    xmin = atoms.get_positions().min(axis=0)
    xmax = atoms.get_positions().max(axis=0)
    logger.info("xmin({}) = {}".format(xmin.shape,xmin))
    logger.info("xmax({}) = {}".format(xmax.shape,xmax))
    
    ### necessary due to ASE-internal atom position computations
    # see https://gitlab.com/ase/ase/blob/master/ase/io/utils.py#L69-82
    X1 = xmin - atom_radii
    X2 = xmax + atom_radii

    M = (X1 + X2) / 2
    S = 1.05 * (X2 - X1)

    scale = 1
    internal_offset = [ np.array(
        [scale * np.roll(M,i)[0] - scale * np.roll(S,i)[0] / 2, 
         scale * np.roll(M,i)[1] - scale * np.roll(S,i)[1] / 2]) for i in range(3) ]

    ### 
    atom_permut = [ atoms.copy() for i in range(3) ]

    for i, a in enumerate(atom_permut):
        a.set_positions( np.roll(a.get_positions(),i,axis=1) )

    rot      = ['0x,0y,0z']*3#,('90z,90x'),('90x,90y,0z')]
    label    = [ np.roll(np.array(['x','y','z'],dtype=str),i)[0:2] for i in range(3) ]
    
    # dim: sphere, view, coord
    center   = np.array([ 
        [ np.roll(C,i)[0:2] - internal_offset[i] for i in range(3) ] for C in cc ])

    logger.info("projected cc({}) = {}".format(center.shape,center))
    
    color_cycle = cycler(color=[
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
    circle   = [ [ plt.Circle( c , r, fill=False, **col) for c in C ] for C,r,col in zip(center,R,color_cycle) ]
    margin   = 1.1
    
    # dim: view, coord, minmax (i.e., 3,2,2)
    plot_bb = np.rollaxis( np.array(
        [ np.min(center - margin*(np.ones(center.T.shape)*R).T,axis=0),
          np.max(center + margin*(np.ones(center.T.shape)*R).T,axis=0) ] ).T, 1, 0)
    
    #plot_bb  = np.array( [ [
    #    [ [np.min(c[0]-margin*R[0]), np.max(c[0]+margin*R[0])], 
    #      [np.min(c[1]-margin*R[0]), np.max(c[1]+margin*R[0])] ] ) for c in C ] for C,r in zip(center,R) ] )
    logger.info("projected bb({}) = {}".format(plot_bb.shape,plot_bb))
    
    if ax is None:
        fig, ax = plt.subplots(1,3,figsize=figsize)
            
    (ax_xy, ax_xz, ax_yz)  = ax[:]
    logger.info("iterators len(atom_permut={}, len(ax)={}, len(rot)={}, len(circle)={}".format(
            len(atom_permut),len(ax),len(rot),len(circle)))
    
    #logger.info("len(circle)={}".format(len(circle))

    #for aa, a, r, C in zip(atom_permut,ax,rot,circle):
    for i, a in enumerate(ax):
        # rotation strings see https://gitlab.com/ase/ase/blob/master/ase/utils/__init__.py#L235-261
        plot_atoms(atom_permut[i],a,rotation=rot[i],radii=0.5,show_unit_cell=0,offset=(0,0))
        for j, c in enumerate(circle):
            logger.info("len(circle[{}])={}".format(j,len(c)))
            a.add_patch(c[i])

    for a,l,bb in zip(ax,label,plot_bb): 
        a.set_xlabel(l[0])
        a.set_ylabel(l[1])
        a.set_xlim(*bb[0,:])
        a.set_ylim(*bb[1,:])

    return fig, ax


# In[6]:


def pack_sphere(C,
    R_inner_constraint, # shell inner radius
    R_outer_constraint, # shell outer radius
    sfN, # number  of surfactant molecules
    inner_atom_number, # inner atom
    outer_atom_number, # outer atom
    surfactant = 'SDS',
    counterion = 'NA',
    tolerance = 2):
    """Creates context for filling Jinja2 PACKMOL input template in order to
    generate preassembled surfactant spheres with couinterions at polar heads"""

    logger = logging.getLogger(__name__)
    logger.info(
        "sphere with {:d} surfactant molecules in total.".format(sfN ) )

    # sbX, sbY, sbZ = sb_measures

    # spheres parallelt to x-axis
    sphere = {}
    ionsphere = {}

    # surfactant spheres
    #   inner constraint radius: R + 1*tolerance
    #   outer constraint radius: R + 1*tolerance + l_surfactant
    # ions between cylindric planes at
    #   inner radius:            R + 1*tolerance + l_surfactant
    #   outer radius:            R + 2*tolerance + l_surfactant
    sphere["surfactant"] = surfactant

    
    sphere["inner_atom_number"] = inner_atom_number
    sphere["outer_atom_number"] = outer_atom_number

    sphere["N"] = sfN
        
    sphere["c"] = C

    sphere["r_inner"] = R_inner
    sphere["r_inner_constraint"] = R_inner_constraint
    sphere["r_outer_constraint"] = R_outer_constraint
    sphere["r_outer"] = R_outer
    
    logging.info(
        "sphere with {:d} molecules at {}, radius {}".format(
        sphere["N"], sphere["c"], sphere["r_outer"]))

    # ions at outer surface
    ionsphere["ion"] = counterion

    
    ionsphere["N"] = sphere["N"]
    ionsphere["c"] = sphere["c"]
    ionsphere["r_inner"] = sphere["r_outer"]
    ionsphere["r_outer"] = sphere["r_outer"] + tolerance


    # experience shows: movebadrandom advantegous for (hemi-) spheres
    context = {
        'spheres':     [sphere],
        'ionspheres':  [ionsphere],
        'movebadrandom': True,
    }
    return context


# In[7]:


def memuse():
    """Quick overview on memory usage of objects in Jupyter notebook"""
    # https://stackoverflow.com/questions/40993626/list-memory-usage-in-ipython-and-jupyter
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    return sorted([(x, sys.getsizeof(globals().get(x))) for x in dir(sys.modules['__main__']) if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# ### Global settings

# In[8]:


os.environ['GMXLIB']


# In[9]:


pmd.gromacs.GROMACS_TOPDIR = os.environ['GMXLIB']


# In[10]:


# prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'
prefix = '/home/jotelha/git/N_surfactant_on_substrate_template'


# In[11]:


work_prefix = '/home/jotelha/tmp/20200329_fw/'


# In[11]:


os.chdir(work_prefix)


# ### HPC-related settings

# In[ ]:


hpc_max_specs = {
    'forhlr2': {
        'fw_queue_category':   'forhlr2_queue',
        'fw_noqueue_category': 'forhlr2_noqueue',
        'queue':'develop',
        'physical_cores_per_node': 20,
        'logical_cores_per_node':  40,
        'nodes': 4,
        'walltime':  '00:60:00'
    },
    'juwels_devel': {
        'fw_queue_category':   'juwels_queue',
        'fw_noqueue_category': 'juwels_noqueue',
        'queue':'devel',
        'physical_cores_per_node': 48,
        'logical_cores_per_node':  96,
        'nodes': 8,
        'walltime':  '00:30:00'   
    },
    'juwels': {
        'fw_queue_category':   'juwels_queue',
        'fw_noqueue_category': 'juwels_noqueue',
        'queue':'batch',
        'physical_cores_per_node': 48,
        'logical_cores_per_node':  96,
        'nodes': 1024,
        'walltime':  '00:30:00'   
    }
}


# In[ ]:


std_exports = {
    'forhlr2': {
        'OMP_NUM_THREADS': 1,
        'KMP_AFFINITY':    "'verbose,compact,1,0'",
        'I_MPI_PIN_DOMAIN':'core'
    },
    'juwels': {
        'OMP_NUM_THREADS': 1,
        'KMP_AFFINITY':    "'verbose,compact,1,0'",
        'I_MPI_PIN_DOMAIN':'core'
    }
}


# ### FireWorks LaunchPad and FilePad

# In[ ]:


# the FireWorks LaunchPad
lp = LaunchPad.auto_load() #Define the server and database


# In[ ]:


# FilePad behaves analogous to LaunchPad
fp = FilePad.auto_load()


# #### Sub-WF: PACKMOL

# In[27]:


def sub_wf_pack(d, fws_root):    
    global project_id,         C, R_inner_constraint, R_outer_constraint,         tail_atom_number, head_atom_number, surfactant, counterion, tolerance,         fw_name_template, hpc_max_specs, machine
    # TODO: instead of global variables, use class
    
    fw_list = []
### Template
    
    files_in = {'input_file': 'input.template' }
    files_out = { 'input_file': 'input.inp' }
    
    # exports = std_exports[machine].copy()
        
    # Jinja2 context:
    packmol_script_context = {
        'header':        '{:s} packing SDS around AFM probe model'.format(project_id),
        'system_name':   '{:d}_SDS_on_50_Ang_AFM_tip_model'.format(d["nmolecules"]),
        'tolerance':     tolerance,
        'write_restart': True,

        'static_components': [
            {
                'name': 'indenter'
            }
        ]
    }

    # use pack_sphere function at the notebook's head to generate template context
    packmol_script_context.update(
        pack_sphere(
            C,R_inner_constraint,R_outer_constraint, d["nmolecules"], 
            tail_atom_number+1, head_atom_number+1, surfactant, counterion, tolerance))
    
    ft_template = TemplateWriterTask( {
        'context': packmol_script_context,
        'template_file': 'input.template',
        'template_dir': '.',
        'output_file': 'input.inp'} )
    
    
    fw_template = Firework([ft_template],
        name = ', '.join(('template', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in,
            '_files_out': files_out, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'fill_template',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_template)

### PACKMOL

    files_in = {
        'input_file': 'input.inp',
        'indenter_file': 'indenter.pdb',
        'surfatcant_file': '1_SDS.pdb',
        'counterion_file': '1_NA.pdb' }
    files_out = {
        'data_file': '*_packmol.pdb'}
    
    ft_pack = CmdTask(
        cmd='packmol',
        opt=['< input.inp'],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True)
  
    fw_pack = Firework([ft_pack],
        name = ', '.join(('packmol', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_queue_category'],
            '_queueadapter': {
                'queue':           hpc_max_specs[machine]['queue'],
                'walltime' :       hpc_max_specs[machine]['walltime'],
                'ntasks':          1,
            },
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'packmol',
                 **d
            }
        },
        parents = [ *fws_root, fw_template] )
    
    fw_list.append(fw_pack)

    return fw_list, fw_pack

def sub_wf_pack_push(d, fws_root):
    global project_id, hpc_max_specs, machine

    fw_list = []   

    files_in = {'data_file': 'packed.pdb' }
    
    fts_push = [ AddFilesTask( {
        'compress': True ,
        'paths': "packed.pdb",
        'metadata': {
            'project': project_id,
            'datetime': str(datetime.datetime.now()),
            'type':    'initial_config',
             **d } 
        } ) ]
    
    fw_push = Firework(fts_push,
        name = ', '.join(('transfer', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'transfer',
                 **d
            }
        },
        parents = fws_root )
        
    fw_list.append(fw_push)
    
    return fw_list, fw_push


# #### Sub-WF: GMX prep

# In[28]:


def sub_wf_gmx_prep_pull(d, fws_root):
    global project_id, hpc_max_specs, machine
    
    fw_list = []   
    
    files_in = {}
    files_out = { 'data_file': 'in.pdb' }
            
    fts_pull = [ GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'initial_config',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['in.pdb'] ) ]
    
    fw_pull = Firework(fts_pull,
        name = ', '.join(('fetch', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            '_files_out': files_out, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'fetch',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_pull)
    
    return fw_list, fw_pull

def sub_wf_gmx_prep(d, fws_root):    
    global project_id, hpc_max_specs, machine
    # TODO: instead of global variables, use class
    
    fw_list = []   
    
### PDB chain

    files_in =  {'data_file': 'in.pdb' }
    files_out = {'data_file': 'out.pdb'}
    
    fts_pdb_chain = CmdTask(
        cmd='pdb_chain',
        opt=['< in.pdb > out.pdb'],
        store_stdout = False,
        store_stderr = False,
        use_shell    = True,
        fizzle_bad_rc= True)
  
    fw_pdb_chain = Firework(fts_pdb_chain,
        name = ', '.join(('pdb_chain', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'pdb_chain',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_pdb_chain)
    
### PDB tidy
    files_in =  {'data_file': 'in.pdb' }
    files_out = {'data_file': 'out.pdb'}
    
    fts_pdb_tidy = CmdTask(
        cmd='pdb_tidy',
        opt=['< in.pdb > out.pdb'],
        store_stdout = False,
        store_stderr = False,
        use_shell    = True,
        fizzle_bad_rc= True)
  
    fw_pdb_tidy = Firework(fts_pdb_tidy,
        name = ', '.join(('pdb_tidy', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'pdb_tidy',
                 **d
            }
        },
        parents = [ fw_pdb_chain ] )
    
    fw_list.append(fw_pdb_tidy)
    
### GMX pdb2gro
    
    files_in =  {'data_file': 'in.pdb' }
    files_out = {
        'data_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp'}
    
    fts_gmx_pdb2gro = [ CmdTask(
        cmd='gmx',
        opt=['pdb2gmx',
             '-f', 'in.pdb',
             '-o', 'default.gro',
             '-p', 'default.top',
             '-i', 'default.posre.itp', 
             '-ff', 'charmm36',
             '-water' , 'tip3p'],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True) ]
  
    fw_gmx_pdb2gro = Firework(fts_gmx_pdb2gro,
        name = ', '.join(('gmx_pdb2gro', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_pdb2gro',
                 **d
            }
        },
        parents = [ fw_pdb_tidy ] )
    
    fw_list.append(fw_gmx_pdb2gro)
    
    
### GMX editconf
    files_in = {
        'data_file': 'in.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp'}
    files_out = {
        'data_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp'}
    
    fts_gmx_editconf = [ CmdTask(
        cmd='gmx',
        opt=['editconf',
             '-f', 'in.gro',
             '-o', 'default.gro',
             '-d', 2.0, # distance between content and box boundary in nm
             '-bt', 'cubic', # box type
          ],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True) ]
  
    fw_gmx_editconf = Firework(fts_gmx_editconf,
        name = ', '.join(('gmx_editconf', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_editconf',
                 **d
            }
        },
        parents = [ fw_gmx_pdb2gro ] )
    
    fw_list.append(fw_gmx_editconf)
    
    return fw_list, fw_gmx_editconf

def sub_wf_gmx_prep_push(d, fws_root):
    global project_id, hpc_max_specs, machine
    fw_list = []
    files_in = {
        'data_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp' }
    
    fts_push = [ 
        AddFilesTask( {
            'compress': True ,
            'paths': "default.gro",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'initial_config_gro',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.top",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'initial_config_top',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.posre.itp",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'initial_config_posre_itp',
                 **d } 
        } ) ]
        
               
    
    fw_push = Firework(fts_push,
        name = ', '.join(('push', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'push',
                 **d
            }
        },
        parents = fws_root )
        
    fw_list.append(fw_push)
    
    return fw_list, fw_push


# #### Sub-WF: GMX EM

# In[221]:


def sub_wf_gmx_em_pull(d, fws_root):
    global project_id, source_project_id, hpc_max_specs, machine
    
    fw_list = []   

    files_in = {}
    files_out = {
        'data_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp',
    }
            
    fts_fetch = [ 
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'initial_config_gro',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.gro'] ),
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'initial_config_top',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.top'] ), 
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'initial_config_posre_itp',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.posre.itp'] ) ]
    
    fw_fetch = Firework(fts_fetch,
        name = ', '.join(('fetch', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            '_files_out': files_out, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'fetch',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_fetch)
    
    return fw_list, fw_fetch

def sub_wf_gmx_em(d, fws_root):
    global project_id, hpc_max_specs, machine
    
    fw_list = []
### GMX grompp
    files_in = {
        'input_file':      'default.mdp',
        'data_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp'}
    files_out = {
        'input_file': 'default.tpr',
        'parameter_file': 'mdout.mdp' }
    
    fts_gmx_grompp = [ CmdTask(
        cmd='gmx',
        opt=['grompp',
             '-f', 'default.mdp',
             '-c', 'default.gro',
             '-r', 'default.gro',
             '-o', 'default.tpr',
             '-p', 'default.top',
          ],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True) ]
  
    fw_gmx_grompp = Firework(fts_gmx_grompp,
        name = ', '.join(('gmx_grompp_em', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_grompp_em',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_gmx_grompp)
    
### GMX mdrun

    files_in = {'input_file':   'em.tpr'}
    files_out = {
        'log_file':        'em.log',
        'energy_file':     'em.edr',
        'trajectory_file': 'em.trr',
        'data_file':    'em.gro' }
    
    fts_gmx_mdrun = [ CmdTask(
        cmd='gmx',
        opt=[' mdrun',
             '-deffnm', 'em', '-v' ],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True) ]
    
    fw_gmx_mdrun = Firework(fts_gmx_mdrun,
        name = ', '.join(('gmx_mdrun_em', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_queue_category'],
            '_queueadapter': {
                'queue':           hpc_max_specs[machine]['queue'],
                'walltime' :       hpc_max_specs[machine]['walltime'],
                'ntasks':          96,
            },
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_mdrun_em',
                 **d
            }
        },
        parents = [ fw_gmx_grompp ] )
    
    fw_list.append(fw_gmx_mdrun)
    
    return fw_list, fw_gmx_mdrun

def sub_wf_gmx_em_push(d, fws_root):
    global project_id, hpc_max_specs, machine

    fw_list = []
    files_in = {
        'log_file':        'em.log',
        'energy_file':     'em.edr',
        'trajectory_file': 'em.trr',
        'data_file':    'em.gro' }
    files_out = {}
    
    fts_push = [ 
        AddFilesTask( {
            'compress': True ,
            'paths': "em.log",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'em_log',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "em.edr",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'em_edr',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "em.trr",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'em_trr',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "em.gro",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'em_gro',
                 **d } 
        } ) ]
    
    fw_push = Firework(fts_push,
        name = ', '.join(('push', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'push',
                 **d
            }
        },
        parents = fws_root )
        
    fw_list.append(fw_push)

    return fw_list, fw_push


# #### Sub-WF: pulling preparations

# In[235]:


def sub_wf_pull_prep_pull(d, fws_root):
    global project_id, source_project_id, hpc_max_specs, machine
    
    fw_list = []   

    files_in = {}
    files_out = {
        'data_file': 'default.gro',
    }
            
    fts_pull = [ 
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'initial_config_gro',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.gro'] ) ]
    
    fw_pull = Firework(fts_pull,
        name = ', '.join(('fetch', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            '_files_out': files_out, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'fetch',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_pull)
    
    return fw_list, fw_pull

def sub_wf_pull_prep(d, fws_root):    
    global project_id,         surfactant, counterion, substrate, nsubstrate,         tail_atom_name,         fw_name_template, hpc_max_specs, machine
    
    fw_list = []

### System topolog template
    files_in = { 'template_file': 'sys.top.template' }
    files_out = { 'topology_file': 'sys.top' }
        
    # Jinja2 context:
    template_context = {
        'system_name':   '{nsurfactant:d}_{surfactant:s}_{ncounterion:d}_{counterion:s}_{nsubstrate:d}_{substrate:s}'.format(
            project_id=project_id, 
            nsurfactant=d["nmolecules"], surfactant=surfactant, 
            ncounterion=d["nmolecules"], counterion=counterion,
            nsubstrate=nsubstrate, substrate=substrate),
        'header':        '{project_id:s}: {nsurfactant:d} {surfactant:s} and {ncounterion:d} {counterion:s} around {nsubstrate:d}_{substrate:s} AFM probe model'.format(
            project_id=project_id, 
            nsurfactant=d["nmolecules"], surfactant=surfactant, 
            ncounterion=d["nmolecules"], counterion=counterion,
            nsubstrate=nsubstrate, substrate=substrate),
        'nsurfactant': d["nmolecules"],
        'surfactant':  surfactant,
        'ncounterion': d["nmolecules"],
        'counterion':  counterion,
        'nsubstrate':  nsubstrate,
        'substrate':   substrate,
    }
    
    fts_template = [ TemplateWriterTask( {
        'context': template_context,
        'template_file': 'sys.top.template',
        'template_dir': '.',
        'output_file': 'sys.top'} ) ]
    
    
    fw_template = Firework(fts_template,
        name = ', '.join(('template', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in,
            '_files_out': files_out, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'fill_template',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_template)

### Index file

    files_in = {'data_file': 'default.gro'}
    files_out = {'index_file': 'default.ndx'}
    
    fts_gmx_make_ndx = [ 
        CmdTask(
            cmd='gmx',
            opt=['make_ndx',
                 '-f', 'default.gro',
                 '-o', 'default.ndx',
              ],
            env = 'python',
            stdin_key    = 'stdin',
            stderr_file  = 'std.err',
            stdout_file  = 'std.out',
            store_stdout = True,
            store_stderr = True,
            fizzle_bad_rc= True) ]
    
    fw_gmx_make_ndx = Firework(fts_gmx_make_ndx,
        name = ', '.join(('gmx_make_ndx', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'stdin':    'q\n',
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_make_ndx',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_gmx_make_ndx)
    
### pulling groups

    files_in = {
        'data_file':      'default.gro',
        'topology_file':  'default.top',
        'index_file':     'in.ndx',
        'parameter_file': 'in.mdp',
    }
    files_out = {
        'data_file':      'default.gro', # pass through unmodified
        'topology_file':  'default.top', # pass unmodified
        'index_file':     'out.ndx',
        'input_file':     'out.mdp',
    }
    
    
    fts_make_pull_groups = [ CmdTask(
        cmd='gmx_tools',
        opt=['--verbose', '--log', 'default.log',
            'make','pull_groups',
            '--topology-file', 'default.top',
            '--coordinates-file', 'default.gro',
            '--residue-name', surfactant,
            '--atom-name', tail_atom_name,
            '--reference-group-name', 'Substrate',
             '-k', 1000, 
             '--rate', 0.1, '--',
            'in.ndx', 'out.ndx', 'in.mdp', 'out.mdp'],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        env = 'python',
        store_stdout = True,
        store_stderr = True,
        fizzle_bad_rc= True) ]
  
    fw_make_pull_groups = Firework(fts_make_pull_groups,
        name = ', '.join(('gmx_tools_make_pull_groups', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_tools_make_pull_groups',
                 **d
            }
        },
        parents = [ *fws_root, fw_template, fw_gmx_make_ndx] )
    
    fw_list.append(fw_make_pull_groups)

    return fw_list, fw_make_pull_groups



def sub_wf_pull_prep_push(d, fws_root):
    global project_id, hpc_max_specs, machine

    fw_list = []   

    files_in = {
        'topology_file':  'default.top',
        'index_file':     'default.ndx',
        'input_file':     'default.mdp',
    }
    
    fts_push = [ 
        AddFilesTask( {
            'compress': True ,
            'paths': "default.top",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'top_pull',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.ndx",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'ndx_pull',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.mdp",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'mdp_pull',
                 **d } 
        } ),
    ]
    
    fw_push = Firework(fts_push,
        name = ', '.join(('push', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'push',
                 **d
            }
        },
        parents = fws_root )
        
    fw_list.append(fw_push)
    
    return fw_list, fw_push


# #### Sub-WF: GMX pull

# In[271]:


def sub_wf_gmx_pull_pull(d, fws_root):
    global project_id, source_project_id, hpc_max_specs, machine
    
    fw_list = []   

    files_in = {}
    files_out = {
        'data_file':       'default.gro',
        'topology_file':   'default.top',
        'input_file':      'default.mdp',
        'indef_file':      'default.ndx',
    }
            
    fts_fetch = [ 
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id, # earlier
                'metadata->type':       'initial_config_gro',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.gro'] ),
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'top_pull',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.top'] ), 
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'ndx_pull',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.ndx'] ), 
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'mdp_pull',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.mdp'] ) ]
    
    fw_fetch = Firework(fts_fetch,
        name = ', '.join(('pull', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            '_files_out': files_out, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'pull',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_fetch)
    
    return fw_list, fw_fetch

def sub_wf_gmx_pull(d, fws_root):
    global project_id, hpc_max_specs, machine
    
    fw_list = []
### GMX grompp
    files_in = {
        'input_file':      'default.mdp',
        'index_file':      'default.ndx',
        'data_file':       'default.gro',
        'topology_file':   'default.top'}
    files_out = {
        'input_file':      'default.tpr',
        'parameter_file':  'mdout.mdp',
    }
    
    # gmx grompp -f pull.mdp -n pull_groups.ndx -c em.gro -r em.gro -o pull.tpr -p sys.top
    fts_gmx_grompp = [ CmdTask(
        cmd='gmx',
        opt=['grompp',
             '-f', 'default.mdp',  # parameter file
             '-n', 'default.ndx',  # index file
             '-c', 'default.gro',  # coordinates file
             '-r', 'default.gro',  # restraint positions
             '-p', 'default.top',  # topology file
             '-o', 'default.tpr',  # compiled output
          ],
        env          = 'python',
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        fizzle_bad_rc= True) ]
  
    fw_gmx_grompp = Firework(fts_gmx_grompp,
        name = ', '.join(('gmx_grompp_pull', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_grompp_pull',
                 **d
            }
        },
        parents = fws_root )
    
    fw_list.append(fw_gmx_grompp)
    
### GMX mdrun

    files_in = {'input_file': 'default.tpr'}
    files_out = {
        'log_file':        'default.log',
        'energy_file':     'default.edr',
        'trajectory_file': 'default.trr',
        'compressed_trajectory_file': 'default.xtc',
        'data_file':       'default.gro',
        'pullf_file':      'default_pullf.xvg',
        'pullx_file':      'default_pullx.xvg',}
    
    fts_gmx_mdrun = [ CmdTask(
        cmd='gmx',
        opt=['mdrun',
             '-deffnm', 'default', '-v' ],
        env='python',
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        fizzle_bad_rc= True) ]
    
    fw_gmx_mdrun = Firework(fts_gmx_mdrun,
        name = ', '.join(('gmx_mdrun_pull', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_queue_category'],
            '_queueadapter': {
                'queue':           hpc_max_specs[machine]['queue'],
                'walltime' :       hpc_max_specs[machine]['walltime'],
                'ntasks':          hpc_max_specs[machine]['physical_cores_per_node'],
                'ntasks_per_node': hpc_max_specs[machine]['physical_cores_per_node'],
                # JUWELS GROMACS
                # module("load","Stages/2019a","Intel/2019.3.199-GCC-8.3.0","IntelMPI/2019.3.199")
                # module("load","GROMACS/2019.3","GROMACS-Top/2019.3")
                # fails with segmentation fault when using SMT (96 logical cores)
            },
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_mdrun_pull',
                 **d
            }
        },
        parents = [ fw_gmx_grompp ] )
    
    fw_list.append(fw_gmx_mdrun)
    
    return fw_list, fw_gmx_mdrun

def sub_wf_gmx_pull_push(d, fws_root):
    global project_id, hpc_max_specs, machine

    fw_list = []
    files_in = {
        'log_file':        'default.log',
        'energy_file':     'default.edr',
        'trajectory_file': 'default.trr',
        'compressed_trajectory_file': 'default.xtc',
        'data_file':       'default.gro',
        'pullf_file':      'default_pullf.xvg',
        'pullx_file':      'default_pullx.xvg',}
    
    files_out = {}
    
    fts_push = [ 
        AddFilesTask( {
            'compress': True ,
            'paths': "default.log",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'pull_log',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.edr",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'pull_edr',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.trr",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'pull_trr',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.xtc",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'pull_xtc',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.gro",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'pull_gro',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default_pullf.xvg",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'pullf_xvg',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default_pullx.xvg",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'pullx_xvg',
                 **d }
        } ) ]
    
    fw_push = Firework(fts_push,
        name = ', '.join(('push', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'push',
                 **d
            }
        },
        parents = fws_root )
        
    fw_list.append(fw_push)

    return fw_list, fw_push


# ## Conversion from LAMMPS data format to PDB

# The vollowing bash / tcl snippet converts a LAMMPS data file to PDB, assigning the desired names as mapped in a yaml file
# ```bash
# #!/bin/bash
# # echo "package require jlhvmd; jlh lmp2pdb indenter.lammps indenter.pdb" | vmd -eofexit
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

# ## Inspect AFM tip model

# ### Read pdb

# In[17]:


ls $prefix


# In[18]:


infile = os.path.join(prefix,'dat','indenter','AU_111_r_25.pdb')


# In[19]:


atoms = ase.io.read(infile,format='proteindatabank')


# In[20]:


atoms


# ### Display with ASE view

# In[21]:


v = view(atoms,viewer='ngl')
v.view._remote_call("setSize", target="Widget", args=["400px", "400px"])
v.view.center()
v.view.background='#ffc'
v


# ### Get the bounding sphere around point set

# In[25]:


S = atoms.get_positions()
C, R_sq = miniball.get_bounding_ball(S)
R = np.sqrt(R_sq)
del S

xmin = atoms.get_positions().min(axis=0)
xmax = atoms.get_positions().max(axis=0)


# In[26]:


C # sphere center


# In[27]:


R # sphere radius


# In[28]:


xmin


# 
# ### Derive surfactant numbers from sphere dimensions

# In[29]:


A_Ang = 4*np.pi*R**2 # area in Ansgtrom
A_nm = A_Ang / 10**2
n_per_nm_sq = np.array([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
N = np.round(A_nm*n_per_nm_sq).astype(int)


# In[30]:


A_nm


# In[31]:


N # molecule numbers corresponding to surface concentrations


# ### Plot 2d projections of point set and bounding sphere

# In[32]:


# plot side views with sphere projections
plot_side_views_with_spheres(atoms,C,R)


# ### Plot 3d point set and bounding sphere

# In[44]:


# bounding sphere surface
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
us = np.array([
    np.outer(np.cos(u), np.sin(v)),
    np.outer(np.sin(u), np.sin(v)), 
    np.outer(np.ones(np.size(u)), np.cos(v))])
bs = C + R*us.T


# In[45]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
x = atoms.get_positions()
ax.scatter(x[:,0], x[:,1],x[:,2], c='y', marker='o')
ax.plot_surface(*bs.T, color='b',alpha=0.1) # bounding sphere
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# ## Measure surfactant molecule

# In[46]:


tol = 2 # Ang


# ### Read single surfactant molecule PDB with ParmEd

# Utilize parmed to read pbd files ASE has difficulties to decipher.

# In[168]:


infile = os.path.join(prefix,'1_SDS.pdb')


# In[169]:


surfactant_pmd = pmd.load_file(infile)


# In[170]:


surfactant_pmd.atoms[-1].atomic_number


# ### Convert ParmEd structure to ASE atoms

# In[171]:


surfactant_ase = ase.Atoms(
    numbers=[1 if a.atomic_number == 0 else a.atomic_number for a in surfactant_pmd.atoms],
    positions=surfactant_pmd.get_coordinates(0))


# ### Get bounding sphere of single surfactant molecule

# In[172]:


C_surfactant, R_sq_surfactant = miniball.get_bounding_ball(surfactant_ase.get_positions())


# In[173]:


C_surfactant


# In[174]:


R_surfactant = np.sqrt(R_sq_surfactant)


# In[175]:


R_surfactant


# In[176]:


C_surfactant


# In[177]:


surfactant_ase[:5][1]


# ### Estimate constraint sphere radii

# In[178]:


R_OSL = np.linalg.norm(C_surfactant - surfactant_ase[1].position)


# In[179]:


R_OSL


# In[180]:


d_head = R_surfactant - R_OSL # roughly: diameter of head group


# In[181]:


R_inner = R + tol # place surfactant molecules outside of this sphere


# In[182]:


R_inner_constraint = R + tol + d_head # place surfactant tail hydrocarbon within this sphere


# In[183]:


R_outer_constraint = R + 2*R_surfactant + tol # place head group sulfur outside this sphere


# In[184]:


R_outer = R + 2*R_surfactant + 2*tol # place suractant molecules within this sphere


# In[185]:


rr = [R,R_inner,R_inner_constraint,R_outer_constraint,R_outer]


# In[186]:


cc = [C]*5


# ### Show 2d projections of geometrical constraints around AFM tip model

# In[187]:


plot_side_views_with_spheres(atoms,cc,rr,figsize=(20,8))
plt.show()


# ## Packing the surfactant film

# In[188]:


infile_prefix = os.path.join(prefix,'packmol_infiles')


# ### Identify placeholders in jinja2 template

# The template looks like this:

# In[189]:


with open(os.path.join(infile_prefix,'surfactants_on_sphere.inp'),'r') as f:
    print(f.read())


# In[190]:


# get all placholders in template
template_file = os.path.join(infile_prefix,'surfactants_on_sphere.inp')


# In[191]:


v = find_undeclared_variables(template_file)


# In[192]:


v # we want to fill in these placeholder variables


# ### System and constraint parameters

# In[193]:


surfactant = 'SDS'
counterion = 'NA'
tolerance = 2 # Ang
sfN = 200


# In[194]:


l_surfactant = 2*R_surfactant


# In[195]:


# head atom to be geometrically constrained
surfactant_head_bool_ndx = np.array([ a.name == 'S' for a in surfactant_pmd.atoms ],dtype=bool)


# In[196]:


# tail atom to be geometrically constrained
surfactant_tail_bool_ndx = np.array([ a.name == 'C12' for a in surfactant_pmd.atoms ],dtype=bool)


# In[197]:


head_atom_number = surfactant_head_ndx = np.argwhere(surfactant_head_bool_ndx)[0,0]


# In[198]:


tail_atom_number = surfactant_tail_ndx = np.argwhere(surfactant_tail_bool_ndx)[0,0]


# In[199]:


# settings can be overridden
packmol_script_context = {
    'header':        '20191113 TEST PACKING',
    'system_name':   '200_SDS_on_50_Ang_AFM_tip_model',
    'tolerance':     tolerance,
    'write_restart': True,
    
    'static_components': [
        {
            'name': 'indenter_reres'
        }
    ]
}

# use pack_sphere function at the notebook's head to generate template context
packmol_script_context.update(
    pack_sphere(
        C,R_inner_constraint,R_outer_constraint, sfN, 
        tail_atom_number+1, head_atom_number+1, surfactant, counterion, tolerance))


# In[200]:


packmol_script_context # context generated from system and constraint settings


# ### Fill a packmol input script template with jinja2

# In[201]:


env = jinja2.Environment()


# In[202]:


template = jinja2.Template(open(template_file).read())


# In[203]:


rendered = template.render(**packmol_script_context)


# In[204]:


rendered_file = os.path.join(prefix,'rendered.inp')


# In[205]:


with open(rendered_file,'w') as f:
    f.write(rendered)


# That's the rendered packmol input file:

# In[206]:


print(rendered)


# ### Fail running packmol once

# In[207]:


packmol = subprocess.Popen(['packmol'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=prefix, encoding='utf-8')


# In[208]:


outs, errs = packmol.communicate(input=rendered)


# In[209]:


print(errs) # error with input from PIPE


# ### Read packmol input from file to avoid obscure Fortran error

# In[210]:


packmol = subprocess.Popen(['packmol'],
        stdin=open(rendered_file,'r'),stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=prefix, encoding='utf-8')


# In[211]:


outs, errs = packmol.communicate(input=rendered)


# In[212]:


print(outs)


# In[213]:


with open('packmol.log','w') as f:
    f.write(outs)


# ### Inspect packed systems

# In[214]:


packmol_pdb = '200_SDS_on_50_Ang_AFM_tip_model_packmol.pdb'


# In[215]:


infile = os.path.join(prefix, packmol_pdb)


# In[216]:


surfactant_shell_pmd = pmd.load_file(infile)


# In[217]:


# with ParmEd and nglview we get automatic bond guessing
pmd_view = nglview.show_parmed(surfactant_shell_pmd)
pmd_view.clear_representations()
pmd_view.background = 'white'
pmd_view.add_representation('ball+stick')
pmd_view


# In[218]:


surfactant_shell_ase = ase.Atoms(
    numbers=[1 if a.atomic_number == 0 else a.atomic_number for a in surfactant_shell_pmd.atoms],
    positions=surfactant_shell_pmd.get_coordinates(0))


# In[219]:


# with ASE, we get no bonds at all
ase_view = nglview.show_ase(surfactant_shell_ase)
ase_view.clear_representations()
ase_view.background = 'white'
ase_view.add_representation('ball+stick')
ase_view


# Get bounding sphere again and display AFM tip bounding spphere as well as surfactant layer bounding sphere

# In[220]:


C_shell, R_sq_shell = miniball.get_bounding_ball(surfactant_shell_ase.get_positions())


# In[221]:


C_shell


# In[222]:


R_shell = np.sqrt(R_sq_shell)


# In[223]:


R_shell


# In[224]:


plot_side_views_with_spheres(surfactant_shell_ase,[C,C_shell],[R,R_shell])


# In[342]:


surfactant_shell_pmd


# ### Batch processing: Parametric jobs

# #### Provide PACKMOL template 

# In[54]:


project_id = 'juwels-packmol-2020-03-09'


# In[17]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[18]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[160]:


infiles = sorted(glob.glob(os.path.join(infile_prefix,'*.inp')))

files = { os.path.basename(f): f for f in infiles }

# metadata common to all these files 
metadata = {
    'project': project_id,
    'type': 'template'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )


# In[224]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.type': 'template'
}

# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[171]:


print(identifier)


# In[172]:


# on a lower level, each object has a unique "GridFS id":
pprint(fp_files) # underlying GridFS id and readable identifiers


# #### Provide data files

# In[272]:


data_prefix = os.path.join(prefix,'packmol_datafiles')


# In[273]:


datafiles = sorted(glob.glob(os.path.join(data_prefix,'*')))

files = { os.path.basename(f): f for f in datafiles }

# metadata common to all these files 
metadata = {
    'project': project_id,
    'type': 'data'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )


# In[274]:


fp_files


# #### Span parameter sets

# In[485]:


machine = 'juwels_devel'


# In[480]:


parametric_dimension_labels = ['nmolecules']


# In[481]:


parametric_dimensions = [ {
    'nmolecules': N } ]


# In[458]:


# for testing
parametric_dimensions = [ {
    'nmolecules': [N[0]] } ]


# In[482]:


parameter_sets = list( 
    itertools.chain(*[ 
            itertools.product(*list(
                    p.values())) for p in parametric_dimensions ]) )

parameter_dict_sets = [ dict(zip(parametric_dimension_labels,s)) for s in parameter_sets ]


# In[486]:


wf_name = 'PACKMOL {machine:}, {id:}'.format(machine=machine,id=project_id)

fw_name_template = 'nmolecules: {nmolecules:d}'

fw_list = []

fts = [ 
        GetFilesByQueryTask(
            query = {
                'metadata->project': project_id,
                'metadata->name':    'surfactants_on_sphere.inp'
            },
            limit = 1,
            new_file_names = ['input.template'] ),
        GetFilesByQueryTask(
            query = {
                'metadata->project': project_id,
                'metadata->name':    'indenter_reres.pdb'
            },
            limit = 1,
            new_file_names = ['indenter.pdb'] ),
        GetFilesByQueryTask(
            query = {
                'metadata->project': project_id,
                'metadata->name':    '1_SDS.pdb'
            },
            limit = 1,
            new_file_names = ['1_SDS.pdb'] ),
        GetFilesByQueryTask(
            query = {
                'metadata->project': project_id,
                'metadata->name':    '1_NA.pdb'
            },
            limit = 1,
            new_file_names = ['1_NA.pdb'] )
        ]

files_out = {
    'input_file': 'input.template',
    'indenter_file': 'indenter.pdb',
    'surfatcant_file': '1_SDS.pdb',
    'counterion_file': '1_NA.pdb'}

fw_root = Firework(fts,
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        '_files_out': files_out, 
        'metadata': {
          'project': project_id,
          'datetime': str(datetime.datetime.now()),
          'step':    'input_file_query'
        }
    }
)

fw_list.append(fw_root)

## Parametric sweep

for d in parameter_dict_sets:        
    
### Template
    
    files_in = {'input_file': 'input.template' }
    files_out = { 'input_file': 'input.inp' }
    
    # exports = std_exports[machine].copy()
        
    # Jinja2 context:
    packmol_script_context = {
        'header':        '{:s} packing SDS around AFM probe model'.format(project_id),
        'system_name':   '{:d}_SDS_on_50_Ang_AFM_tip_model'.format(n),
        'tolerance':     tolerance,
        'write_restart': True,

        'static_components': [
            {
                'name': 'indenter'
            }
        ]
    }

    # use pack_sphere function at the notebook's head to generate template context
    packmol_script_context.update(
        pack_sphere(
            C,R_inner_constraint,R_outer_constraint, d["nmolecules"], 
            tail_atom_number+1, head_atom_number+1, surfactant, counterion, tolerance))
    
    ft_template = TemplateWriterTask( {
        'context': packmol_script_context,
        'template_file': 'input.template',
        'template_dir': '.',
        'output_file': 'input.inp'} )
    
    
    fw_template = Firework([ft_template],
        name = ', '.join(('template', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in,
            '_files_out': files_out, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'fill_template',
                 **d
            }
        },
        parents = [ fw_root ] )
    
    fw_list.append(fw_template)

### PACKMOL

    files_in = {
        'input_file': 'input.inp',
        'indenter_file': 'indenter.pdb',
        'surfatcant_file': '1_SDS.pdb',
        'counterion_file': '1_NA.pdb' }
    files_out = {
        'data_file': '*_packmol.pdb'}
    
    ft_pack = CmdTask(
        cmd='packmol',
        opt=['< input.inp'],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True)
  
    fw_pack = Firework([ft_pack],
        name = ', '.join(('packmol', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_queue_category'],
            '_queueadapter': {
                'queue':           hpc_max_specs[machine]['queue'],
                'walltime' :       hpc_max_specs[machine]['walltime'],
                'ntasks':          1,
            },
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'packmol',
                 **d
            }
        },
        parents = [ fw_root, fw_template] )
    
    fw_list.append(fw_pack)

### Store

    files_in = {'data_file': 'packed.pdb' }
    
    ft_transfer = AddFilesTask( {
        'compress': True ,
        'paths': "packed.pdb",
        'metadata': {
            'project': project_id,
            'datetime': str(datetime.datetime.now()),
            'type':    'initial_config',
             **d } 
        } )
    
    fw_transfer = Firework([ft_transfer],
        name = ', '.join(('transfer', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'transfer',
                 **d
            }
        },
        parents = [ fw_pack ] )
        
    fw_list.append(fw_transfer)

wf = Workflow(fw_list,
    name = wf_name,
    metadata = {
        'project': project_id,
        'datetime': str(datetime.datetime.now()),
        'type':    'packing'
    })


# In[488]:


wf.to_file('packing.json')


# In[489]:


lp.add_wf(wf)


# #### Inspect sweep results

# In[31]:


query = { 
    "metadata.project": project_id,
}
fp.filepad.count_documents(query)


# In[32]:


query = { 
    "metadata.project": project_id,
    "metadata.type":    'initial_config',
}
fp.filepad.count_documents(query)


# In[33]:


parameter_names = ['nmolecules']


# In[34]:


surfactant_shell_pmd_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(content)
        surfactant_shell_pmd_list.append(pmd.load_file(tmp.name))
        
    print('.',end='')
print('')
    


# In[80]:


system_selection = surfactant_shell_pmd_list
nx = 3
ny = len(system_selection)
view_labels = ['xy','xz','yz']

molecule_counter = lambda s: np.count_nonzero([r.name == 'SDS' for r in s.residues])

label_generator = lambda i,j: '{:d} SDS, {:s} projection'.format(
    molecule_counter(system_selection[i]),view_labels[j])

figsize = (4*nx,4*ny)
fig, axes = plt.subplots(ny,3,figsize=figsize)

for i,system in enumerate(system_selection):
    system_ase = ase.Atoms(
        numbers=[1 if a.atomic_number == 0 else a.atomic_number for a in system.atoms],
        positions=system.get_coordinates(0))
    
    C_shell, R_sq_shell = miniball.get_bounding_ball(system_ase.get_positions())
    R_shell = np.sqrt(R_sq_shell)
    plot_side_views_with_spheres(
        system_ase,[C,C_shell],[R,R_shell],fig=fig,ax=axes[i,:])
    
    for j, ax in enumerate(axes[i,:]):
        ax.set_title(label_generator(i,j))
        
    del system_ase
    gc.collect()


# In[108]:


del fig
del axes


# ## Prepare a Gromacs-processible system

# In[740]:


gromacs.config.logfilename


# In[741]:


gromacs.environment.flags


# In[742]:


# if true, then stdout and stderr are returned as strings by gromacs wrapper commands
gromacs.environment.flags['capture_output'] = False


# In[84]:


print(gromacs.release())


# In[85]:


prefix


# In[539]:


system = '200_SDS_on_50_Ang_AFM_tip_model'
pdb = system + '.pdb'
gro = system + '.gro'
top = system + '.top'
posre = system + '.posre.itp'


# ### Tidy up packmol's non-standard pdb

# In[540]:


# Remove any chain ID from pdb and tidy up
pdb_chain = subprocess.Popen(['pdb_chain',],
        stdin=open(packmol_pdb,'r'),stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=prefix, encoding='utf-8')
pdb_tidy = subprocess.Popen(['pdb_tidy',],
        stdin=pdb_chain.stdout,stdout=open(pdb,'w'), stderr=subprocess.PIPE,
        cwd=prefix, encoding='utf-8')


# ### Generate Gromacs .gro and .top

# In[541]:


rc,out,err=gromacs.pdb2gmx(
    f=pdb,o=gro,p=top,i=posre,ff='charmm36',water='tip3p',
    stdout=False,stderr=False)


# In[542]:


print(out)


# ### Set simulation box size around system

# In[543]:


gro_boxed = system + '_boxed.gro'


# In[544]:


rc,out,err=gromacs.editconf(
    f=gro,o=gro_boxed,d=2.0,bt='cubic',
    stdout=False,stderr=False)


# In[545]:


print(out)


# ### Batch processing

# #### Buld workflow

# In[43]:


machine = 'juwels_devel'


# In[44]:


parametric_dimension_labels = ['nmolecules']


# In[45]:


parametric_dimensions = [ {
    'nmolecules': N } ]


# In[263]:


# for testing
parametric_dimensions = [ {
    'nmolecules': [N[0]] } ]


# In[264]:


parameter_sets = list( 
    itertools.chain(*[ 
            itertools.product(*list(
                    p.values())) for p in parametric_dimensions ]) )

parameter_dict_sets = [ dict(zip(parametric_dimension_labels,s)) for s in parameter_sets ]


# In[265]:


source_project_id = 'juwels-packmol-2020-03-09'
project_id = 'juwels-gromacs-prep-2020-03-11'


# In[266]:


wf_name = 'GROMACS preparations {machine:}, {id:}'.format(machine=machine,id=project_id)

fw_name_template = 'nmolecules: {nmolecules:d}'

fw_list = []

fts = [
    CmdTask(
        cmd='echo',
        opt=['"Dummy root"'],
        store_stdout = False,
        store_stderr = False,
        use_shell    = True,
        fizzle_bad_rc= True) ]
  
files_out = []
    
fw_root = Firework(fts,
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        '_files_out': files_out, 
        'metadata': {
          'project': project_id,
          'datetime': str(datetime.datetime.now()),
          'step':    'dummy_root'
        }
    }
)

fw_list.append(fw_root)

## Parametric sweep

for d in parameter_dict_sets:        
    
### File retrieval
    
    #files_in = {'input_file': 'input.template' }
    files_in = {}
    files_out = { 'data_file': 'in.pdb' }
    
    # exports = std_exports[machine].copy()
        
    fts_fetch = [ GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'initial_config',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['in.pdb'] ) ]
    
    fw_fetch = Firework(fts_fetch,
        name = ', '.join(('fetch', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            '_files_out': files_out, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'fetch',
                 **d
            }
        },
        parents = [ fw_root ] )
    
    fw_list.append(fw_fetch)
    
### PDB chain

    files_in =  {'data_file': 'in.pdb' }
    files_out = {'data_file': 'out.pdb'}
    
    fts_pdb_chain = CmdTask(
        cmd='pdb_chain',
        opt=['< in.pdb > out.pdb'],
        store_stdout = False,
        store_stderr = False,
        use_shell    = True,
        fizzle_bad_rc= True)
  
    fw_pdb_chain = Firework(fts_pdb_chain,
        name = ', '.join(('pdb_chain', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'pdb_chain',
                 **d
            }
        },
        parents = [ fw_fetch ] )
    
    fw_list.append(fw_pdb_chain)
    
### PDB tidy
    files_in =  {'data_file': 'in.pdb' }
    files_out = {'data_file': 'out.pdb'}
    
    fts_pdb_tidy = CmdTask(
        cmd='pdb_tidy',
        opt=['< in.pdb > out.pdb'],
        store_stdout = False,
        store_stderr = False,
        use_shell    = True,
        fizzle_bad_rc= True)
  
    fw_pdb_tidy = Firework(fts_pdb_tidy,
        name = ', '.join(('pdb_tidy', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'pdb_tidy',
                 **d
            }
        },
        parents = [ fw_pdb_chain ] )
    
    fw_list.append(fw_pdb_tidy)
    
### GMX pdb2gro
    
    files_in =  {'data_file': 'in.pdb' }
    files_out = {
        'coordinate_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp'}
    
    fts_gmx_pdb2gro = [ CmdTask(
        cmd='gmx',
        opt=['pdb2gmx',
             '-f', 'in.pdb',
             '-o', 'default.gro',
             '-p', 'default.top',
             '-i', 'default.posre.itp', 
             '-ff', 'charmm36',
             '-water' , 'tip3p'],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True) ]
  
    fw_gmx_pdb2gro = Firework(fts_gmx_pdb2gro,
        name = ', '.join(('gmx_pdb2gro', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_pdb2gro',
                 **d
            }
        },
        parents = [ fw_pdb_tidy ] )
    
    fw_list.append(fw_gmx_pdb2gro)
    
    
### GMX editconf
    files_in = {
        'coordinate_file': 'in.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp'}
    files_out = {
        'coordinate_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp'}
    
    fts_gmx_editconf = [ CmdTask(
        cmd='gmx',
        opt=['editconf',
             '-f', 'in.gro',
             '-o', 'default.gro',
             '-d', 2.0, # distance between content and box boundary in nm
             '-bt', 'cubic', # box type
          ],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True) ]
  
    fw_gmx_editconf = Firework(fts_gmx_editconf,
        name = ', '.join(('gmx_editconf', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_editconf',
                 **d
            }
        },
        parents = [ fw_gmx_pdb2gro ] )
    
    fw_list.append(fw_gmx_editconf)
    
### Store

    #files_in = {'data_file': 'packed.pdb' }
    files_in = {
        'coordinate_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp' }
    
    fts_push = [ 
        AddFilesTask( {
            'compress': True ,
            'paths': "default.gro",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'initial_config_gro',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.top",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'initial_config_top',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "default.posre.itp",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'initial_config_posre_itp',
                 **d } 
        } ) ]
        
               
    
    fw_push = Firework(fts_push,
        name = ', '.join(('push', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'push',
                 **d
            }
        },
        parents = [ fw_gmx_editconf ] )
        
    fw_list.append(fw_push)

wf = Workflow(fw_list,
    name = wf_name,
    metadata = {
        'project': project_id,
        'datetime': str(datetime.datetime.now()),
        'type':    'gmx_prep'
    })


# In[267]:


wf.as_dict()


# In[268]:


lp.add_wf(wf)


# In[269]:


wf.to_file('wf-{:s}.yaml'.format(project_id))


# #### Inspect sweep results

# In[93]:


query = { 
    "metadata.project": project_id,
}
fp.filepad.count_documents(query)


# In[94]:


query = { 
    "metadata.project": project_id,
    "metadata.type":    'initial_config_gro',
}
fp.filepad.count_documents(query)


# In[95]:


parameter_names = ['nmolecules']


# In[96]:


gro_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    with tempfile.NamedTemporaryFile(suffix='.gro') as tmp:
        tmp.write(content)
        gro_list.append(pmd.load_file(tmp.name))
        
    print('.',end='')
print('')
    


# In[99]:


gro_list


# In[100]:


# with ParmEd and nglview we get automatic bond guessing
pmd_view = nglview.show_parmed(gro_list[0])
pmd_view.clear_representations()
pmd_view.background = 'white'
pmd_view.add_representation('ball+stick')
pmd_view


# ## Energy minimization with restraints

# Just to be safe, relax the system a little with positional constraints applied to all ions.

# ### Compile system

# In[546]:


os.getcwd()


# In[547]:


em_mdp = gromacs.fileformats.MDP('em.mdp.template')
# no change
em_mdp.write('em.mdp')


# In[548]:


gmx_grompp = gromacs.grompp.Popen(
    f='em.mdp',c=gro,r=gro,o='em.tpr',p=top,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)

out = gmx_grompp.stdout.read()
err = gmx_grompp.stderr.read()


# In[549]:


print(err)


# In[551]:


print(out)


# ### Run energy minimization

# In[552]:


gmx_mdrun = gromacs.mdrun.Popen(
    deffnm='em',v=True,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)


# In[553]:


for line in gmx_mdrun.stdout: 
    print(line.decode(), end='')


# In[ ]:


out = gmx_mdrun.stdout.read()
err = gmx_mdrun.stderr.read()


# In[ ]:


print(err)


# In[ ]:


print(out)


# ### Energy minimization analysis

# In[319]:


em_file = 'em.edr'


# In[320]:


em_df = panedr.edr_to_df(em_file)


# In[554]:


em_df.columns


# In[322]:


fig, ax = plt.subplots(3,2,figsize=(10,12))
em_df.plot('Time','Potential',ax=ax[0,0])
em_df.plot('Time','Pressure',ax=ax[0,1])
em_df.plot('Time','Bond',ax=ax[1,0])
em_df.plot('Time','Position Rest.',ax=ax[1,1])
#em_df.plot('Time','COM Pull En.',ax=ax[1,1])
em_df.plot('Time','Coulomb (SR)',ax=ax[2,0])
em_df.plot('Time','Coul. recip.',ax=ax[2,1])


# In[233]:


mda_trr = mda.Universe(gro,'em.trr')

mda_view = nglview.show_mdanalysis(mda_trr)

mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation('ball+stick')
mda_view


# ### Batch processing

# #### Build workflow: em

# In[42]:


machine = 'juwels_devel'


# In[52]:


parametric_dimension_labels = ['nmolecules']


# In[53]:


parametric_dimensions = [ {
    'nmolecules': N } ]


# In[273]:


# for testing
parametric_dimensions = [ {
    'nmolecules': [N[0]] } ]


# In[274]:


parameter_sets = list( 
    itertools.chain(*[ 
            itertools.product(*list(
                    p.values())) for p in parametric_dimensions ]) )

parameter_dict_sets = [ dict(zip(parametric_dimension_labels,s)) for s in parameter_sets ]


# In[275]:


source_project_id = 'juwels-gromacs-prep-2020-03-11'
project_id = 'juwels-gromacs-em-2020-03-12'
infile_prefix = 'gmx_em_infiles'


# In[276]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[277]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[278]:


infiles = sorted(glob.glob(os.path.join(infile_prefix,'*')))

files = { os.path.basename(f): f for f in infiles }

# metadata common to all these files 
metadata = {
    'project': project_id,
    'type': 'input'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )


# In[291]:


wf_name = 'GROMACS energy minimization {machine:}, {id:}'.format(machine=machine,id=project_id)

fw_name_template = 'nmolecules: {nmolecules:d}'

fw_list = []

files_out = { 'input_file': 'default.mdp' }

fts_root = [ 
    GetFilesByQueryTask(
        query = {
            'metadata->project': project_id,
            'metadata->name':    'em.mdp'
        },
        limit = 1,
        new_file_names = ['default.mdp'] ) ]

fw_root = Firework(fts_root,
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        '_files_out': files_out, 
        'metadata': {
          'project': project_id,
          'datetime': str(datetime.datetime.now()),
          'step':    'input_file_query'
        }
    }
)

fw_list.append(fw_root)

## Parametric sweep

for d in parameter_dict_sets:        
    
### File retrieval
    
    #files_in = {'input_file': 'input.template' }
    files_in = {}
    files_out = {
        'coordinate_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp',
    }
            
    fts_fetch = [ 
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'initial_config_gro',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.gro'] ),
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'initial_config_top',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.top'] ), 
        GetFilesByQueryTask(
            query = {
                'metadata->project':    source_project_id,
                'metadata->type':       'initial_config_posre_itp',
                'metadata->nmolecules': d["nmolecules"]
            },
            sort_key = 'metadata.datetime',
            sort_direction = pymongo.DESCENDING,
            limit = 1,
            new_file_names = ['default.posre.itp'] ) ]
    
    fw_fetch = Firework(fts_fetch,
        name = ', '.join(('fetch', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            '_files_out': files_out, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'fetch',
                 **d
            }
        },
        parents = [ fw_root ] )
    
    fw_list.append(fw_fetch)
    
### GMX grompp
    files_in = {
        'input_file':      'default.mdp',
        'coordinate_file': 'default.gro',
        'topology_file':   'default.top',
        'restraint_file':  'default.posre.itp'}
    files_out = {
        'input_file': 'default.tpr',
        'parameter_file': 'mdout.mdp' }
    
    fts_gmx_grompp = [ CmdTask(
        cmd='gmx',
        opt=['grompp',
             '-f', 'default.mdp',
             '-c', 'default.gro',
             '-r', 'default.gro',
             '-o', 'default.tpr',
             '-p', 'default.top',
          ],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True) ]
  
    fw_gmx_grompp = Firework(fts_gmx_grompp,
        name = ', '.join(('gmx_grompp_em', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_grompp_em',
                 **d
            }
        },
        parents = [ fw_fetch, fw_root ] )
    
    fw_list.append(fw_gmx_grompp)
    
### GMX mdrun

    files_in = {'input_file':   'em.tpr'}
    files_out = {
        'energy_file':     'em.edr',
        'trajectory_file': 'em.trr',
        'final_config':    'em.gro' }
    
    fts_gmx_mdrun = [ CmdTask(
        cmd='gmx',
        opt=[' mdrun',
             '-deffnm', 'em', '-v' ],
        stderr_file  = 'std.err',
        stdout_file  = 'std.out',
        store_stdout = True,
        store_stderr = True,
        use_shell    = True,
        fizzle_bad_rc= True) ]
    
    fw_gmx_mdrun = Firework(fts_gmx_mdrun,
        name = ', '.join(('gmx_mdrun_em', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_queue_category'],
            '_queueadapter': {
                'queue':           hpc_max_specs[machine]['queue'],
                'walltime' :       hpc_max_specs[machine]['walltime'],
                'ntasks':          96,
            },
            '_files_in':  files_in,
            '_files_out': files_out,
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'gmx_mdrun_em',
                 **d
            }
        },
        parents = [ fw_gmx_grompp ] )
    
    fw_list.append(fw_gmx_mdrun)
    
# Store results
    files_in = {
        'energy_file':     'em.edr',
        'trajectory_file': 'em.trr',
        'final_config':    'em.gro' }
    files_out = {}
    
    fts_push = [ 
        AddFilesTask( {
            'compress': True ,
            'paths': "em.edr",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'em_edr',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "em.trr",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'em_trr',
                 **d } 
        } ),
        AddFilesTask( {
            'compress': True ,
            'paths': "em.gro",
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'type':    'em_gro',
                 **d } 
        } ) ]
    
    fw_push = Firework(fts_push,
        name = ', '.join(('push', fw_name_template.format(**d))),
        spec = {
            '_category': hpc_max_specs[machine]['fw_noqueue_category'],
            '_files_in': files_in, 
            'metadata': {
                'project': project_id,
                'datetime': str(datetime.datetime.now()),
                'step':    'push',
                 **d
            }
        },
        parents = [ fw_gmx_mdrun ] )
        
    fw_list.append(fw_push)

wf = Workflow(fw_list,
    name = wf_name,
    metadata = {
        'project': project_id,
        'datetime': str(datetime.datetime.now()),
        'type':    'gmx_prep'
    })


# In[292]:


wf.to_file('wf-{:s}.yaml'.format(project_id))


# In[294]:


lp.add_wf(wf)


# #### Build workflow: prep & em

# In[450]:


machine = 'juwels_devel'


# In[451]:


parametric_dimension_labels = ['nmolecules']


# In[452]:


parametric_dimensions = [ {
    'nmolecules': N } ]


# In[453]:


# for testing
parametric_dimensions = [ {
    'nmolecules': [N[0]] } ]


# In[454]:


parameter_sets = list( 
    itertools.chain(*[ 
            itertools.product(*list(
                    p.values())) for p in parametric_dimensions ]) )

parameter_dict_sets = [ dict(zip(parametric_dimension_labels,s)) for s in parameter_sets ]


# In[411]:


source_project_id = 'juwels-packmol-2020-03-09'
project_id = 'juwels-gromacs-em-2020-03-12'
infile_prefix = 'gmx_em_infiles'


# In[412]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[413]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[414]:


infiles = sorted(glob.glob(os.path.join(infile_prefix,'*')))

files = { os.path.basename(f): f for f in infiles }

# metadata common to all these files 
metadata = {
    'project': project_id,
    'type': 'input'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )


# In[415]:


wf_name = 'GROMACS preparations & energy minimization, {machine:}, {id:}'.format(machine=machine,id=project_id)

fw_name_template = 'nmolecules: {nmolecules:d}'

fw_list = []

fts_root_dummy = [
    CmdTask(
        cmd='echo',
        opt=['"Dummy root"'],
        store_stdout = False,
        store_stderr = False,
        use_shell    = True,
        fizzle_bad_rc= True) ]
  
files_out = {}
    
fw_root_dummy = Firework(fts_root_dummy,
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        '_files_out': files_out, 
        'metadata': {
          'project': project_id,
          'datetime': str(datetime.datetime.now()),
          'step':    'dummy_root'
        }
    }
)

fw_list.append(fw_root_dummy)

files_out = { 'input_file': 'default.mdp' }

fts_root_pull = [ 
    GetFilesByQueryTask(
        query = {
            'metadata->project': project_id,
            'metadata->name':    'em.mdp'
        },
        limit = 1,
        new_file_names = ['default.mdp'] ) ]

fw_root_pull = Firework(fts_root_pull,
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        '_files_out': files_out, 
        'metadata': {
          'project': project_id,
          'datetime': str(datetime.datetime.now()),
          'step':    'input_file_query'
        }
    }
)

fw_list.append(fw_root_pull)

## Parametric sweep

for d in parameter_dict_sets:        
    fw_list_tmp, fw_gmx_prep_pull_leaf = sub_wf_gmx_prep_pull(d, [fw_root_dummy])
    fw_list.extend(fw_list_tmp)
    
    fw_list_tmp, fw_gmx_prep_leaf = sub_wf_gmx_prep(d, [fw_gmx_prep_pull_leaf])
    fw_list.extend(fw_list_tmp)
    
    fw_list_tmp, _ = sub_wf_gmx_prep_push(d, [fw_gmx_prep_leaf])
    fw_list.extend(fw_list_tmp)
    
    fw_list_tmp, fw_gmx_mdrun = sub_wf_gmx_em(d, [fw_root_pull,fw_gmx_prep_leaf])
    fw_list.extend(fw_list_tmp)

    fw_list_tmp, fw_gmx_mdrun_push = sub_wf_gmx_em_psuh(d, [fw_gmx_mdrun])
    fw_list.extend(fw_list_tmp)


wf = Workflow(fw_list,
    name = wf_name,
    metadata = {
        'project': project_id,
        'datetime': str(datetime.datetime.now()),
        'type':    'gmx_prep'
    })


# In[440]:


fw_list_tmp, fw_gmx_mdrun_push = sub_wf_gmx_em_push(d, None)
#fw_list.extend(fw_list_tmp)


# In[441]:


len(fw_list_tmp)


# In[442]:


sub_wf = Workflow(fw_list_tmp)


# In[449]:


lp.append_wf(sub_wf,[24198])


# In[447]:


wf.as_dict()


# In[417]:


wf.to_file('wf-{:s}.yaml'.format(project_id))


# In[418]:


lp.add_wf(wf)


# #### Inspect sweep results

# In[461]:


query = { 
    "metadata.project": project_id,
    "metadata.type":    'em_edr',
}
fp.filepad.count_documents(query)


# In[462]:


parameter_names = ['nmolecules']


# In[532]:


em_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)

for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    with tempfile.NamedTemporaryFile(suffix='.edr') as tmp:
        tmp.write(content)
        em_df = panedr.edr_to_df(tmp.name)
        
        mi = pd.MultiIndex.from_product(
            [[int(metadata["metadata"]["nmolecules"])],em_df.index],
            names=['nmolecules','step'])
        em_mi_df = em_df.set_index(mi)
        em_list.append(em_mi_df)
    print('.',end='')
print('')
em_df = pd.concat(em_list)


# In[537]:


fig, ax = plt.subplots(3,2,figsize=(10,12))
em_df.plot('Time','Potential',ax=ax[0,0])
em_df.plot('Time','Pressure',ax=ax[0,1])
em_df.plot('Time','Bond',ax=ax[1,0])
#em_df.plot('Time','Position Rest.',ax=ax[1,1])
#em_df.plot('Time','COM Pull En.',ax=ax[1,1])
em_df.plot('Time','Coulomb (SR)',ax=ax[2,0])
em_df.plot('Time','Coul. recip.',ax=ax[2,1])


# In[559]:


query = { 
    "metadata.project": project_id,
    "metadata.type":    { '$in': ['em_trr','em_gro'] },
}
fp.filepad.count_documents(query)


# In[731]:


# Building a rather sophisticated aggregation pipeline

parameter_names = ['nmolecules', 'type']

query = { 
    "metadata.project": project_id,
    "metadata.type":    { '$in': ['em_trr','em_gro'] },
}

aggregation_pipeline = []

aggregation_pipeline.append({ 
    "$match": query
})


aggregation_pipeline.append({ 
    "$sort": { 
        "metadata.nmolecules": pymongo.ASCENDING,
        "metadata.datetime": pymongo.DESCENDING,
    }
})

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
})

parameter_names = ['nmolecules']

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { p: '$_id.{}'.format(p) for p in parameter_names },
        "type":     {"$addToSet": "$_id.type"},
        "gfs_id":   {"$addToSet": "$latest"} 
        #"$_id.type": "$latest"
    }
})

aggregation_pipeline.append({
    '$project': {
         '_id': False,
        **{ p: '$_id.{}'.format(p) for p in parameter_names},
        'objects': { 
            '$zip': {
                'inputs': [ '$type', '$gfs_id' ],
                'useLongestLength': True,
                'defaults':  [None,None]
            }
        }
    }
})

aggregation_pipeline.append({ 
    '$project': {
        **{ p: True for p in parameter_names},
        'objects': {'$arrayToObject': '$objects'}
        #'objects': False 
    }
})

aggregation_pipeline.append({ 
    '$addFields': {
        'objects': { **{ p: '${}'.format(p) for p in parameter_names} }
    }
})

aggregation_pipeline.append({ 
    '$replaceRoot': { 'newRoot': '$objects' }
})

# display results with
# for i, c in enumerate(cursor): 
#    print(c)
# yields documents in the form
# {'em_gro': '5e6a4e3d6c26f976ceae5e38', 'em_trr': '5e6a4e3a6c26f976ceae5e14', 'nmolecules': '44'}
# i.e. most recent topology file and trajectory file per concentration

cursor = fp.filepad.aggregate(aggregation_pipeline)


# In[732]:


mda_trr_list = []
for i, c in enumerate(cursor): 
    em_gro_content, _ = fp.get_file_by_id(c["em_gro"])
    em_trr_content, _ = fp.get_file_by_id(c["em_trr"])
    # STream approach won't work
    # with io.TextIOWrapper( io.BytesIO(em_gro_content) ) as gro, \
    #    io.BytesIO(em_trr_content) as trr:   
        #mda_trr_list.append( 
        #    mda.Universe( 
        #        gro,trr, topology_format = 'GRO', format='TRR') )
    with tempfile.NamedTemporaryFile(suffix='.gro') as gro,         tempfile.NamedTemporaryFile(suffix='.trr') as trr:
        gro.write(em_gro_content)
        trr.write(em_trr_content)
        mda_trr_list.append( mda.Universe(gro.name,trr.name) )
    print('.',end='')
print('')


# In[733]:


mda_trr = mda_trr_list[0]

mda_view = nglview.show_mdanalysis(mda_trr)

mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation('ball+stick')
mda_view


# ### Batch processing: packing, prep & em

# In[785]:


machine = 'juwels_devel'


# In[786]:


parametric_dimension_labels = ['nmolecules']


# In[787]:


N


# In[788]:


parametric_dimensions = [ {
    'nmolecules': N } ]


# In[789]:


# for testing
parametric_dimensions = [ {
    'nmolecules': [N[-1]] } ]


# In[790]:


parameter_sets = list( 
    itertools.chain(*[ 
            itertools.product(*list(
                    p.values())) for p in parametric_dimensions ]) )

parameter_dict_sets = [ dict(zip(parametric_dimension_labels,s)) for s in parameter_sets ]


# In[791]:


# source_project_id = 'juwels-packmol-2020-03-09'
project_id = 'juwels-afm-probe-solvation-trial-a-2020-03-13'
# infile_prefix = 'gmx_em_infiles'


# In[792]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[793]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# #### Provide PACKMOL template 

# In[794]:


infile_prefix = os.path.join(prefix,'packmol_infiles')

infiles = sorted(glob.glob(os.path.join(infile_prefix,'*.inp')))

files = { os.path.basename(f): f for f in infiles }

# metadata common to all these files 
metadata = {
    'project': project_id,
    'type': 'template'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )


# In[795]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
    'metadata.type': 'template'
}

# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[796]:


print(identifier)


# In[797]:


# on a lower level, each object has a unique "GridFS id":
pprint(fp_files) # underlying GridFS id and readable identifiers


# #### Provide data files

# In[798]:


data_prefix = os.path.join(prefix,'packmol_datafiles')


# In[799]:


datafiles = sorted(glob.glob(os.path.join(data_prefix,'*')))

files = { os.path.basename(f): f for f in datafiles }

# metadata common to all these files 
metadata = {
    'project': project_id,
    'type': 'data'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )


# #### Provide energy minimization input files

# In[800]:


infile_prefix = 'gmx_em_infiles'

infiles = sorted(glob.glob(os.path.join(infile_prefix,'*')))

files = { os.path.basename(f): f for f in infiles }

# metadata common to all these files 
metadata = {
    'project': project_id,
    'type': 'input'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )


# #### Build workflow

# In[809]:


wf_name = 'pack, preparations & energy minimization, {machine:}, {id:}'.format(machine=machine,id=project_id)

fw_name_template = 'nmolecules: {nmolecules:d}'

fw_list = []

# sub-wf pack root
fts_root_pack = [ 
        GetFilesByQueryTask(
            query = {
                'metadata->project': project_id,
                'metadata->name':    'surfactants_on_sphere.inp'
            },
            limit = 1,
            new_file_names = ['input.template'] ),
        GetFilesByQueryTask(
            query = {
                'metadata->project': project_id,
                'metadata->name':    'indenter_reres.pdb'
            },
            limit = 1,
            new_file_names = ['indenter.pdb'] ),
        GetFilesByQueryTask(
            query = {
                'metadata->project': project_id,
                'metadata->name':    '1_SDS.pdb'
            },
            limit = 1,
            new_file_names = ['1_SDS.pdb'] ),
        GetFilesByQueryTask(
            query = {
                'metadata->project': project_id,
                'metadata->name':    '1_NA.pdb'
            },
            limit = 1,
            new_file_names = ['1_NA.pdb'] )
        ]

files_out = {
    'input_file': 'input.template',
    'indenter_file': 'indenter.pdb',
    'surfatcant_file': '1_SDS.pdb',
    'counterion_file': '1_NA.pdb'}

fw_root_pack = Firework(fts_root_pack,
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        '_files_out': files_out, 
        'metadata': {
          'project': project_id,
          'datetime': str(datetime.datetime.now()),
          'subwf':   'pack',
          'step':    'input_file_query'
        }
    }
)

fw_list.append(fw_root_pack)

# sub-wf gmx prep root

fts_root_prep = [
    CmdTask(
        cmd='echo',
        opt=['"Dummy root"'],
        store_stdout = False,
        store_stderr = False,
        use_shell    = True,
        fizzle_bad_rc= True) ]
  
files_out = {}

fw_root_prep = Firework(fts_root_prep,
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        '_files_out': files_out, 
        'metadata': {
          'project': project_id,
          'datetime': str(datetime.datetime.now()),
          'subwf':   'prep',
          'step':    'dummy_root'
        }
    }
)

fw_list.append(fw_root_prep)

# sub-wf gmx em root
files_out = { 'input_file': 'default.mdp' }

fts_root_em = [ 
    GetFilesByQueryTask(
        query = {
            'metadata->project': project_id,
            'metadata->name':    'em.mdp'
        },
        limit = 1,
        new_file_names = ['default.mdp'] ) ]

fw_root_em = Firework(fts_root_em,
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        '_files_out': files_out, 
        'metadata': {
          'project': project_id,
          'datetime': str(datetime.datetime.now()),
          'subwf':   'em',
          'step':    'input_file_query'
        }
    }
)

fw_list.append(fw_root_em)

## Parametric sweep

for d in parameter_dict_sets:        
    # pack
    fw_list_tmp, fw_pack_leaf = sub_wf_pack(d, [fw_root_pack])
    fw_list.extend(fw_list_tmp)
    
    fw_list_tmp, _ = sub_wf_pack_push(d, [fw_pack_leaf])
    fw_list.extend(fw_list_tmp)
    
    # prep
    fw_list_tmp, fw_prep_leaf = sub_wf_gmx_prep(d, [fw_pack_leaf])
    fw_list.extend(fw_list_tmp)
    
    fw_list_tmp, _ = sub_wf_gmx_prep_push(d, [fw_prep_leaf])
    fw_list.extend(fw_list_tmp)
    
    # em
    fw_list_tmp, fw_gmx_mdrun = sub_wf_gmx_em(d, [fw_root_em,fw_prep_leaf])
    fw_list.extend(fw_list_tmp)

    fw_list_tmp, fw_gmx_mdrun_push = sub_wf_gmx_em_push(d, [fw_gmx_mdrun])
    fw_list.extend(fw_list_tmp)
    
wf = Workflow(fw_list,
    name = wf_name,
    metadata = {
        'project': project_id,
        'datetime': str(datetime.datetime.now()),
        'type':    'pack, prep, em'
    })


# In[812]:


wf.to_file('wf-{:s}.json'.format(project_id))


# In[813]:


lp.add_wf(wf)


# ## Pulling

# Utilize harmonic pulling to attach surfactants to substrate closely.

# ### Create index groups for pulling

# In[37]:


#pdb = '200_SDS_on_50_Ang_AFM_tip_model.pdb'
gro = 'em.gro'
top = 'sys.top'
ndx = 'standard.ndx'


# In[743]:


import parmed as pmd
pmd_top_gro = pmd.gromacs.GromacsTopologyFile(top)
#pmd_top_pdb = pmd.gromacs.GromacsTopologyFile(top)

pmd_gro = pmd.gromacs.GromacsGroFile.parse(gro)
pmd_top_gro.box = pmd_gro.box
pmd_top_gro.positions = pmd_gro.positions

#pmd_pdb = pmd.formats.pdb.PDBFile.parse(pdb)
#pmd_top_pdb.box = pmd_pdb.box
#pmd_top_pdb.positions = pmd_pdb.positions


# In[327]:


tail_atom_ndx = np.array([
        i+1 for i,a in enumerate(pmd_top_gro.atoms) if a.name == 'C12' and a.residue.name == 'SDS' ])
# gromacs ndx starts at 1


# In[814]:


tail_atom_ndx


# ### Generate standard index file for system

# In[823]:


gmx_make_ndx = gromacs.make_ndx.Popen(
    f=gro,o=ndx,
    input='q',
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)


# In[824]:


out_str, err_str = gmx_make_ndx.communicate()


# In[825]:


print(out_str)


# In[826]:


print(err_str)


# ### Enhance standard index file by pulling groups

# In[333]:


pull_groups_ndx_in = gromacs.fileformats.NDX(ndx)


# In[334]:


pull_groups_ndx_out = gromacs.fileformats.NDX()


# In[335]:


for i, a in enumerate(tail_atom_ndx):
    pull_group_name = 'pull_group_{:04d}'.format(i)
    pull_groups_ndx_out[pull_group_name] = a


# In[336]:


pull_groups_ndx_in.update(pull_groups_ndx_out)


# In[337]:


pull_groups_ndx_in.write('pull_groups.ndx')


# ### Create mdp input file with pulling groups and coordinates

# In[445]:


# gromacs wrapper parses mdp files
pull_mdp = gromacs.fileformats.MDP('pull.mdp.template')

pull_mdp['nsteps']  = 10000

N_pull_coords = len(pull_groups_ndx_out)

pull_mdp['pull_ncoords']  = N_pull_coords
pull_mdp['pull_ngroups']  = N_pull_coords + 1
pull_mdp['pull_group1_name'] = 'Substrate' # the reference group

for i, n in enumerate(pull_groups_ndx_out):
    pull_mdp["pull_group{:d}_name".format(i+2)]   = n
    pull_mdp["pull_coord{:d}_type".format(i+1)]     = 'umbrella'  # harmonic potential
    pull_mdp["pull_coord{:d}_geometry".format(i+1)] = 'distance'  # simple distance increase
    pull_mdp["pull_coord{:d}_dim".format(i+1)]      = 'Y Y Y'     # pull in all directions
    pull_mdp["pull_coord{:d}_groups".format(i+1)]   = "1 {:d}".format(i+2) # groups 1 (Chain A) and 2 (Chain B) define the reaction coordinate
    pull_mdp["pull_coord{:d}_start".format(i+1)]    = 'yes'       # define initial COM distance > 0
    pull_mdp["pull_coord{:d}_rate".format(i+1)]     = -0.1         # 0.1 nm per ps = 10 nm per ns
    pull_mdp["pull_coord{:d}_k".format(i+1)]        = 1000        # kJ mol^-1 nm^-2

pull_mdp.write('pull.mdp')


# ### Compile system

# In[446]:


gmx_grompp = gromacs.grompp.Popen(
    f='pull.mdp',n='pull_groups.ndx',c=gro,r=gro,o='pull.tpr',p=top,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)

out = gmx_grompp.stdout.read()

err = gmx_grompp.stderr.read()


# In[447]:


print(err)


# In[448]:


print(out)


# ### Run pulling simulation

# In[449]:


gmx_mdrun = gromacs.mdrun.Popen(
    deffnm='pull',v=True,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)


# In[450]:


out = gmx_mdrun.stdout.read()
err = gmx_mdrun.stderr.read()


# In[451]:


print(err)


# In[779]:


print(out)


# ### Batch processing

# In[59]:


# cast into parameters
surfactant = 'SDS'
tail_atom_name = 'C12'
substrate = 'AUM'
counterion = 'NA'
nsubstrate = 3873


# #### Build workflow: pull

# In[190]:


machine = 'juwels_devel'


# In[191]:


parametric_dimension_labels = ['nmolecules']


# In[192]:


parametric_dimensions = [ {
    'nmolecules': N } ]


# In[240]:


# for testing
parametric_dimensions = [ {
    'nmolecules': [N[1]] } ]


# In[243]:


parameter_sets = list( 
    itertools.chain(*[ 
            itertools.product(*list(
                    p.values())) for p in parametric_dimensions ]) )

parameter_dict_sets = [ dict(zip(parametric_dimension_labels,s)) for s in parameter_sets ]


# In[244]:


source_project_id = 'juwels-gromacs-prep-2020-03-11'
project_id = 'juwels-pull-2020-03-17'
infile_prefix = 'gmx_infiles'


# In[245]:


# queries to the data base are simple dictionaries
query = {
    'metadata.project': project_id,
}


# In[246]:


# use underlying MongoDB functionality to check total number of documents matching query
fp.filepad.count_documents(query)


# In[247]:


infiles = sorted(glob.glob(os.path.join(infile_prefix,'*')))

files = { os.path.basename(f): f for f in infiles }

# metadata common to all these files 
metadata = {
    'project': project_id,
    'type': 'input'
}

fp_files = []

# insert these input files into data base
for name, file_path in files.items():
    identifier = '/'.join((project_id,name)) # identifier is like a path on a file system
    metadata["name"] = name
    fp_files.append( fp.add_file(file_path,identifier=identifier,metadata = metadata) )


# In[248]:


fp_files


# In[272]:


wf_name = 'GROMACS surfactant pulling {machine:}, {id:}'.format(machine=machine,id=project_id)

fw_name_template = 'nmolecules: {nmolecules:d}'

fw_list = []

files_out = { 
    'template_file': 'sys.top.template',
    'parameter_file':      'pull.mdp.template',
}

fts_root_pull = [ 
    GetFilesByQueryTask(
        query = {
            'metadata->project': project_id,
            'metadata->name':    'sys.top.template',
        },
        limit = 1,
        new_file_names = ['sys.top.template'] ),
    GetFilesByQueryTask(
        query = {
            'metadata->project': project_id,
            'metadata->name':   'pull.mdp.template',
        },
        limit = 1,
        new_file_names = ['pull.mdp.template'] ) ]

fw_root_pull = Firework(fts_root_pull,
    name = wf_name,
    spec = {
        '_category': hpc_max_specs[machine]['fw_noqueue_category'],
        '_files_out': files_out, 
        'metadata': {
          'project': project_id,
          'datetime': str(datetime.datetime.now()),
          'step':    'input_file_query'
        }
    }
)

fw_list.append(fw_root_pull)

## Parametric sweep

for d in parameter_dict_sets:        
    # pull prep pull files
    fw_list_tmp, fw_pull_prep_pull_leaf = sub_wf_pull_prep_pull(d, [])
    fw_list.extend(fw_list_tmp)
    
    # pull prep
    fw_list_tmp, fw_pull_prep_leaf = sub_wf_pull_prep(d, [fw_root_pull,fw_pull_prep_pull_leaf])
    fw_list.extend(fw_list_tmp)

    # pull prep push files
    fw_list_tmp, _ = sub_wf_pull_prep_push(d, [fw_pull_prep_leaf])
    fw_list.extend(fw_list_tmp)
    
    # pull
    fw_list_tmp, fw_pull_leaf = sub_wf_gmx_pull(d, [fw_pull_prep_leaf])
    fw_list.extend(fw_list_tmp)
    
    # pull push files
    fw_list_tmp, _ = sub_wf_gmx_pull_push(d, [fw_pull_leaf])
    fw_list.extend(fw_list_tmp)
    
    
wf = Workflow(fw_list,
    name = wf_name,
    metadata = {
        'project': project_id,
        'datetime': str(datetime.datetime.now()),
        'type':    'pull'
    })


# In[273]:


wf.to_file('wf-{:s}.yaml'.format(project_id))


# In[274]:


lp.add_wf(wf)


# ## Pulling analysis

# ### Global observables

# The `gmx energy` table:

# ```
#   1  Restraint-Pot.   2  U-B              3  Proper-Dih.      4  LJ-14         
#   5  Coulomb-14       6  LJ-(SR)          7  Coulomb-(SR)     8  Coul.-recip.  
#   9  Position-Rest.  10  COM-Pull-En.    11  Potential       12  Kinetic-En.   
#  13  Total-Energy    14  Temperature     15  Pressure        16  Constr.-rmsd  
#  17  Vir-XX          18  Vir-XY          19  Vir-XZ          20  Vir-YX        
#  21  Vir-YY          22  Vir-YZ          23  Vir-ZX          24  Vir-ZY        
#  25  Vir-ZZ          26  Pres-XX         27  Pres-XY         28  Pres-XZ       
#  29  Pres-YX         30  Pres-YY         31  Pres-YZ         32  Pres-ZX       
#  33  Pres-ZY         34  Pres-ZZ         35  #Surf*SurfTen   36  T-rest       
#  ```
#  converted to dict with regex
#  
#      \s+([0-9]+)\s+([^\s]+)
#  
#  and replacement
#  
#      '$2': $1,\n

# In[420]:


gmx_energy_dict = {
    'Restraint-Pot.': 1,
    'U-B': 2,
    'Proper-Dih.': 3,
    'LJ-14': 4,
    'Coulomb-14': 5,
    'LJ-(SR)': 6,
    'Coulomb-(SR)': 7,
    'Coul.-recip.': 8,
    'Position-Rest.': 9,
    'COM-Pull-En.': 10,
    'Potential': 11,
    'Kinetic-En.': 12,
    'Total-Energy': 13,
    'Temperature': 14,
    'Pressure': 15,
    'Constr.-rmsd': 16,
    'Vir-XX': 17,
    'Vir-XY': 18,
    'Vir-XZ': 19,
    'Vir-YX': 20,
    'Vir-YY': 21,
    'Vir-YZ': 22,
    'Vir-ZX': 23,
    'Vir-ZY': 24,
    'Vir-ZZ': 25,
    'Pres-XX': 26,
    'Pres-XY': 27,
    'Pres-XZ': 28,
    'Pres-YX': 29,
    'Pres-YY': 30,
    'Pres-YZ': 31,
    'Pres-ZX': 32,
    'Pres-ZY': 33,
    'Pres-ZZ': 34,
    '#Surf*SurfTen': 35,
    'T-rest': 36,
}


# In[429]:


query = { 
    "metadata.project": project_id,
    "metadata.type":    'pull_edr',
    #"metadata.nmolecules": '44'
}

fp.filepad.count_documents(query)


# In[430]:


parameter_names = ['nmolecules']


# In[431]:


gmx_energy_selection = [
    'Restraint-Pot.',
    'Position-Rest.',
    'COM-Pull-En.',
    'Potential',
    'Kinetic-En.',
    'Total-Energy',
    'Temperature',
    'Pressure',
    'Constr.-rmsd',
]


# In[508]:


res_list = []
failed_list = []

match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)


for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    nmolecules = int(metadata["metadata"]["nmolecules"])
    
    #df = panedr.edr_to_df(tmp.name), fails
    tmpin = tempfile.NamedTemporaryFile(mode='w+b',suffix='.edr', delete=False)
    
    # cur_res_dict = {}
    with tmpin:
        tmpin.write(content)
        #tmpin.seek(0)
        
    for sel in gmx_energy_selection:  
        try:
            tmpout = tempfile.NamedTemporaryFile(suffix='.xvg', delete=False)
            res = gromacs.energy(f=tmpin.name,o=tmpout.name,
                                 input=str(gmx_energy_dict[sel]))
            #with open(tmpout.name,'r') as f:
            #    xvg = f.read()
            #tmpout.delete()
            xvg = mda.auxiliary.XVG.XVGReader(tmpout.name)
            xvg_time = xvg.read_all_times()
            xvg_data = np.array([ f.data[1:] for f in xvg ]).flatten() # 1st entry contains times
            os.unlink(tmpout.name)
        except: 
            logger.warning("Failed to read '{:s}' from data set {:d}.".format(sel,i))
            failed_list.append((nmolecules, sel))
        else:
            res_list.append({'nmolecules': [nmolecules]*len(xvg_time), 'time': xvg_time, sel: xvg_data})
            #[[int(metadata["metadata"]["nmolecules"])],df.index]
            #mi = pd.MultiIndex.from_product(
            #    [[int(metadata["metadata"]["nmolecules"])],df.index],
            #    names=['nmolecules','step'])
            #mi_df = df.set_index(mi)
            #res_list.append(mi_df)
            
    #tmpin.delete()
    os.unlink(tmpin.name)
    print('.',end='')
print('')
#res_df = pd.concat(res_list)


# In[538]:


res_df = None
for r in res_list:
    #nmolecules = r['nmolecules']
    #data={key: value for key, value in r.items() if key != 'nmolecules'}
    # columns=[key if key == 'time' else (nmolecules,key) for key in data]
    
    cur_df = pd.DataFrame(r)
    if res_df is None:
        res_df = cur_df
    else:
        res_df = pd.merge(res_df, cur_df, how='outer', on=['nmolecules', 'time'])
res_df_mi = res_df.set_index(['nmolecules','time'])


# In[539]:


res_df


# In[544]:


fig, ax = plt.subplots(3,2,figsize=(10,12))
res_df.plot('time','Potential',ax=ax[0,0])
res_df.plot('time','Pressure',ax=ax[0,1])
res_df.plot('time','Restraint-Pot.',ax=ax[1,0])
res_df.plot('time','Position-Rest.',ax=ax[1,1])
res_df.plot('time','COM-Pull-En.',ax=ax[2,0])
#res_df.plot('Coulomb (SR)',ax=ax[2,0])
#res_df.plot('Coul. recip.',ax=ax[2,1])


# ### Pulling forces

# In[567]:


res_list = []
failed_list = []

query = { 
    "metadata.project": project_id,
    "metadata.type":    'pullf_xvg',
}

fp.filepad.count_documents(query)
match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)


for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    nmolecules = int(metadata["metadata"]data:image/png;base64,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OGcd63Yc8Ds0ERGR4Btd67f/9/pKVpWUNekxczLacP/V/T+3zooVKxg+fPgJ5StXrjyhvEePHhw8eJCyskCcZsa4ceN466232L9/PxMnTmTTplNfUSovL2fu3Lk88MADp3E2J/fiiy/y4YcfHn38ySefALB06VJWrlxJRkYGY8aM4aOPPuKCCy5oktf009/e38DBqhq+P74PZsaD1wxgwaYP+P5L+bz2jfOJjdZ3KBER8Y/+CoWJyZMnM336dKZPn86UKZ87BysbNmxgyJAhjBkzhquuuoorrriiSWI4vru2VatWAIwcOZKsrCyioqIYMmQImzdvbpLX89OOsgqmfbyZa4Zk0qdzMgCpSfE8OGkAy4v387/vb/A5QhERiXS6knecU11xay79+/dvcKBETk4OH3zwwTFlGzduJCkpiTZt2hwtGzlyJMuXLycxMZHevXsfLV+wYAH/+Z//CcADDzzAoEGDjt6TV19MTMwxAzBONYXJI488whNPPAHA7NmzP7dufHz80e3o6Ghqamo+t34o+Mu766mtc3x3XO9jyq8alM6bK9L589z1XNqvE/3S25zkCCIiIs1LV/KCxCWXXEJlZSWPP/740bJly5bRp08fPvzww6OjZw8fPsy3vvUtfvSjH51wjIceeohf/epXx5SNGjXq6JW1iRMnnvT1u3XrxqpVq6isrGTfvn3MnTv3c+O95557jh43IyPjdE415G3dXc70TwuZPLILXTsknrD/gUkDaNsqlu+/lE+1RtuKiIhPlOQFCTNjxowZvPPOO/To0YP+/fvzk5/8hM6dOzNr1ix+8Ytf0KdPHwYOHMi5557Lvffee8IxrrjiCi6++OIzev0uXbpwww03MGDAAKq0OVoAACAASURBVG644QaGDh162sd48cUXj5lC5eOPP/7c+vfddx+5uWc8taFv/vjOOqKjjG9e0qvB/e1bx/GLawayalsZj8wraOHoREREAixcRjo21ogRI9yROd+OWL16Nf369fMpImlIsL4na7cf4PI/f8BdF57DT678/Pi+M30pbyzbxsx7xjAgs20LRSgNMbPFzrkRfsfRFBpqw0QkfJ1N+6UreSKn4fdvryUpLoa7v9DjlHX/e2J/2rWO4wcv51NVo25bERFpWUryRBopr3Afb6/awZ0XnUO71nGnrJ+SGMevrx3Imu0H+Mu761sgQhER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["nmolecules"])
    
    tmpin = tempfile.NamedTemporaryFile(mode='w+b',suffix='.xvg', delete=False)
    
    with tmpin:
        tmpin.write(content)
        
    try:
        xvg = mda.auxiliary.XVG.XVGReader(tmpin.name)
        xvg_time = xvg.read_all_times()
        xvg_data = np.array([ f.data[1:] for f in xvg ])# .flatten() # 1st entry contains times
    except: 
        logger.warning("Failed to read data set {:d}.".format(i))
        failed_list.append(nmolecules)
    else:
        res_list.append({
            'nmolecules': np.array([nmolecules]*len(xvg_time), dtype=int),
            'time': xvg_time, 
            **{i: xvg_data[:,i] for i in range(nmolecules)}
        })
       
    os.unlink(tmpin.name)
    print('.',end='')
print('')


# In[568]:


res_list


# In[581]:


res_df = None
for r in res_list:
    #r['data'] = {i: r['data'][:,i] for i in range(r['data'].shape[1])}
    cur_df = pd.DataFrame(r)
    if res_df is None:
        res_df = cur_df
    else:
        res_df = pd.merge(res_df, cur_df, how='outer', on=['nmolecules'])
res_df_mi = res_df.set_index(['nmolecules','time'])


# In[583]:


res_df_mi


# In[590]:


res_df_mi.plot(legend=False)


# In[591]:


res_df_mi.mean(axis=1).plot()


# ### Pulling groups movement

# In[733]:


res_list = []
failed_list = []

query = { 
    "metadata.project": project_id,
    "metadata.type":    'pullx_xvg',
}

fp.filepad.count_documents(query)
match_aggregation = {
        "$match": query
    }
sort_aggregation = {
        "$sort": { 
            "metadata.datetime": pymongo.DESCENDING,
        }
    }
group_aggregation = { 
    "$group": { 
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
}
aggregation_pipeline = [ match_aggregation, sort_aggregation, group_aggregation ]
cursor = fp.filepad.aggregate(aggregation_pipeline)


for i, c in enumerate(cursor): 
    content, metadata = fp.get_file_by_id(c["latest"])
    nmolecules = int(metadata["metadata"]["nmolecules"])
    
    tmpin = tempfile.NamedTemporaryFile(mode='w+b',suffix='.xvg', delete=False)
    
    with tmpin:
        tmpin.write(content)
        
    try:
        xvg = gromacs.fileformats.XVG(tmpin.name)
        xvg_time = xvg.array[0,:]
        
        #xvg_labels = ['1', '1 ref', '1 dX', '1 dY', '1 dZ', '1 g 1 X', '1 g 1 Y', '1 g 1 Z', '1 g 2 X', '1 g 2 Y', '1 g 2 Z']
        N_pull_coords = nmolecules
        N_cols = len(xvg.names)
        N_cols_per_coord = int(N_cols / N_pull_coords)
        
        xvg_labels = xvg.names[:N_cols_per_coord]
        xvg_data = {}
        for j in range(N_pull_coords):
            for k in range(N_cols_per_coord):
                xvg_data[(j,xvg_labels[k])] = xvg.array[
                    1+j*N_cols_per_coord+k,:]
        
    except: 
        logger.exception("Failed to read data set {:d}.".format(i))
        failed_list.append(nmolecules)

    else:
        res_list.append({
            'nmolecules': nmolecules, # np.array([nmolecules]*len(xvg_time), dtype=int),
            'time': xvg_time, 
            **xvg_data})
       
    #os.unlink(tmpin.name)
    print('.',end='')
print('')


# In[734]:


res_list


# In[743]:


res_df = None
for r in res_list:
    #r['data'] = {i: r['data'][:,i] for i in range(r['data'].shape[1])}
    cur_df = pd.DataFrame(r)
    if res_df is None:
        res_df = cur_df
    else:
        res_df = pd.merge(res_df, cur_df, how='outer', on=['nmolecules'])
res_df_mi = res_df.set_index(['nmolecules','time'])
res_df_mi.columns = pd.MultiIndex.from_tuples(res_df_mi.columns, names=['nmolecule', 'coord'])


# In[750]:


res_df_mi.groupby(axis=1,level='coord').mean()


# In[780]:


res_df_mi[0,'1'].plot()


# In[781]:


res_df_mi.groupby(axis=1,level='coord').mean()['1'].plot()


# In[782]:


res_df_mi.groupby(axis=1,level='coord').mean()['1 ref'].plot()


# In[472]:


# sqrt(dx^2+dy^2+dz^2), the distance between pulling groups (i.e. one surfactant tail atom and the Au COM)
for i in range(0,150,30):
    plt.plot(pull_x_xvg.array[0,:],
             np.sqrt(pull_x_xvg.array[i*11+3,:]**2+pull_x_xvg.array[i*11+4,:]**2+pull_x_xvg.array[i*11+5,:]**2))
plt.legend()


# In[473]:


pull_x_xvg.plot(columns=[0,12])


# ### Visualize trajectory

# In[795]:


# Building a rather sophisticated aggregation pipeline

parameter_names = ['nmolecules', 'type']

query = { 
    "metadata.project": project_id,
    "metadata.type":    { '$in': ['pull_trr','pull_gro'] },
}

aggregation_pipeline = []

aggregation_pipeline.append({ 
    "$match": query
})


aggregation_pipeline.append({ 
    "$sort": { 
        "metadata.nmolecules": pymongo.ASCENDING,
        "metadata.datetime": pymongo.DESCENDING,
    }
})

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { p: '$metadata.{}'.format(p) for p in parameter_names },
        "degeneracy": {"$sum": 1}, # number matching data sets
        "latest":     {"$first": "$gfs_id"} # unique gridfs id of file
    }
})

parameter_names = ['nmolecules']

aggregation_pipeline.append({ 
    "$group": { 
        "_id": { p: '$_id.{}'.format(p) for p in parameter_names },
        "type":     {"$addToSet": "$_id.type"},
        "gfs_id":   {"$addToSet": "$latest"} 
        #"$_id.type": "$latest"
    }
})

aggregation_pipeline.append({
    '$project': {
         '_id': False,
        **{ p: '$_id.{}'.format(p) for p in parameter_names},
        'objects': { 
            '$zip': {
                'inputs': [ '$type', '$gfs_id' ],
                'useLongestLength': True,
                'defaults':  [None,None]
            }
        }
    }
})

aggregation_pipeline.append({ 
    '$project': {
        **{ p: True for p in parameter_names},
        'objects': {'$arrayToObject': '$objects'}
        #'objects': False 
    }
})

aggregation_pipeline.append({ 
    '$addFields': {
        'objects': { **{ p: '${}'.format(p) for p in parameter_names} }
    }
})

aggregation_pipeline.append({ 
    '$replaceRoot': { 'newRoot': '$objects' }
})

# display results with
# for i, c in enumerate(cursor): 
#    print(c)
# yields documents in the form
# {'em_gro': '5e6a4e3d6c26f976ceae5e38', 'em_trr': '5e6a4e3a6c26f976ceae5e14', 'nmolecules': '44'}
# i.e. most recent topology file and trajectory file per concentration

cursor = fp.filepad.aggregate(aggregation_pipeline)


# In[797]:


failed_list
mda_trr_list = []
for i, c in enumerate(cursor): 
    try:
        gro_content, _ = fp.get_file_by_id(c["pull_gro"])
        trr_content, _ = fp.get_file_by_id(c["pull_trr"])
        with tempfile.NamedTemporaryFile(suffix='.gro') as gro,             tempfile.NamedTemporaryFile(suffix='.trr') as trr:
            gro.write(gro_content)
            trr.write(trr_content)
            mda_trr_list.append( mda.Universe(gro.name,trr.name) )
    except: 
        logger.exception("Failed to read data set {}.".format(c))
        failed_list.append(c)
    print('.',end='')
print('')


# In[798]:


failed_list


# In[799]:


mda_trr = mda_trr_list[0]

mda_view = nglview.show_mdanalysis(mda_trr)

mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation('ball+stick')
mda_view


# ### MSD

# In[476]:


substrate = mda_trr.atoms[mda_trr.atoms.names == 'AU']


# In[477]:


surfactant_head = mda_trr.atoms[mda_trr.atoms.names == 'S']


# In[478]:


rms_substrate = mda_rms.RMSD(substrate,ref_frame=0)


# In[479]:


rms_substrate.run()


# In[480]:


rmsd = rms_substrate.rmsd.T   # transpose makes it easier for plotting
time = rmsd[1]


# In[481]:


plt.plot(time,rmsd[2])


# In[482]:


rms_surfactant_head = mda_rms.RMSD(surfactant_head,ref_frame=0)


# In[483]:


rms_surfactant_head.run()


# In[484]:


rmsd = rms_surfactant_head.rmsd.T   # transpose makes it easier for plotting
time = rmsd[1]


# In[485]:


plt.plot(time,rmsd[2])


# ### Au-S (substrate - head group )RDF

# In[486]:


len(mda_trr.trajectory)


# In[487]:


rdf_substrate_headgroup = mda_rdf.InterRDF(
    substrate,surfactant_head,range=(0.0,80.0),verbose=True)


# In[488]:


bins = []
rdf  = []
for i in range(len(mda_trr.trajectory)):
    rdf_substrate_headgroup = mda_rdf.InterRDF(
        substrate,surfactant_head,range=(0.0,80.0),verbose=True)
    rdf_substrate_headgroup.run(start=i,stop=i+1)
    bins.append(rdf_substrate_headgroup.bins.copy())
    rdf.append(rdf_substrate_headgroup.rdf.copy())
bins = np.array(bins)
rdf = np.array(rdf)


# In[489]:


# indicates desired approach towards substrate
plt.plot(bins[0],rdf[0],label="Initial RDF")
plt.plot(bins[3],rdf[4],label="Intermediat RDF")
plt.plot(bins[-1],rdf[-1],label="Final RDF")
plt.legend()


# ### Single system global observables

# In[457]:


edr_file = 'pull.edr'


# In[458]:


edr_df = panedr.edr_to_df(edr_file)


# In[459]:


edr_df.columns


# In[460]:


fig, ax = plt.subplots(3,2,figsize=(10,12))
edr_df.plot('Time','Potential',ax=ax[0,0])
edr_df.plot('Time','Pressure',ax=ax[0,1])
#edr_df.plot('Time','Bond',ax=ax[1,0])
edr_df.plot('Time','Position Rest.',ax=ax[1,0])
edr_df.plot('Time','COM Pull En.',ax=ax[1,1])
edr_df.plot('Time','Coulomb (SR)',ax=ax[2,0])
edr_df.plot('Time','Coul. recip.',ax=ax[2,1])


# ### Pulling forces

# In[461]:


# read xvg file
pull_f_xvg = mda.auxiliary.XVG.XVGFileReader('pull_pullf.xvg')
pull_f_t = pull_f_xvg.read_all_times()
# first data column contains time, strip
pull_f = np.array([ f.data[1:] for f in pull_f_xvg ])


# In[462]:


for i in range(0,199,50):
    plt.plot(pull_f_t,pull_f[:,i])


# ### Pulling groups movement

# In[463]:


pull_x_xvg = gromacs.fileformats.XVG('pull_pullx.xvg',)


# In[464]:


pull_x_xvg.array


# In[465]:


len(pull_x_xvg.names)


# In[466]:


# that many columns perr pull coordinate
N_cols_per_coord = int(len(pull_x_xvg.names) / N_pull_coords)


# In[467]:


# with content
legend = pull_x_xvg.names[:11]


# In[468]:


legend


# In[469]:


pull_x_xvg.names[-3:]


# In[470]:


for i in range(11):
    plt.plot(pull_x_xvg.array[0,:],pull_x_xvg.array[i+1,:],label=legend[i])
plt.legend()


# In[471]:


for i in range(11):
    plt.plot(pull_x_xvg.array[0,:],pull_x_xvg.array[i+1,:],label=legend[i])
plt.legend()


# In[472]:


# sqrt(dx^2+dy^2+dz^2), the distance between pulling groups (i.e. one surfactant tail atom and the Au COM)
for i in range(0,150,30):
    plt.plot(pull_x_xvg.array[0,:],
             np.sqrt(pull_x_xvg.array[i*11+3,:]**2+pull_x_xvg.array[i*11+4,:]**2+pull_x_xvg.array[i*11+5,:]**2))
plt.legend()


# In[473]:


pull_x_xvg.plot(columns=[0,12])


# ### Visualize trajectory

# In[272]:


gro_em = 'pull.gro'


# In[273]:


mda_trr = mda.Universe('em.gro','pull.trr')

mda_view = nglview.show_mdanalysis(mda_trr)

mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation('ball+stick')
mda_view


# In[536]:


mda_xtc = mda.Universe(gro,'pull.xtc')
mda_view = nglview.show_mdanalysis(mda_xtc)
mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation('ball+stick')
mda_view


# ### MSD

# In[476]:


substrate = mda_trr.atoms[mda_trr.atoms.names == 'AU']


# In[477]:


surfactant_head = mda_trr.atoms[mda_trr.atoms.names == 'S']


# In[478]:


rms_substrate = mda_rms.RMSD(substrate,ref_frame=0)


# In[479]:


rms_substrate.run()


# In[480]:


rmsd = rms_substrate.rmsd.T   # transpose makes it easier for plotting
time = rmsd[1]


# In[481]:


plt.plot(time,rmsd[2])


# In[482]:


rms_surfactant_head = mda_rms.RMSD(surfactant_head,ref_frame=0)


# In[483]:


rms_surfactant_head.run()


# In[484]:


rmsd = rms_surfactant_head.rmsd.T   # transpose makes it easier for plotting
time = rmsd[1]


# In[485]:


plt.plot(time,rmsd[2])


# ### Au-S (substrate - head group )RDF

# In[486]:


len(mda_trr.trajectory)


# In[487]:


rdf_substrate_headgroup = mda_rdf.InterRDF(
    substrate,surfactant_head,range=(0.0,80.0),verbose=True)


# In[488]:


bins = []
rdf  = []
for i in range(len(mda_trr.trajectory)):
    rdf_substrate_headgroup = mda_rdf.InterRDF(
        substrate,surfactant_head,range=(0.0,80.0),verbose=True)
    rdf_substrate_headgroup.run(start=i,stop=i+1)
    bins.append(rdf_substrate_headgroup.bins.copy())
    rdf.append(rdf_substrate_headgroup.rdf.copy())
bins = np.array(bins)
rdf = np.array(rdf)


# In[489]:


# indicates desired approach towards substrate
plt.plot(bins[0],rdf[0],label="Initial RDF")
plt.plot(bins[3],rdf[4],label="Intermediat RDF")
plt.plot(bins[-1],rdf[-1],label="Final RDF")
plt.legend()


# ## Solvation

# Now, fill the box with water.

# In[560]:


gro = 'pull.gro'


# In[561]:


# use -scale 0.5 -maxsol N for non-standard conditions
gmx_solvate = gromacs.solvate.Popen(
    cp=gro, cs='spc216.gro',o='solvated.gro',p=top,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)

out = gmx_solvate.stdout.read()
err = gmx_solvate.stderr.read()


# In[562]:


print(out)


# In[563]:


print(err)


# ## Energy minimization with restraints

# Again, relax the system a little with positional constraints applied to all ions.

# ### Execute trial task via Fireworks on remote resource

# In[20]:


lpad = LaunchPad.auto_load()


# A trial task sent to FORHLR2:

# In[71]:


gmx_test_task = CmdTask(
    cmd = 'gmx',
    opt = '-h',
    stderr_file = 'std.err', 
    stdout_file = 'std.out', 
    use_shell = True)


# In[72]:


gmx_test_fw = Firework(
    gmx_test_task, 
    name = 'FORHLR2 GMX test fw',
    spec = { 
        '_category': 'forhlr2_noqueue',
        '_files_out': {
            'stdout': 'std.out',
            'stderr': 'std.err'} 
        } )


# In[73]:


fw_ids = lpad.add_wf(gmx_test_fw)


# In[74]:


fw_ids


# In[86]:


# lpad.delete_wf(INSERT_ID,delete_launch_dirs=True)


# ### Compile system

# In[237]:


top = 'sys.top'


# In[238]:


gro = 'solvated.gro'


# In[137]:


gmx_grompp = gromacs.grompp.Popen(
    f='em_solvated.mdp',c=gro,r=gro,o='em_solvated.tpr',p=top,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)

out = gmx_grompp.stdout.read()
err = gmx_grompp.stderr.read()


# In[138]:


print(err)


# In[139]:


print(out)


# ### Remote file transfer

# Utilize fabric to transfer files files to remote resource conveniently:

# In[30]:


c = fabric.Connection('forhlr2') # host defined in ssh config


# In[31]:


res = c.run('ws_find fw') # get remote directory of Fireworks workspace


# In[32]:


res.command


# In[144]:


now = datetime.now().isoformat()


# In[145]:


remote_path = os.path.sep.join((res.stdout.strip(),'file_transfer',now))


# In[194]:


remote_path


# In[147]:


res = c.run(' '.join(['mkdir','-p',remote_path]))


# In[148]:


file_name = 'em_solvated.tpr'


# In[188]:


local_file = os.path.sep.join((prefix,file_name))


# In[189]:


remote_file = os.path.sep.join((remote_path,file_name))


# In[198]:


res = c.put(local_file,remote_file)


# In[200]:


res.local


# In[193]:


# FileTransferTask does not work anymore


# In[191]:


#ft = FileTransferTask(
#    mode   = 'rtransfer',
#    files  = [ {'src':local_file, 'dest':remote_path} ],
#    server = c.host,
#    user   = c.user )


# In[192]:


#fw = Firework(
#    ft, 
#    name = 'BWCLOUD remote transef to FORHLR2',
#    spec = { 
#        '_category': 'bwcloud_std',
#        } )


# In[174]:


fw_ids = lpad.add_wf(fw)


# ### Run energy minimization

# In[239]:


ft = FileTransferTask(
    mode   = 'copy',
    files  = [ {'src':remote_file, 'dest':os.path.curdir} ] )


# In[240]:


gmx_mdrun_task = CmdTask(
    cmd = 'gmx',
    opt = ['mdrun','-v','-deffnm','em_solvated'],
    stderr_file = 'std.err', 
    stdout_file = 'std.out', 
    use_shell = True)


# In[241]:


gmx_log_tracker = Tracker('em_solvated.log')


# In[242]:


gmx_mdrun_fw = Firework(
    [ft,gmx_mdrun_task], 
    name = 'FORHLR2 GMX mdrun em_solvated',
    spec = { 
        '_category': 'forhlr2_queue',
        '_queueadapter': {
            'cpus_per_task':    1,
            'ntasks_per_node':  20,
            'ntasks':           40,
            'queue':            'normal',
            'walltime':         '24:00'
        },
        '_files_out': {
            'log': '*.log',
            'trr': '*.trr',
            'edr': '*.edr',
            'gro': '*.gro' },
        '_trackers' : [ gmx_log_tracker ]
        } )


# In[243]:


pprint(gmx_mdrun_fw.as_dict())


# In[244]:


fw_ids = lpad.add_wf(gmx_mdrun_fw)


# In[315]:


fw_id = list(fw_ids.values())[0]


# ### File transfer back

# instead of relying on the returned fw_id, we can also query the Firework added latest

# In[23]:


fw_ids = lpad.get_fw_ids(sort=[('created_on',pymongo.DESCENDING)],limit=1)


# In[25]:


fw_id = fw_ids[0]


# In[34]:


fw_id


# We query the remote directory our FireWork ran in

# In[28]:


launch_dir = lpad.get_launchdir(fw_id)


# In[35]:


launch_dir


# In[30]:


c = fabric.Connection('forhlr2') # host defined in ssh config


# In[36]:


res = c.run('ls -lht {}'.format(launch_dir)) # look at remote directory contents


# In[39]:


glob_pattern = os.path.join(launch_dir,'em_solvated.*')


# In[49]:


res = c.run('ls {}'.format(glob_pattern))


# In[53]:


res.stdout


# In[57]:


for f in res.stdout.splitlines():
    c.get(f)


# ### Energy minimization analysis

# In[97]:


em_file = 'em_solvated.edr'


# In[98]:


em_df = panedr.edr_to_df(em_file)


# In[99]:


em_df.columns


# In[100]:


fig, ax = plt.subplots(3,2,figsize=(10,12))
em_df.plot('Time','Potential',ax=ax[0,0])
em_df.plot('Time','Pressure',ax=ax[0,1])
em_df.plot('Time','Bond',ax=ax[1,0])
em_df.plot('Time','Position Rest.',ax=ax[1,1])
#em_df.plot('Time','COM Pull En.',ax=ax[1,1])
em_df.plot('Time','Coulomb (SR)',ax=ax[2,0])
em_df.plot('Time','Coul. recip.',ax=ax[2,1])


# In[173]:


try:
    del em_df
except:
    pass


# ### Visualize trajectory

# In[102]:


mda_trr = mda.Universe('solvated.gro','em_solvated.trr')


# In[103]:


# check unique resiude names in system
resnames = np.unique([ r.resname for r in mda_trr.residues ])


# In[104]:


resnames


# In[117]:


mda_view = nglview.show_mdanalysis(mda_trr)
mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation(repr_type='ball+stick',selection='SDS')
mda_view.add_representation(repr_type='ball+stick',selection='NA')
mda_view.add_representation(repr_type='spacefill',selection='AUM',color='yellow')
mda_view.center()


# In[118]:


mda_view


# In[159]:


try:
    del mda_trr
except:
    pass
try:
    del mda_view
except:
    pass


# ## NVT equilibration

# In[134]:


top = 'sys.top'
gro = 'em_solvated.gro'
ndx = 'nvt.ndx'


# In[133]:


lpad = LaunchPad.auto_load()


# ### Generate non-substrate index group

# In[141]:


pmd_top_gro = pmd.gromacs.GromacsTopologyFile(top)

pmd_gro = pmd.gromacs.GromacsGroFile.parse(gro)
pmd_top_gro.box = pmd_gro.box
pmd_top_gro.positions = pmd_gro.positions


# In[142]:


non_substrate_ndx = np.array([
        i+1 for i,a in enumerate(pmd_top_gro.atoms) if a.residue.name != 'AUM' ])
# gromacs ndx starts at 1


# len(pmd_top_gro.atoms)

# In[144]:


len(non_substrate_ndx)


# In[147]:


len(pmd_top_gro.atoms) - len(non_substrate_ndx) # double-check non-substrate and substrate atom numbers


# In[160]:


try:
    del pmd_top_gro
except:
    pass
try:
    del pmd_gro
except:
    pass


# ### Generate standard index file for system

# In[135]:


gmx_make_ndx = gromacs.make_ndx.Popen(
    f=gro,o=ndx,
    input='q',
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)


# In[136]:


out_str, err_str = gmx_make_ndx.communicate()


# In[137]:


print(out_str)


# In[138]:


print(err_str)


# ### Enhance standard index file by pulling groups

# In[150]:


ndx_in = gromacs.fileformats.NDX(ndx)


# In[152]:


ndx_in['non-Substrate'] = non_substrate_ndx


# In[155]:


ndx_in.write(ndx)


# In[175]:


try:
    del ndx_in
except:
    pass
try:
    del non_substrate_ndx
except:
    pass


# ### Compile system

# In[183]:


gmx_grompp = gromacs.grompp.Popen(
    f='nvt.mdp',n=ndx,c=gro,r=gro,o='nvt.tpr',p=top,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)

out = gmx_grompp.stdout.read()
err = gmx_grompp.stderr.read()


# In[184]:


print(err)


# In[185]:


print(out)


# ### Remote file transfer

# Utilize fabric to transfer files files to remote resource conveniently:

# In[279]:


c = fabric.Connection('forhlr2') # host defined in ssh config


# In[280]:


res = c.run('ws_find fw') # get remote directory of Fireworks workspace


# In[281]:


res.command


# In[282]:


now = datetime.now().isoformat()


# In[283]:


remote_path = os.path.sep.join((res.stdout.strip(),'file_transfer',now))


# In[191]:


remote_path


# In[192]:


res = c.run(' '.join(['mkdir','-p',remote_path]))


# In[193]:


file_name = 'nvt.tpr'


# In[194]:


local_file = os.path.sep.join((prefix,file_name))


# In[195]:


remote_file = os.path.sep.join((remote_path,file_name))


# In[196]:


res = c.put(local_file,remote_file)


# In[197]:


res.local


# ### Execute trial task via Fireworks on remote resource queue

# In[317]:


ft = FileTransferTask(
    mode   = 'copy',
    files  = [ {'src':remote_file, 'dest':os.path.curdir} ] )


# In[318]:


gmx_mdrun_task = CmdTask(
    cmd = 'gmx',
    opt = ['mdrun','-v','-deffnm','nvt'],
    stderr_file = 'std.err', 
    stdout_file = 'std.out', 
    use_shell = True)


# In[319]:


gmx_log_tracker = Tracker('nvt.log')


# In[320]:


gmx_mdrun_fw = Firework(
    [ft,gmx_mdrun_task], 
    name = 'FORHLR2 GMX mdrun nvt',
    spec = { 
        '_category': 'forhlr2_queue',
        '_queueadapter': {
            'cpus_per_task':    1,
            'ntasks_per_node':  20,
            'ntasks':           20,
            'queue':            'develop',
            'walltime':         '00:03:00'
        },
        '_files_out': {
            'log': '*.log',
            'trr': '*.trr',
            'edr': '*.edr',
            'gro': '*.gro' },
        '_trackers' : [ gmx_log_tracker ]
        } )


# In[321]:


pprint(gmx_mdrun_fw.as_dict())


# In[322]:


fw_ids = lpad.add_wf(gmx_mdrun_fw)


# In[332]:


fw_ids = lpad.get_fw_ids(query={'name':'FORHLR2 GMX mdrun nvt','spec._queueadapter.queue':'develop'})


# In[333]:


for fw_id in fw_ids:
    try:
        print("Deleting {}...".format(fw_id))
        lpad.delete_wf(fw_id)
    except:
        print("Failed deleting {}...".format(fw_id))
    


# ### Run NVT equilibration

# In[198]:


ft = FileTransferTask(
    mode   = 'copy',
    files  = [ {'src':remote_file, 'dest':os.path.curdir} ] )


# In[199]:


gmx_mdrun_task = CmdTask(
    cmd = 'gmx',
    opt = ['mdrun','-v','-deffnm','nvt'],
    stderr_file = 'std.err', 
    stdout_file = 'std.out', 
    use_shell = True)


# In[200]:


gmx_log_tracker = Tracker('nvt.log')


# In[201]:


gmx_mdrun_fw = Firework(
    [ft,gmx_mdrun_task], 
    name = 'FORHLR2 GMX mdrun nvt',
    spec = { 
        '_category': 'forhlr2_queue',
        '_queueadapter': {
            'cpus_per_task':    1,
            'ntasks_per_node':  20,
            'ntasks':           40,
            'queue':            'normal',
            'walltime':         '24:00'
        },
        '_files_out': {
            'log': '*.log',
            'trr': '*.trr',
            'edr': '*.edr',
            'gro': '*.gro' },
        '_trackers' : [ gmx_log_tracker ]
        } )


# In[202]:


pprint(gmx_mdrun_fw.as_dict())


# In[203]:


fw_ids = lpad.add_wf(gmx_mdrun_fw)


# In[204]:


fw_id = list(fw_ids.values())[0]


# ### File transfer back

# instead of relying on the returned fw_id, we can also query the Firework added latest

# In[205]:


fw_ids = lpad.get_fw_ids(sort=[('created_on',pymongo.DESCENDING)],limit=1)


# In[206]:


fw_id = fw_ids[0]


# In[207]:


fw_id


# We query the remote directory our FireWork ran in

# In[208]:


launch_dir = lpad.get_launchdir(fw_id)


# In[209]:


launch_dir


# In[210]:


c = fabric.Connection('forhlr2') # host defined in ssh config


# In[211]:


res = c.run('ls -lht {}'.format(launch_dir)) # look at remote directory contents


# In[213]:


glob_pattern = os.path.join(launch_dir,'nvt.*')


# In[214]:


res = c.run('ls {}'.format(glob_pattern))


# In[215]:


for f in res.stdout.splitlines():
    c.get(f)


# ### Analysis

# In[242]:


edr_file = 'nvt.edr'


# In[243]:


edr_df = panedr.edr_to_df(edr_file)


# In[244]:


edr_df.columns


# In[248]:


fig, ax = plt.subplots(3,2,figsize=(10,12))
edr_df.plot('Time','Temperature',ax=ax[0,0])
edr_df.plot('Time','Pressure',ax=ax[0,1])
edr_df.plot('Time','Potential',ax=ax[1,0])
edr_df.plot('Time','Bond',ax=ax[1,1])
#edr_df.plot('Time','Position Rest.',ax=ax[1,1])
#edr_df.plot('Time','COM Pull En.',ax=ax[1,1])
edr_df.plot('Time','Coulomb (SR)',ax=ax[2,0])
edr_df.plot('Time','Coul. recip.',ax=ax[2,1])


# ### Visualize trajectory

# In[102]:


mda_trr = mda.Universe('solvated.gro','em_solvated.trr')


# In[103]:


# check unique resiude names in system
resnames = np.unique([ r.resname for r in mda_trr.residues ])


# In[104]:


resnames


# In[117]:


mda_view = nglview.show_mdanalysis(mda_trr)
mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation(repr_type='ball+stick',selection='SDS')
mda_view.add_representation(repr_type='ball+stick',selection='NA')
mda_view.add_representation(repr_type='spacefill',selection='AUM',color='yellow')
mda_view.center()


# In[118]:


mda_view


# In[119]:


try:
    del mda_trr
except:
    pass
try:
    del mda_view
except:
    pass


# ## NPT equilibration

# In[110]:


top = 'sys.top'
gro = 'nvt.gro'
ndx = 'nvt.ndx'


# In[111]:


lpad = LaunchPad.auto_load()


# ### Compile system

# In[117]:


gmx_grompp = gromacs.grompp.Popen(
    f='npt.mdp',n=ndx,c=gro,r=gro,o='npt.tpr',p=top,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)

out = gmx_grompp.stdout.read()
err = gmx_grompp.stderr.read()


# In[118]:


print(err)


# In[119]:


print(out)


# ### Remote file transfer

# Utilize fabric to transfer files files to remote resource conveniently:

# In[120]:


c = fabric.Connection('forhlr2') # host defined in ssh config


# In[121]:


res = c.run('ws_find fw') # get remote directory of Fireworks workspace


# In[122]:


res.command


# In[123]:


now = datetime.now().isoformat()


# In[124]:


remote_path = os.path.sep.join((res.stdout.strip(),'file_transfer',now))


# In[125]:


remote_path


# In[126]:


res = c.run(' '.join(['mkdir','-p',remote_path]))


# In[127]:


file_name = 'npt.tpr'


# In[128]:


local_file = os.path.sep.join((prefix,file_name))


# In[129]:


remote_file = os.path.sep.join((remote_path,file_name))


# In[130]:


res = c.put(local_file,remote_file)


# ### Run NPT equilibration

# In[131]:


ft = FileTransferTask(
    mode   = 'copy',
    files  = [ {'src':remote_file, 'dest':os.path.curdir} ] )


# In[132]:


gmx_mdrun_task = CmdTask(
    cmd = 'gmx',
    opt = ['mdrun','-v','-deffnm','npt'],
    stderr_file = 'std.err', 
    stdout_file = 'std.out', 
    use_shell = True)


# In[133]:


gmx_log_tracker = Tracker('npt.log')


# In[134]:


gmx_mdrun_fw = Firework(
    [ft,gmx_mdrun_task], 
    name = 'FORHLR2 GMX mdrun npt',
    spec = { 
        '_category': 'forhlr2_queue',
        '_queueadapter': {
            'cpus_per_task':    1,
            'ntasks_per_node':  20,
            'ntasks':           20,
            'queue':            'normal',
            'walltime':         '24:00:00'
        },
        '_files_out': {
            'log': '*.log',
            'trr': '*.trr',
            'edr': '*.edr',
            'gro': '*.gro' },
        '_trackers' : [ gmx_log_tracker ]
        } )


# In[135]:


fw_ids = lpad.add_wf(gmx_mdrun_fw)


# In[136]:


fw_id = list(fw_ids.values())[0]


# In[137]:


fw_id


# ### File transfer back

# instead of relying on the returned fw_id, we can also query the Firework added latest

# In[47]:


fw_ids = lpad.get_fw_ids(sort=[('created_on',pymongo.DESCENDING)],limit=1)


# In[48]:


fw_id = fw_ids[0]


# In[49]:


fw_id


# We query the remote directory our FireWork ran in

# In[50]:


launch_dir = lpad.get_launchdir(fw_id)


# In[51]:


launch_dir


# In[52]:


c = fabric.Connection('forhlr2') # host defined in ssh config


# In[53]:


res = c.run('ls -lht {}'.format(launch_dir)) # look at remote directory contents


# In[54]:


glob_pattern = os.path.join(launch_dir,'npt.*')


# In[55]:


res = c.run('ls {}'.format(glob_pattern))


# In[56]:


for f in res.stdout.splitlines():
    c.get(f)


# ### Analysis

# In[57]:


edr_file = 'npt.edr'


# In[58]:


edr_df = panedr.edr_to_df(edr_file)


# In[59]:


edr_df.columns


# In[60]:


fig, ax = plt.subplots(3,2,figsize=(10,12))
edr_df.plot('Time','Temperature',ax=ax[0,0])
edr_df.plot('Time','Pressure',ax=ax[0,1])
edr_df.plot('Time','Potential',ax=ax[1,0])
edr_df.plot('Time','Bond',ax=ax[1,1])
#edr_df.plot('Time','Position Rest.',ax=ax[1,1])
#edr_df.plot('Time','COM Pull En.',ax=ax[1,1])
edr_df.plot('Time','Coulomb (SR)',ax=ax[2,0])
edr_df.plot('Time','Coul. recip.',ax=ax[2,1])


# ### Visualize trajectory

# In[61]:


mda_trr = mda.Universe('nvt.gro','npt.trr')


# In[62]:


# check unique resiude names in system
resnames = np.unique([ r.resname for r in mda_trr.residues ])


# In[63]:


resnames


# In[64]:


mda_view = nglview.show_mdanalysis(mda_trr)
mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation(repr_type='ball+stick',selection='SDS')
mda_view.add_representation(repr_type='ball+stick',selection='NA')
mda_view.add_representation(repr_type='spacefill',selection='AUM',color='yellow')
mda_view.center()


# In[65]:


mda_view


# In[67]:


substrate = mda_trr.select_atoms('resname AUM')


# In[103]:


substrate.masses= ase.data.atomic_masses[ase.data.atomic_numbers['Au']]


# In[105]:


substrtate_com_traj = np.array([substrate.center_of_mass() for ts in mda_trr.trajectory ])


# In[106]:


substrtate_rgyr_traj = np.array([substrate.radius_of_gyration() for ts in mda_trr.trajectory ])


# In[107]:


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d',azim=-30)
ax.plot(*substrtate_com_traj.T)
ax.scatter(*substrtate_com_traj[0,:],color='green')
ax.scatter(*substrtate_com_traj[-1,:],color='red')


# In[109]:


plt.plot(substrtate_rgyr_traj)


# In[119]:


try:
    del mda_trr
except:
    pass
try:
    del mda_view
except:
    pass

