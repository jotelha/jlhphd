#!/usr/bin/env python
# coding: utf-8

# # Prepare AFM tip solvation

# This notebook demonstrates deposition of an SDS adsorption layer on a non-spherical AFM tip model.

# ## Initialization

# ### IPython magic

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '3')


# ### Imports

# In[ ]:


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
from jlhfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask
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
from jlhfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask # custom Fireworks additions
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

# In[ ]:


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ParmEd needs to know the GROMACS topology folder, usually get this from 
# envionment variable `GMXLIB`:

# ### Function decfinitions

# In[ ]:


def find_undeclared_variables(infile):
    """identify all variables evaluated in a jinja 2 template file"""
    env = jinja2.Environment()
    with open(infile) as template_file:
        parsed = env.parse(template_file.read())

    undefined = jinja2.meta.find_undeclared_variables(parsed)
    return undefined


# In[ ]:


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


# In[ ]:


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


# In[ ]:


def memuse():
    """Quick overview on memory usage of objects in Jupyter notebook"""
    # https://stackoverflow.com/questions/40993626/list-memory-usage-in-ipython-and-jupyter
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    return sorted([(x, sys.getsizeof(globals().get(x))) for x in dir(sys.modules['__main__']) if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# ### Global settings

# In[ ]:


os.environ['GMXLIB']


# In[ ]:


pmd.gromacs.GROMACS_TOPDIR = os.environ['GMXLIB']


# In[ ]:


prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'


# In[ ]:


os.chdir(prefix)


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

# In[35]:


infile = os.path.join(prefix,'indenter_reres.pdb')


# In[36]:


atoms = ase.io.read(infile,format='proteindatabank')


# In[37]:


atoms


# ### Display with ASE view

# In[19]:


v = view(atoms,viewer='ngl')
v.view._remote_call("setSize", target="Widget", args=["400px", "400px"])
v.view.center()
v.view.background='#ffc'
v


# ### Get the bounding sphere around point set

# In[38]:


S = atoms.get_positions()


# In[39]:


C, R_sq = miniball.get_bounding_ball(S)


# In[40]:


C # sphere center


# In[41]:


R = np.sqrt(R_sq)


# In[42]:


R # sphere radius


# In[43]:


xmin = atoms.get_positions().min(axis=0)
xmax = atoms.get_positions().max(axis=0)


# In[44]:


xmin


# In[45]:


del S


# ### Plot 2d projections of point set and bounding sphere

# In[46]:


# plot side views with sphere projections
plot_side_views_with_spheres(atoms,C,R)


# ### Plot 3d point set and bounding sphere

# In[39]:


# bounding sphere surface
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
us = np.array([
    np.outer(np.cos(u), np.sin(v)),
    np.outer(np.sin(u), np.sin(v)), 
    np.outer(np.ones(np.size(u)), np.cos(v))])
bs = C + R*us.T


# In[40]:


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

# In[41]:


tol = 2 # Ang


# ### Read single surfactant molecule PDB with ParmEd

# Utilize parmed to read pbd files ASE has difficulties to decipher.

# In[42]:


infile = os.path.join(prefix,'1_SDS.pdb')


# In[43]:


surfactant_pmd = pmd.load_file(infile)


# In[44]:


surfactant_pmd.atoms[-1].atomic_number


# ### Convert ParmEd structure to ASE atoms

# In[45]:


surfactant_ase = ase.Atoms(
    numbers=[1 if a.atomic_number == 0 else a.atomic_number for a in surfactant_pmd.atoms],
    positions=surfactant_pmd.get_coordinates(0))


# ### Get bounding sphere of single surfactant molecule

# In[46]:


C_surfactant, R_sq_surfactant = miniball.get_bounding_ball(surfactant_ase.get_positions())


# In[47]:


C_surfactant


# In[48]:


R_surfactant = np.sqrt(R_sq_surfactant)


# In[49]:


R_surfactant


# In[50]:


C_surfactant


# In[51]:


surfactant_ase[:5][1]


# ### Estimate constraint sphere radii

# In[52]:


R_OSL = np.linalg.norm(C_surfactant - surfactant_ase[1].position)


# In[53]:


R_OSL


# In[54]:


d_head = R_surfactant - R_OSL # roughly: diameter of head group


# In[55]:


R_inner = R + tol # place surfactant molecules outside of this sphere


# In[56]:


R_inner_constraint = R + tol + d_head # place surfactant tail hydrocarbon within this sphere


# In[57]:


R_outer_constraint = R + 2*R_surfactant + tol # place head group sulfur outside this sphere


# In[58]:


R_outer = R + 2*R_surfactant + 2*tol # place suractant molecules within this sphere


# In[59]:


rr = [R,R_inner,R_inner_constraint,R_outer_constraint,R_outer]


# In[60]:


cc = [C]*5


# ### Show 2d projections of geometrical constraints around AFM tip model

# In[61]:


plot_side_views_with_spheres(atoms,cc,rr,figsize=(20,8))
plt.show()


# ## Packing the surfactant film

# ### Identify placeholders in jinja2 template

# The template looks like this:

# In[424]:


with open(os.path.join(infile_prefix,'surfactants_on_sphere.inp'),'r') as f:
    print(f.read())


# In[425]:


# get all placholders in template
template_file = os.path.join(infile_prefix,'surfactants_on_sphere.inp')


# In[426]:


v = find_undeclared_variables(template_file)


# In[427]:


v # we want to fill in these placeholder variables


# ### System and constraint parameters

# In[428]:


surfactant = 'SDS'
counterion = 'NA'
tolerance = 2 # Ang
sfN = 200


# In[429]:


l_surfactant = 2*R_surfactant


# In[430]:


# head atom to be geometrically constrained
surfactant_head_bool_ndx = np.array([ a.name == 'S' for a in surfactant_pmd.atoms ],dtype=bool)


# In[431]:


# tail atom to be geometrically constrained
surfactant_tail_bool_ndx = np.array([ a.name == 'C12' for a in surfactant_pmd.atoms ],dtype=bool)


# In[432]:


head_atom_number = surfactant_head_ndx = np.argwhere(surfactant_head_bool_ndx)[0,0]


# In[433]:


tail_atom_number = surfactant_tail_ndx = np.argwhere(surfactant_tail_bool_ndx)[0,0]


# In[434]:


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


# In[435]:


packmol_script_context # context generated from system and constraint settings


# ### Fill a packmol input script template with jinja2

# In[436]:


env = jinja2.Environment()


# In[437]:


template = jinja2.Template(open(template_file).read())


# In[438]:


rendered = template.render(**packmol_script_context)


# In[439]:


rendered_file = os.path.join(prefix,'rendered.inp')


# In[440]:


with open(rendered_file,'w') as f:
    f.write(rendered)


# That's the rendered packmol input file:

# In[441]:


print(rendered)


# ### Fail running packmol once

# In[67]:


packmol = subprocess.Popen(['packmol'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=prefix, encoding='utf-8')


# In[68]:


outs, errs = packmol.communicate(input=rendered)


# In[69]:


print(errs) # error with input from PIPE


# ### Read packmol input from file to avoid obscure Fortran error

# In[445]:


packmol = subprocess.Popen(['packmol'],
        stdin=open(rendered_file,'r'),stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=prefix, encoding='utf-8')


# In[446]:


outs, errs = packmol.communicate(input=rendered)


# In[447]:


print(outs)


# In[448]:


with open('packmol.log','w') as f:
    f.write(outs)


# ### Inspect packed systems

# In[450]:


packmol_pdb = '200_SDS_on_50_Ang_AFM_tip_model_packmol.pdb'


# In[451]:


infile = os.path.join(prefix, packmol_pdb)


# In[452]:


surfactant_shell_pmd = pmd.load_file(infile)


# In[453]:


# with ParmEd and nglview we get automatic bond guessing
pmd_view = nglview.show_parmed(surfactant_shell_pmd)
pmd_view.clear_representations()
pmd_view.background = 'white'
pmd_view.add_representation('ball+stick')
pmd_view


# In[454]:


surfactant_shell_ase = ase.Atoms(
    numbers=[1 if a.atomic_number == 0 else a.atomic_number for a in surfactant_shell_pmd.atoms],
    positions=surfactant_shell_pmd.get_coordinates(0))


# In[455]:


# with ASE, we get no bonds at all
ase_view = nglview.show_ase(surfactant_shell_ase)
ase_view.clear_representations()
ase_view.background = 'white'
ase_view.add_representation('ball+stick')
ase_view


# Get bounding sphere again and display AFM tip bounding spphere as well as surfactant layer bounding sphere

# In[399]:


C_shell, R_sq_shell = miniball.get_bounding_ball(surfactant_shell_ase.get_positions())


# In[400]:


C_shell


# In[401]:


R_shell = np.sqrt(R_sq_shell)


# In[402]:


R_shell


# In[403]:


plot_side_views_with_spheres(surfactant_shell_ase,[C,C_shell],[R,R_shell])


# In[404]:


surfactant_shell_pmd


# ### Batch processing: Parametric jobs

# 
# #### Generate parameter sets

# In[47]:


R # Angstrom


# In[48]:


A_Ang = 4*np.pi*R**2 # area in Ansgtrom


# In[49]:


A_nm = A_Ang / 10**2


# In[50]:


A_nm


# In[51]:


n_per_nm_sq = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm


# In[52]:


N = np.round(A_nm*n_per_nm_sq).astype(int)


# In[53]:


N # molecule numbers corresponding to surface concentrations


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


# In[159]:


infile_prefix = os.path.join(prefix,'packmol_infiles')


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


# In[144]:


# settings can be overridden
for n in N:
    packmol_script_context = {
        'header':        '{:s} packing SDS around AFM probe model'.format(project_id),
        'system_name':   '{:d}_SDS_on_50_Ang_AFM_tip_model'.format(n),
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
            C,R_inner_constraint,R_outer_constraint, n, 
            tail_atom_number, head_atom_number, surfactant, counterion, tolerance))


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

# In[81]:


gromacs.config.logfilename


# In[82]:


gromacs.environment.flags


# In[83]:


# if true, then stdout and stderr are returned as strings by gromacs wrapper commands
gromacs.environment.flags['capture_output'] = False


# In[84]:


print(gromacs.release())


# In[85]:


prefix


# In[95]:


system = '200_SDS_on_50_Ang_AFM_tip_model'
pdb = system + '.pdb'
gro = system + '.gro'
top = system + '.top'
posre = system + '.posre.itp'


# ### Tidy up packmol's non-standard pdb

# In[55]:


# Remove any chain ID from pdb and tidy up
pdb_chain = subprocess.Popen(['pdb_chain',],
        stdin=open(packmol_pdb,'r'),stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=prefix, encoding='utf-8')
pdb_tidy = subprocess.Popen(['pdb_tidy',],
        stdin=pdb_chain.stdout,stdout=open(pdb,'w'), stderr=subprocess.PIPE,
        cwd=prefix, encoding='utf-8')


# ### Generate Gromacs .gro and .top

# In[418]:


rc,out,err=gromacs.pdb2gmx(
    f=pdb,o=gro,p=top,i=posre,ff='charmm36',water='tip3p',
    stdout=False,stderr=False)


# In[419]:


print(out)


# ### Set simulation box size around system

# In[422]:


gro_boxed = system + '_boxed.gro'


# In[423]:


rc,out,err=gromacs.editconf(
    f=gro,o=gro_boxed,d=2.0,bt='cubic',
    stdout=False,stderr=False)


# In[599]:


print(out)


# ### Batch processing

# In[58]:


machine = 'juwels_devel'


# In[59]:


parametric_dimension_labels = ['nmolecules']


# In[103]:


parametric_dimensions = [ {
    'nmolecules': N } ]


# In[61]:


# for testing
parametric_dimensions = [ {
    'nmolecules': [N[0]] } ]


# In[105]:


parameter_sets = list( 
    itertools.chain(*[ 
            itertools.product(*list(
                    p.values())) for p in parametric_dimensions ]) )

parameter_dict_sets = [ dict(zip(parametric_dimension_labels,s)) for s in parameter_sets ]


# In[106]:


source_project_id = 'juwels-packmol-2020-03-09'
project_id = 'juwels-gromacs-prep-2020-03-11'


# In[107]:


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
        'coordinate_file': 'out.gro',
        'topology_file':   'out.top',
        'restraint_file':  'out.posre.itp'}
    
    fts_gmx_pdb2gro = [ CmdTask(
        cmd='gmx',
        opt=['pdb2gmx',
             '-f', 'in.pdb',
             '-o', 'out.gro',
             '-p', 'out.top',
             '-i', 'out.posre.itp', 
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
        'topology_file':   'in.top',
        'restraint_file':  'in.posre.itp'}
    files_out = {
        'coordinate_file': 'out.gro',
        'topology_file':   'in.top',
        'restraint_file':  'in.posre.itp'}
    
    fts_gmx_editconf = [ CmdTask(
        cmd='gmx',
        opt=['editconf',
             '-f', 'in.gro',
             '-o', 'out.gro',
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


# In[108]:


wf.as_dict()


# In[109]:


lp.add_wf(wf)


# In[111]:


wf.to_file('wf-{:s}.yaml'.format(project_id))


# ### Inspect results

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


# In[98]:


system_selection = gro_list
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


# ## Energy minimization with restraints

# Just to be safe, relax the system a little with positional constraints applied to all ions.

# ### Compile system

# In[308]:


os.getcwd()


# In[310]:


em_mdp = gromacs.fileformats.MDP('em.mdp.template')
# no change
em_mdp.write('em.mdp')


# In[311]:


gmx_grompp = gromacs.grompp.Popen(
    f='em.mdp',c=gro,r=gro,o='em.tpr',p=top,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)

out = gmx_grompp.stdout.read()
err = gmx_grompp.stderr.read()


# In[312]:


print(err)


# In[313]:


print(out)


# ### Run energy minimization

# In[314]:


gmx_mdrun = gromacs.mdrun.Popen(
    deffnm='em',v=True,
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)


# In[316]:


for line in gmx_mdrun.stdout: 
    print(line.decode(), end='')


# In[317]:


out = gmx_mdrun.stdout.read()
err = gmx_mdrun.stderr.read()


# In[318]:


print(err)


# In[305]:


print(out)


# ### Energy minimization analysis

# In[319]:


em_file = 'em.edr'


# In[320]:


em_df = panedr.edr_to_df(em_file)


# In[321]:


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


# In[324]:


mda_trr = mda.Universe(gro,'em.trr')

mda_view = nglview.show_mdanalysis(mda_trr)

mda_view.clear_representations()
mda_view.background = 'white'
mda_view.add_representation('ball+stick')
mda_view


# ## Pulling

# Utilize harmonic pulling to attach surfactants to substrate closely.

# ### Create index groups for pulling

# In[325]:


#pdb = '200_SDS_on_50_Ang_AFM_tip_model.pdb'
gro = 'em.gro'
top = 'sys.top'
ndx = 'standard.ndx'


# In[326]:


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


# In[328]:


tail_atom_ndx


# ### Generate standard index file for system

# In[329]:


gmx_make_ndx = gromacs.make_ndx.Popen(
    f=gro,o=ndx,
    input='q',
    stdout=subprocess.PIPE,stderr=subprocess.PIPE)


# In[330]:


out_str, err_str = gmx_make_ndx.communicate()


# In[331]:


print(out_str)


# In[332]:


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


# In[456]:


print(out)


# ## Pulling analysis

# ### Global observables

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

