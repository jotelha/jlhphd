i#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:52:29 2020

@author: jotelha
"""
        

# In[20]:
import os, os.path

# FireWorks functionality 
from fireworks.utilities.dagflow import DAGFlow, plot_wf
from fireworks import Firework, LaunchPad, Workflow
from fireworks.utilities.filepad import FilePad


# prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'
prefix = '/home/jotelha/git/jlhphd'
work_prefix = '/home/jotelha/tmp/20200329_fw/'
os.chdir(work_prefix)

# the FireWorks LaunchPad
lp = LaunchPad.auto_load() #Define the server and database
# FilePad behaves analogous to LaunchPad
fp = FilePad.auto_load()

# In[25]:
import numpy as np
R = 26.3906
A_Ang = 4*np.pi*R**2 # area in Ansgtrom
A_nm = A_Ang / 10**2
n_per_nm_sq = np.array([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
N = np.round(A_nm*n_per_nm_sq).astype(int)


# In[80]: gmx prep
    
from jlhpy.utilities.wf.packing.sub_wf_gromacs_prep import GromacsPrepSubWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

source_project_id = '2020-04-14-packmol-trial'
project_id = '2020-04-14-gmx-prep-trial'
wfg = GromacsPrepSubWorkflowGenerator(
    project_id, 
    source_project_id=source_project_id,
    machine='juwels_devel',
    parameter_keys=['system->surfactant->nmolecules'],
    system = { 
        # 'packing' : {
        #     'surfactant_indenter': {
        #         'outer_atom_index': SURFACTANTS["SDS"]["head_atom_index"],
        #         'inner_atom_index': SURFACTANTS["SDS"]["tail_atom_index"],
        #         'tolerance': TOLERANCE
        #     },
        # },
        'counterion': {
            'name': 'NA',
        },
        'surfactant': {
            'name': 'SDS',
            'nmolecules': 100,
            'connector_atom': {
                'index': int(SURFACTANTS["SDS"]["connector_atom_index"])
            },
        },
    })
# fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()


# In[100]: gmx pull prep
    
from jlhpy.utilities.wf.packing.sub_wf_gromacs_pull_prep import GromacsPullPrepSubWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

source_project_id = '2020-04-15-intermediate-trial'
project_id = '2020-04-15-gmx-pull-prep-trial'

wfg = GromacsPullPrepSubWorkflowGenerator(
    project_id, 
    source_project_id=source_project_id,
    infile_prefix=prefix,
    machine='juwels_devel',
    parameter_keys=['system->surfactant->nmolecules'],
    mode='trial',
    system = { 
        'packing' : {
            'surfactant_indenter': {
                'outer_atom_index': SURFACTANTS["SDS"]["head_atom_index"],
                'inner_atom_index': SURFACTANTS["SDS"]["tail_atom_index"],
                'tolerance': TOLERANCE
            },
        },
        'counterion': {
            'name': 'NA',
            'nmolecules': int(N[-2]),
        },
        'surfactant': {
            'name': 'SDS',
            'nmolecules': int(N[-2]),
            'connector_atom': {
                'index': int(SURFACTANTS["SDS"]["connector_atom_index"])
            },
        },
        'substrate': {
            'name': 'AUM',
            'natoms': 3873, 
        }
    },
    step_specific={
        'pulling': {
            'pull_atom_name': SURFACTANTS["SDS"]["tail_atom_name"],
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        }
    })
fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[100]: gmx pull
    
from jlhpy.utilities.wf.packing.sub_wf_gromacs_pull import GromacsPullSubWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

source_project_id = '2020-04-15-intermediate-trial'
project_id = '2020-04-21-gmx-pull-trial'

wfg = GromacsPullSubWorkflowGenerator(
    project_id, 
    source_project_id=source_project_id,
    #infile_prefix=prefix,
    machine='juwels_devel',
    parameter_keys=['system->surfactant->nmolecules'],
    mode='trial',
    system = { 
        'packing' : {
            'surfactant_indenter': {
                'outer_atom_index': SURFACTANTS["SDS"]["head_atom_index"],
                'inner_atom_index': SURFACTANTS["SDS"]["tail_atom_index"],
                'tolerance': TOLERANCE
            },
        },
        'counterion': {
            'name': 'NA',
            'nmolecules': int(N[-2]),
        },
        'surfactant': {
            'name': 'SDS',
            'nmolecules': int(N[-2]),
            'connector_atom': {
                'index': int(SURFACTANTS["SDS"]["connector_atom_index"])
            },
        },
        'substrate': {
            'name': 'AUM',
            'natoms': 3873, 
        }
    },
    step_specific={
        'pulling': {
            'pull_atom_name': SURFACTANTS["SDS"]["tail_atom_name"],
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        }
    })
# fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()



# In[90]: Trial
    
from jlhpy.utilities.wf.packing.chain_wf_spherical_indenter_passivation import IntermediateTestingWorkflow
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

# source_project_id = '2020-04-14-gmx-prep-trial'
project_id = '2020-04-21-intermediate-trial'
wfg = IntermediateTestingWorkflow(
    project_id, 
    #source_project_id=source_project_id,
    infile_prefix=prefix,
    machine='juwels_devel',
    parameter_keys=['system->surfactant->nmolecules'],
    mode='trial',
    system = { 
        'counterion': {
            'name': 'NA',
            'nmolecules': int(N[-2]),
        },
        'surfactant': {
            'name': 'SDS',
            'nmolecules': int(N[-2]),
            'connector_atom': {
                'index': int(SURFACTANTS["SDS"]["connector_atom_index"])
            },
        },
        'substrate': {
            'name': 'AUM',
            'natoms': 3873, 
        }
    },
    step_specific={
        'packing' : {
            'surfactant_indenter': {
                'outer_atom_index': SURFACTANTS["SDS"]["head_atom_index"],
                'inner_atom_index': SURFACTANTS["SDS"]["tail_atom_index"],
                'tolerance': TOLERANCE
            },
        },
        'pulling': {
            'pull_atom_name': SURFACTANTS["SDS"]["tail_atom_name"],
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        }
    }
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()


# In[100]: gmx pull
    
from jlhpy.utilities.wf.packing.sub_wf_150_gromacs_solvate import GromacsSolvateSubWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

source_project_id = '2020-04-21-intermediate-trial'
project_id = '2020-04-21-gmx-solvate-trial'

wfg = GromacsSolvateSubWorkflowGenerator(
    project_id, 
    source_project_id=source_project_id,
    machine='juwels_devel',
    parameter_keys=['system->surfactant->nmolecules'],
    mode='trial',
    system = { 
        'counterion': {
            'name': 'NA',
            'nmolecules': int(N[-2]),
        },
        'surfactant': {
            'name': 'SDS',
            'nmolecules': int(N[-2]),
            'connector_atom': {
                'index': int(SURFACTANTS["SDS"]["connector_atom_index"])
            },
        },
        'substrate': {
            'name': 'AUM',
            'natoms': 3873, 
        }
    },
    step_specific={
        'packing' : {
            'surfactant_indenter': {
                'outer_atom_index': SURFACTANTS["SDS"]["head_atom_index"],
                'inner_atom_index': SURFACTANTS["SDS"]["tail_atom_index"],
                'tolerance': TOLERANCE
            },
        },
        'pulling': {
            'pull_atom_name': SURFACTANTS["SDS"]["tail_atom_name"],
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        }
    })
# fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[120]:

from jlhpy.utilities.wf.packing.chain_wf_spherical_indenter_passivation import GromacsPackingMinimizationChainWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

source_project_id = '2020-04-21-intermediate-trial'
project_id = '2020-04-21-gmx-chain-wf-trial'

wfg = GromacsPackingMinimizationChainWorkflowGenerator(
    project_id, 
    source_project_id=source_project_id,
    infile_prefix=prefix,
    machine='juwels_devel',
    parameter_keys=['system->surfactant->nmolecules'],
    mode='trial',
    system = { 
        'counterion': {
            'name': 'NA',
            'nmolecules': int(N[-2]),
        },
        'surfactant': {
            'name': 'SDS',
            'nmolecules': int(N[-2]),
            'connector_atom': {
                'index': int(SURFACTANTS["SDS"]["connector_atom_index"])
            },
        },
        'substrate': {
            'name': 'AUM',
            'natoms': 3873, 
        }
    },
    step_specific={
        'packing' : {
            'surfactant_indenter': {
                'outer_atom_index': SURFACTANTS["SDS"]["head_atom_index"],
                'inner_atom_index': SURFACTANTS["SDS"]["tail_atom_index"],
                'tolerance': TOLERANCE
            },
        },
        'pulling': {
            'pull_atom_name': SURFACTANTS["SDS"]["tail_atom_name"],
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        }
    })
fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()
