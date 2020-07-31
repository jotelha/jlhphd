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

# In[25]:
import numpy as np
R = 26.3906 # indenter radius
A_Ang = 4*np.pi*R**2 # area in Ansgtrom
A_nm = A_Ang / 10**2
n_per_nm_sq = np.arange(0.25, 6.25, 0.25)
# n_per_nm_sq = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
# n_per_nm_sq = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
N = np.round(A_nm*n_per_nm_sq).astype(int).tolist()

# In[20]:
    
# SDS on Au(111)
from jlhpy.utilities.wf.packing.chain_wf_spherical_indenter_passivation import IndenterPassivationParametricWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

parameter_values = [{'n': n, 'm': n } for n in N]
# parameter_values = [{'n': n, 'm': n } for n in [N[6]]]
project_id = '2020-07-29-sds-on-au-111-indenter-passivation'

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': parameter_values})
wfg = IndenterPassivationParametricWorkflowGenerator(
    project_id=project_id, 
    integrate_push=True,
    description="Parametric trial runs for SDS on Au(111) indenter passivation",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    infile_prefix=prefix,
    machine='juwels',
    parameter_label_key_dict={
        'n': 'system->surfactant->nmolecules', 
        'm': 'system->counterion->nmolecules'},
    parameter_values=parameter_values,
    mode='production',
    system = { 
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': None,
            'reference_atom': {
                'name': 'NA',
            },
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': None,
            'connector_atom': {
                'index': int(SURFACTANTS["SDS"]["connector_atom_index"])
            },
            'head_atom': {
                'name': 'S',
            },
            'tail_atom': {
                'name': 'C12',
            },
            
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
            'natoms': 3873,  # TODO: count automatically
            'reference_atom': {
                'name': 'AU',
            },
        },
        'solvent': {
            'name': 'H2O',
            'resname': 'SOL',
            'reference_atom': {
                'name': 'OW',
            },
            # 'natoms':  # TODO: count automatically
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
            'spring_constant': 10000,  # pseudo-units
            'rate': -0.1,  # pseudo-units
            'nsteps': 1000,
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/DATASETS',
            'remote_dataset': None,
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[20]:

# CTAB on Au(111)
from jlhpy.utilities.wf.packing.chain_wf_spherical_indenter_passivation import IndenterPassivationParametricWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS


parameter_values = [{'n': n, 'm': n } for n in N]
# parameter_values = [{'n': n, 'm': n } for n in [N[6]]]
project_id = '2020-07-29-ctab-on-au-111-indenter-passivation'

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': parameter_values})
wfg = IndenterPassivationParametricWorkflowGenerator(
    project_id=project_id, 
    integrate_push=True,
    description="Parametric trial runs for CTAB on Au(111) indenter passivation",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    infile_prefix=prefix,
    machine='juwels',
    parameter_label_key_dict={
        'n': 'system->surfactant->nmolecules', 
        'm': 'system->counterion->nmolecules'},
    parameter_values=parameter_values,
    mode='production',
    system = { 
        'counterion': {
            'name': 'BR',
            'resname': 'BR',
            'nmolecules': None,
            'reference_atom': {
                'name': 'BR',
            },
        },
        'surfactant': {
            'name': 'CTAB',
            'resname': 'CTAB',
            'nmolecules': None,
            'connector_atom': {
                'index': int(SURFACTANTS["CTAB"]["connector_atom_index"])
            },
            'head_atom': {
                'name': 'N1',
            },
            'tail_atom': {
                'name': 'C1',
            },
            
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
            'natoms': 3873,  # TODO: count automatically
            'reference_atom': {
                'name': 'AU',
            },
        },
        'solvent': {
            'name': 'H2O',
            'resname': 'SOL',
            'reference_atom': {
                'name': 'OW',
            },
            # 'natoms':  # TODO: count automatically
        }
    },
    step_specific={
        'packing' : {
            'surfactant_indenter': {
                'outer_atom_index': SURFACTANTS["CTAB"]["head_atom_index"],
                'inner_atom_index': SURFACTANTS["CTAB"]["tail_atom_index"],
                'tolerance': TOLERANCE
            },
        },
        'pulling': {
            'pull_atom_name': SURFACTANTS["CTAB"]["tail_atom_name"],
            'spring_constant': 10000,  # pseudo-units
            'rate': -0.1,  # pseudo-units
            'nsteps': 1000,
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/DATASETS',
            'remote_dataset': None,
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()