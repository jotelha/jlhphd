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
n_per_nm_sq = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
N = np.round(A_nm*n_per_nm_sq).astype(int).tolist()


# In[10]:
# from jlhpy.utilities.wf.packing.sub_wf_120_gromacs_em import GromacsEnergyMinimizationSubWorkflowGenerator
from jlhpy.utilities.wf.packing.chain_wf_spherical_indenter_passivation import SphericalSurfactantPackingChainWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

# source_project_id = '2020-04-21-intermediate-trial'
project_id = '2020-07-01-passiv-trial'
wfg = SphericalSurfactantPackingChainWorkflowGenerator(
    project_id=project_id, 
    description="Trial runs for dtool from JUWELS to Isilon",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    infile_prefix=prefix,
    machine='juwels_devel',
    parameter_label_key_dict={'n': 'system->surfactant->nmolecules'},
    mode='trial',
    system = { 
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': int(N[-2]),
            'reference_atom': {
                'name': 'NA',
            },
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': int(N[-2]),
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
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        },
        'dtool_push': {
            'dtool_target': 'smb://rz-freiburg-user-share',
            'dtool_config': {
                'DTOOL_SMB_SERVER_NAME_rz-freiburg-user-share': 'localhost'
            },
            'ssh_config': {  # options for ssh port forwarding
                'remote_host':  'tfsish01.public.ads.uni-freiburg.de',
                'remote_port':  445,
                'ssh_host':     'simdata.vm.uni-freiburg.de',  # jump host
                'ssh_user':     'sshclient',
                'ssh_keyfile':  '~/.ssh/sshclient-frrzvm',
            },
        }
    },
    dtool_port_config_key='DTOOL_SMB_SERVER_PORT_rz-freiburg-user-share'
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[20]:
from jlhpy.utilities.wf.packing.chain_wf_spherical_indenter_passivation import IndenterPassivationParametricWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

# n = N[-2]
parameter_values = [{'n': n, 'm': n } for n in N[-4:-2]]
# source_project_id = '2020-04-21-intermediate-trial'
project_id = '2020-07-19-passivation-trial-a'
wfg = IndenterPassivationParametricWorkflowGenerator(
    project_id=project_id, 
    integrate_push=True,
    description="Parametric trial runs for indenter passivation",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    infile_prefix=prefix,
    machine='juwels_devel',
    parameter_label_key_dict={
        'n': 'system->surfactant->nmolecules', 
        'm': 'system->counterion->nmolecules'},
    parameter_values=parameter_values,
    mode='trial',
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
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1,  # pseudo-units
            'nsteps': 100,
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/DATASETS',
            'remote_dataset': None,
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

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
project_id = '2020-04-22-intermediate-trial'
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
            'resname': 'NA',
            'nmolecules': int(N[-2]),
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': int(N[-2]),
            'connector_atom': {
                'index': int(SURFACTANTS["SDS"]["connector_atom_index"])
            },
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
            'natoms': 3873,  # TODO: count automatically
        },
        'solvent': {
            'name': 'H2O',
            'resname': 'SOL',
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
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        }
    }
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[95]: Trial
    
# from jlhpy.utilities.wf.packing.sub_wf_120_gromacs_em import GromacsEnergyMinimizationSubWorkflowGenerator
from jlhpy.utilities.wf.packing.sub_wf_120_gromacs_em import GromacsEnergyMinimizationAnalysisVisualizationSubWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

source_project_id = '2020-04-21-intermediate-trial'
project_id = '2020-05-04-gmx-em-dtool-trial'
wfg = GromacsEnergyMinimizationAnalysisVisualizationSubWorkflowGenerator(
    project_id=project_id, 
    description="Trial runs for dtool from JUWELS to Isilon",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    creation_date="2020-05-04",
    expiration_date="2022-05-04",
    source_project_id=source_project_id,
    infile_prefix=prefix,
    machine='juwels_devel',
    parameter_keys=['system->surfactant->nmolecules'],
    mode='trial',
    #  dtool_target='file://localhost/p/project/chfr13/hoermann1/dtool/DATASETS',
    system = { 
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': int(N[-2]),
            'reference_atom': {
                'name': 'NA',
            },
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': int(N[-2]),
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
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        },
        'dtool_push': {
            'dtool_target': 'smb://rz-freiburg-user-share',
            'dtool_metadata': {
                'expiration_date': '2022-05-04'
            },
            'dtool_config': {
                'DTOOL_SMB_SERVER_NAME_rz-freiburg-user-share': 'loacalhost'
            },
            'ssh_config': {
                'remote_host':  'tfsish01.public.ads.uni-freiburg.de',
                'remote_port':  445,
                'ssh_host':     'simdata.vm.uni-freiburg.de',
                'ssh_user':     'sshclient',
                'ssh_keyfile':  '~/.ssh/sshclient-frrzvm',
            },
        }
    },
    dtool_port_config_key='DTOOL_SMB_SERVER_PORT_rz-freiburg-user-share'
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

# In[200]:
from jlhpy.utilities.wf.packing.sub_wf_170_gromacs_nvt import GromacsNVTEquilibrationSubWorkflowGenerator
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

source_project_id = '2020-04-22-intermediate-trial'
project_id = '2020-04-29-gmx-nvt-trial'

wfg = GromacsNVTEquilibrationSubWorkflowGenerator(
    project_id, 
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    creation_date="2020-05-04",
    expiration_date="2022-05-04",
    source_project_id=source_project_id,
    infile_prefix=prefix,
    machine='juwels_devel',
    parameter_keys=['system->surfactant->nmolecules'],
    mode='trial',
    #  dtool_target='file://localhost/p/project/chfr13/hoermann1/dtool/DATASETS',
    system = { 
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': int(N[-2]),
            'reference_atom': {
                'name': 'NA',
            },
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': int(N[-2]),
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
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        },
        'dtool_push': {
            'dtool_target': 'smb://freiburg-user-share',
            'dtool_metadata': {
                'expiration_date': '2022-05-04'
            },
            'ssh_config': {
                'remote_host':  'tfsish01.public.ads.uni-freiburg.de',
                'remote_port':  445,
                'ssh_host':     'simdata.vm.uni-freiburg.de',
                'ssh_user':     'sshclient',
                'ssh_keyfile':  '~/.ssh/sshclient-frrzvm',
            },
        }
    },
    dtool_port_config_key='DTOOL_SMB_SERVER_PORT_rz-freiburg-user-share'
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[220]:
from jlhpy.utilities.wf.packing.chain_wf_spherical_indenter_passivation import IndenterPassivationChainWorkflow
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

wfg = IndenterPassivationChainWorkflow(
    project_id='2020-04-23-indenter-passivation-trial', 
    infile_prefix=prefix,
    machine='juwels',
    parameter_keys=['system->surfactant->nmolecules'],
    mode='trial',
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': 'https://orcid.org/0000-0001-5867-695X'
    }],
    creation_date="2020-04-23",
    expiration_date="2022-04-23",
    system = { 
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': int(N[6]),
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': int(N[6]),
            'connector_atom': {
                'index': int(SURFACTANTS["SDS"]["connector_atom_index"])
            },
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
            'natoms': 3873,  # TODO: count automatically
        },
        'solvent': {
            'name': 'H2O',
            'resname': 'SOL',
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
            'spring_constant': 1000,  # pseudo-units
            'rate': 0.1  # pseudo-units
        }
    }
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()
