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

# plotting defaults
visual_style = {}

# generic plotting defaults
visual_style["layout"] = 'kamada_kawai'
visual_style["bbox"] = (1600, 1200)
visual_style["margin"] = [400, 100, 400, 200]

visual_style["vertex_label_angle"] = -3.14/4.0
visual_style["vertex_size"] = 8
visual_style["vertex_shape"] = 'rectangle'
visual_style["vertex_label_size"] = 10
visual_style["vertex_label_dist"] = 4

# edge defaults
visual_style["edge_color"] = 'black'
visual_style["edge_width"] = 1
visual_style["edge_arrow_size"] = 1
visual_style["edge_arrow_width"] = 1
visual_style["edge_label_size"] = 8

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
# R = 26.3906 # indenter radius
a = 150.0 # approximate substrate measures

A_Ang = a**2 # area in Ansgtrom
A_nm = A_Ang / 10**2
n_per_nm_sq = np.arange(0.25, 6.25, 0.25)
# n_per_nm_sq = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
# n_per_nm_sq = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
N = np.round(A_nm*n_per_nm_sq).astype(int).tolist()


# c, c, probe uuid, substrate (monolayer) uuid
probe_on_monolayer_input_datasets = [
     (0.0024961896078710614,
      0.002513703150711859,
      '43a09eb9-a27b-42fb-a424-66d2cdbdf605',
      '909d8ac8-e6d8-4bea-abb0-fca2dd57b1d6'),
     (0.010029333245910514,
      0.010054812602847437,
      'a72b124b-c5aa-43d8-900b-f6b6ddc05d39',
      'f52d0b2b-fec8-40ea-bce8-aad80edfbb11'),
     (0.012525522853781576,
      0.012454256519436031,
      '02c578b1-b331-42cf-8aef-4e3dcd0b4c77',
      'd541a286-ccf8-4dd4-80e7-eec86e17eaaa'),
     (0.017562476883949966,
      0.01748166282085975,
      '86d2a465-61b8-4f1d-b13b-912c8f1f814b',
      '61ac9afb-e498-47a0-bb83-f5bd0e14bd59'),
     (0.020058666491821028,
      0.01999536597157161,
      '7e128862-4221-4554-bc27-22812a6047ae',
      'f83d4c77-3944-4cb0-bc2f-b93096434b41'),
     (0.02255485609969209,
      0.022509069122283468,
      '1bc8bb4a-f4cf-4e4f-96ee-208b01bc3d02',
      '6d5fe574-3359-4580-ae2d-eeda9ec5b926')]


probe_on_hemicylinders_input_datasets = [
     (0.0024961896078710614,
      0.002513703150711859,
      '43a09eb9-a27b-42fb-a424-66d2cdbdf605',
      'd116b98a-9d40-4179-87b2-49ce41a79cf4'),
     (0.004992379215742123,
      0.005027406301423718,
      '9835cce9-b7d0-4e1f-9bdf-fd9767fea72c',
      '8410c4a5-e8f8-4cac-a767-ff95490caaeb'),
     (0.010029333245910514,
      0.010054812602847437,
      'a72b124b-c5aa-43d8-900b-f6b6ddc05d39',
      'fd8ae366-4d95-4e15-913a-04ac4749b7a7'),
     (0.012525522853781576,
      0.012454256519436031,
      '02c578b1-b331-42cf-8aef-4e3dcd0b4c77',
      'd3f0d381-9df1-4d8e-a961-b8f4a92dc983'),
     (0.015066287276078906,
      0.01496795967014789,
      '974b41b2-de1c-421c-897b-7e091facff3a',
      '4124b74d-b448-4a49-98ff-f06e7fa1677a'),
     (0.017562476883949966,
      0.01748166282085975,
      '86d2a465-61b8-4f1d-b13b-912c8f1f814b',
      '29413408-6222-4e1f-ae73-8275a79ffce6'),
     (0.020058666491821028,
      0.01999536597157161,
      '7e128862-4221-4554-bc27-22812a6047ae',
      '8eec7f62-fa43-44b1-b389-940b93c569b5')]
# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrate
from jlhpy.utilities.wf.building_blocks.sub_wf_count_components import CountComponents

from jlhpy.utilities.wf.probe_on_substrate.sub_wf_010_merge import MergeSubstrateAndProbeSystems
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

project_id = '2020-12-01-sds-on-au-111-probe-and-substrate'

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# parameter_values = [{'n': n, 'm': n, 's': s } for n in N for s in ['monolayer','hemicylinders']][10:11]

# In[25]
wfg = ProbeOnSubstrate(
    project_id=project_id,
    
    files_in_info={
        'substrate_data_file': {  #  506 SDS
            'query': {'uuid': '6d5fe574-3359-4580-ae2d-eeda9ec5b926'},
            'file_name': 'default.gro',
            'metadata_dtool_source_key': 'system->substrate',
            'metadata_fw_dest_key': 'metadata->system->substrate',
            'metadata_fw_source_key': 'metadata->system->substrate',
        },
        'probe_data_file': {  # 197
            'query': {'uuid': '1bc8bb4a-f4cf-4e4f-96ee-208b01bc3d02'},
            'file_name': 'default.gro',
            'metadata_dtool_source_key': 'system->indenter',
            'metadata_fw_dest_key': 'metadata->system->indenter',
            'metadata_fw_source_key': 'metadata->system->indenter',
        }
    },
    #source_project_id="2020-11-25-au-111-150x150x150-fcc-substrate-creation",
    #source_step='FCCSubstrateCreationChainWorkflowGenerator:LAMMPSEquilibrationNPTWorkflowGenerator:push_dtool',
    #metadata_dtool_source_key='system->substrate',
    #metadata_fw_dest_key='metadata->system->substrate',
    #metadata_fw_source_key='metadata->system->substrate',

    integrate_push=True,
    description="SDS on Au(111) substrate and probe trial",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    infile_prefix=prefix,
    machine='juwels',
    mode='trial',
    #parameter_label_key_dict={
    #    'n': 'system->surfactant->nmolecules',
    #    'm': 'system->counterion->nmolecules',
    #    's': 'system->surfactant->aggregates->shape'},
    #parameter_values=parameter_values,
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
                'index': 2,
            },
            'head_atom': {
                'name': 'S',
                'index': 1,
            },
            'tail_atom': {
                'name': 'C12',
                'index': 39,
            },
            'aggregates': {
                'shape': None,
            }
        },
        'indenter': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
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
            'height': 180.0,
        }
    },
    step_specific={
        'merge': {
            'tol': 2.0,
            'z_dist': 50.0,
            'x_shift': 15.0,
            'y_shift': 0.0,
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/TRIAL_DATASETS',
            'remote_dataset': None,
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[25]

project_id = '2020-12-01-count-components-trial'

wfg = CountComponents(
    project_id=project_id,
    
    files_in_info={
        'data_file': {  # 2020-10-13-ctab-on-au-111-substrate-passivation
            'query': {'uuid': '91a835f3-5b81-4bfa-bc16-c125fc905a8b'},
            'file_name': 'out.pdb',
        },
    },

    integrate_push=True,
    description="SDS on Au(111) substrate and probe count components trial",
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
                'index': 2,
            },
            'head_atom': {
                'name': 'S',
                'index': 1,
            },
            'tail_atom': {
                'name': 'C12',
                'index': 39,
            },
            'aggregates': {
                'shape': None,
            }
        },
        'indenter': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
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
            'height': 180.0,
            # 'natoms':  # TODO: count automatically
        }
    },
    step_specific={
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/TRIAL_DATASETS',
            'remote_dataset': None,
        }
    },
)

wf = wfg.build_wf()

# In[25]
wf_list = []
for c1, c2, probe_uuid, substrate_uuid in probe_on_monolayer_input_datasets:
    wfg = MergeSubstrateAndProbeSystems(
        project_id=project_id,
        
        files_in_info={
            'substrate_data_file': {
                'query': {'uuid': substrate_uuid },
                'file_name': 'default.gro',
                'metadata_dtool_source_key': 'system->substrate',
                'metadata_fw_dest_key': 'metadata->system->substrate',
                'metadata_fw_source_key': 'metadata->system->substrate',
            },
            'probe_data_file': {
                'query': {'uuid': probe_uuid},
                'file_name': 'default.gro',
                'metadata_dtool_source_key': 'system->indenter',
                'metadata_fw_dest_key': 'metadata->system->indenter',
                'metadata_fw_source_key': 'metadata->system->indenter',
            }
        },
       
        integrate_push=True,
        description="SDS on Au(111) substrate and probe trial",
        owners=[{
            'name': 'Johannes Laurin Hörmann',
            'email': 'johannes.hoermann@imtek.uni-freiburg.de',
            'username': 'fr_jh1130',
            'orcid': '0000-0001-5867-695X'
        }],
        infile_prefix=prefix,
        machine='juwels_devel',
        mode='trial',
        #parameter_label_key_dict={
        #    'n': 'system->surfactant->nmolecules',
        #    'm': 'system->counterion->nmolecules',
        #    's': 'system->surfactant->aggregates->shape'},
        #parameter_values=parameter_values,
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
                    'index': 2,
                },
                'head_atom': {
                    'name': 'S',
                    'index': 1,
                },
                'tail_atom': {
                    'name': 'C12',
                    'index': 39,
                },
                'aggregates': {
                    'shape': None,
                },
                'surface_concentration': np.round((c1+c2)/2,4),
            },
            'indenter': {
                'name': 'AUM',
                'resname': 'AUM',
                'reference_atom': {
                    'name': 'AU',
                },
            },
            'substrate': {
                'name': 'AUM',
                'resname': 'AUM',
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
                'height': 180.0,
                # 'natoms':  # TODO: count automatically
            }
        },
        step_specific={
            'merge': {
                'tol': 2.0,
                'z_dist': 50.0,
                'x_shift': 0.0,
                'y_shift': 0.0,
            },
            'dtool_push': {
                'dtool_target': '/p/project/chfr13/hoermann4/dtool/TRIAL_DATASETS',
                'remote_dataset': None,
            }
        },
    )
    
    fp_files = wfg.push_infiles(fp)
    wf = wfg.build_wf()
    wf_list.append(wf)
