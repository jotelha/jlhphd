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

# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrate
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_010_merge import MergeSubstrateAndProbeSystems
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

project_id = '2020-11-30-sds-on-au-111-probe-and-substrate'

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# parameter_values = [{'n': n, 'm': n, 's': s } for n in N for s in ['monolayer','hemicylinders']][10:11]

# In[25]
wfg = MergeSubstrateAndProbeSystems(
    project_id=project_id,
    
    files_in_info={
        'substrate_data_file': {
            'query': {'uuid': '6d5fe574-3359-4580-ae2d-eeda9ec5b926'},
            'file_name': 'default.gro',
            'metadata_dtool_source_key': 'system->substrate',
            'metadata_fw_dest_key': 'metadata->system->substrate',
            'metadata_fw_source_key': 'metadata->system->substrate',
        },
        'probe_data_file': {
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
