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
from jlhpy.utilities.wf.flat_packing.chain_wf_flat_substrate_passivation import SubstratePassivation
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

project_id = '2020-08-31-sds-on-au-111-substrate-passivation-trial'

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# parameter_values = [{'n': n, 'm': n } for n in N]
parameter_values = [{'n': n, 'm': n } for n in [N[10]]]

wfg = SubstratePassivation(
    project_id=project_id, 
    
    source_project_id="2020-08-24-au-111-fcc-substrate-creation-trial",
    source_step='LAMMPSEquilibrationNPT:ProcessAnalyzeAndVisualize:push_filepad',
    metadata_fp_source_key='metadata->system->substrate',
    metadata_fw_dest_key='metadata->system->substrate',
    metadata_fw_source_key='metadata->system->substrate',
    
    integrate_push=True,
    description="Trial runs for SDS on Au(111) substrate passivation",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    infile_prefix=prefix,
    machine='juwels_devel',
    mode='trial',
    parameter_label_key_dict={
        'n': 'system->surfactant->nmolecules', 
        'm': 'system->counterion->nmolecules'},
    parameter_values=parameter_values,
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
                'index': SURFACTANTS["SDS"]["head_atom_index"],
            },
            'tail_atom': {
                'name': 'C12',
                'index': SURFACTANTS["SDS"]["tail_atom_index"],
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
            # 'natoms':  # TODO: count automatically
        }
    },
    step_specific={
        'conversion': {
            'lmp_type_to_element_mapping': {
                '11': 'Au',
            },
            'element_to_pdb_atom_name_mapping': {
                'Au': 'AU',
            },
            'element_to_pdb_residue_name_mapping': {
                'Au': 'AUM',
            },
        },
        'packing' : {
            'surfactant_substrate': {
                'tolerance': TOLERANCE
            },
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/TRIAL_DATASETS',
            'remote_dataset': None,
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[55]:
    
pf = PackingOnFlatSubstrate('bla')

wf_pf = pf.build_wf()