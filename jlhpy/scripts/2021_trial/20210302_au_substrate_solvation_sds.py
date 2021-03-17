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
C = np.arange(0.25, 6.25, 0.25) # n_per_nm_sq
# n_per_nm_sq = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
# n_per_nm_sq = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
N = np.round(A_nm*C).astype(int).tolist()

# launched on 2020/12/13
# parameter_values = [
#     {'c': c, 'n': n, 'm': n, 's': s } 
#     for c, n in zip(C,N) for s in ['monolayer','hemicylinders']][14:20]

# launched on 2020/12/14
# parameter_values = [
#     {'c': c, 'n': n, 'm': n, 's': s } 
#     for c, n in zip(C,N) for s in ['monolayer','hemicylinders']][20:22]

# launched on 2020/12/14
# parameter_values = [
#     {'c': c, 'n': n, 'm': n, 's': s } 
#     for c, n in zip(C,N) for s in ['monolayer','hemicylinders']][:14]

# launched on 2020/12/20
parameter_values = [
    {'c': c, 'n': n, 'm': n, 's': s } 
    for c, n in zip(C,N) for s in ['monolayer','hemicylinders']][23:24]

# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.flat_packing.chain_wf_flat_substrate_passivation import SubstratePassivation
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

project_id = '2020-03-02-sds-on-au-111-substrate-passivation-trial'

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# In[25]
wfg = SubstratePassivation(
    project_id=project_id,

    files_in_info={ 
        'data_file': {  # smb://jh1130/b5774404-e151-4398-bda9-36eb523a0ae7
            'query': {'uuid': 'b5774404-e151-4398-bda9-36eb523a0ae7'},
            'file_name': 'default.lammps',
            'metadata_dtool_source_key': 'system->substrate',
            'metadata_fw_dest_key': 'metadata->system->substrate',
            'metadata_fw_source_key': 'metadata->system->substrate',
        },
    },
    integrate_push=True,
    description="SDS on Au(111) substrate passivation trial",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    infile_prefix=prefix,
    machine='juwels',
    mode='trial',
    parameter_label_key_dict={
        'c': 'system->surfactant->surface_concentration',
        'n': 'system->surfactant->nmolecules',
        'm': 'system->counterion->nmolecules',
        's': 'system->surfactant->aggregates->shape'},
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
            'surface_concentration': None,

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
                'tolerance': 1.0  # intead of 1.5 or 2
            },
        },
        'dtool_push': {
            'dtool_target': '/p/project/chka18/hoermann4/dtool/DATASETS',
            'remote_dataset': None,
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()
