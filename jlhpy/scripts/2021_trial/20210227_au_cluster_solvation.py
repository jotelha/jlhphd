# In[20]:
import os.path
import datetime
# FireWorks functionality
from fireworks import LaunchPad
from fireworks.utilities.filepad import FilePad

from fireworks.utilities.dagflow import DAGFlow, plot_wf
# sample for plotting fraph:
#    plot_wf(wf_list[0], view='combined', labels=True, target='wf.png', bbox=(2400,2400))


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
R = 26.3906  # indenter radius
A_Ang = 4*np.pi*R**2  # surface area in square Ansgtrom
A_nm = A_Ang / 10**2  # surface area in square nm

C = np.arange(0.25, 6.25, 0.25)  # n_per_nm_sq

N = np.round(A_nm*C).astype(int).tolist()

N = N[-1:]

parameter_values = [{'c': c, 'n': n, 'm': n } for c, n in zip(C,N)]

# In[220]:
    
from jlhpy.utilities.wf.packing.chain_wf_spherical_indenter_passivation import ParametricIndenterPassivation
from jlhpy.utilities.wf.phys_config import TOLERANCE, SURFACTANTS

project_id = '2021-02-26-sds-on-au-111-cluster-passivation-trial'

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})
wfg = ParametricIndenterPassivation(
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
        'c': 'system->surfactant->surface_concentration',
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
                'index': 2,
            },
            'head_atom': {
                'name': 'S',
            },
            'tail_atom': {
                'name': 'C12',
            },
            'surface_concentration': None,
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
                'outer_atom_index': 1,
                'inner_atom_index': 39,
                'tolerance': 2
            },
        },
        'pulling': {
            'pull_atom_name': 'C12',
            'spring_constant': 10000,  # pseudo-units
            'rate': -0.1,  # pseudo-units
            'nsteps': 1000,
        },
        'dtool_push': {
            'dtool_target': '/p/project/chka18/hoermann4/dtool/DATASETS',
            'remote_dataset': None,
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()
