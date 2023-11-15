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

probe_on_monolayer_input_datasets = []

# c, substrate (monolayer), probe uuid
probe_on_hemicylinders_input_datasets = [
     (0.03,
      'b789ebc7-daec-488b-ba8f-e1c9b2d8fb47', # probe
      'a5582146-ac99-422b-91b4-dd12676b82a4')] # substrate

probe_on_substrate_input_datasets = [*probe_on_monolayer_input_datasets, *probe_on_hemicylinders_input_datasets]

# In[]:
    
    
# from '2021-02-26-sds-on-au-111-probe-and-substrate-conversion'
probe_on_substrate_input_datasets = [
     {'concentration': 0.03,
      'shape': 'hemicylinders',
      'substrate_uri': 'file://jwlogin04.juwels/p/project/hfr21/hoermann4/sandbox/20211003_c30/2021-09-30-11-25-34-604687-c-30-n-675-m-675-s-hemicylinders-substratepassivation',
      'probe_uri': 'file://jwlogin04.juwels/p/project/hfr21/hoermann4/sandbox/20211003_c30/2020-07-29-03-47-41-026744-n-241-m-241-gromacsrelaxation',
     }
]

    
# parameters

parameter_sets = [
    {
        'x_shift': x,  
        'y_shift': y,  # hemicylinders repeat every 50 Ang}
        'constant_indenter_velocity': -1.0e-5, # 1 m / s
        'steps': 2500000,
        'netcdf_frequency': 1000,
        'thermo_frequency': 1000,
        'thermo_average_frequency': 1000,
        'restart_frequency': 1000,
    } for x in np.arange(0.0, 25.0, 25.0) for y in np.arange(-50.0+12.5, 75.0+12.5, 25.0)
]

parameter_dict_list = [{**d, **p} for p in parameter_sets for d in probe_on_substrate_input_datasets]
# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction

from jlhpy.utilities.wf.mappings import psfgen_mappings_template_context

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})


# In[25]:
    
project_id = '2021-12-09-sds-on-au-111-probe-and-substrate-merge-and-approach'

wf_list = []
# for c, substrate_uuid, probe_uuid in probe_on_substrate_input_datasets:
# c = 0.03
for p in parameter_dict_list:
    wfg = ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction(
        project_id=project_id,
        
        files_in_info={
            'substrate_data_file': {
                #'query': {'uuid': substrate_uuid},
                'uri': p['substrate_uri'],
                'file_name': 'default.gro',
                'metadata_dtool_source_key': 'system->substrate',
                'metadata_fw_dest_key': 'metadata->system->substrate',
                'metadata_fw_source_key': 'metadata->system->substrate',
            },
            'probe_data_file': {
                #'query': {'uuid': probe_uuid},
                'uri': p['probe_uri'],
                'file_name': 'default.gro',
                'metadata_dtool_source_key': 'system->indenter',
                'metadata_fw_dest_key': 'metadata->system->indenter',
                'metadata_fw_source_key': 'metadata->system->indenter',
            }
        },
    
        integrate_push=True,
        description="SDS on Au(111) substrate and probe",
        owners=[{
            'name': 'Johannes Laurin Hörmann',
            'email': 'johannes.hoermann@imtek.uni-freiburg.de',
            'username': 'fr_jh1130',
            'orcid': '0000-0001-5867-695X'
        }],
        infile_prefix=prefix,
        machine='juwels',
        mode='trial',
        system = {
            'concentration': p['concentration'],
            'shape': p['shape'],
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
                # 'surface_concentration': c
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
                'x_shift': p['x_shift'],  
                'y_shift': p['y_shift'],
            },
            'psfgen': psfgen_mappings_template_context,
            'split_datafile': {
                'region_tolerance': 5.0,
                'shift_tolerance': 2.0,
            },
            'minimization': {
                'ftol': 1.0e-6,
                'maxiter': 10000,
                'maxeval': 10000,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': True,
                'skin_distance': 3.0,
            },
            'equilibration': {
                'nvt': {
                    'initial_temperature': 1.0,
                    'temperature': 298.0,
                    'langevin_damping': 1000,
                    'steps': 10000,
                    'netcdf_frequency': 100,
                    'thermo_frequency': 100,
                    'thermo_average_frequency': 100,
                
                    'ewald_accuracy': 1.0e-4,
                    'coulomb_cutoff': 8.0,
                    'neigh_delay': 2,
                    'neigh_every': 1,
                    'neigh_check': True,
                    'skin_distance': 3.0,
                },
                'npt': {
                    'pressure': 1.0,
                    'temperature': 298.0,
                    'barostat_damping': 10000,
                    'langevin_damping': 1000,
                    'steps': 10000,
                    'netcdf_frequency': 100,
                    'thermo_frequency': 100,
                    'thermo_average_frequency': 100,
                    
                    'ewald_accuracy': 1.0e-4,
                    'coulomb_cutoff': 8.0,
                    'neigh_delay': 2,
                    'neigh_every': 1,
                    'neigh_check': True,
                    'skin_distance': 3.0
                },
                'dpd': {
                    'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                    'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the core of the indenter
                    'temperature': 298.0,
                    'steps': 10000,
                    'netcdf_frequency': 100,
                    'thermo_frequency': 100,
                    'thermo_average_frequency': 100,
                    
                    'ewald_accuracy': 1.0e-4,
                    'coulomb_cutoff': 8.0,
                    'neigh_delay': 2,
                    'neigh_every': 1,
                    'neigh_check': True,
                    'skin_distance': 3.0
                },
            },
            'probe_normal_approach': {
                'constant_indenter_velocity': p['constant_indenter_velocity'],
                'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the ore of the indenter
                'temperature': 298.0,
                'steps': p['steps'],
                'netcdf_frequency': p['netcdf_frequency'],
                'thermo_frequency': p['thermo_frequency'],
                'thermo_average_frequency': p['thermo_average_frequency'],
                'restart_frequency': p['restart_frequency'],
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': True,
                'skin_distance': 3.0,
            },
            'filter_netcdf': {
                'group': 'indenter',
            },
            'extract_forces': {
                'dimension': 2, # z-direction
            },
            'frame_extraction': {
                'first_distance_to_extract': 50.0,
                'last_distance_to_extract': 0.0,
                'distance_interval': -5.0,
                'time_step': 2.0, # this should be extractable from somewhere else, but not the case
            },
            'dtool_push': {
                'dtool_target': '/p/project/hfr21/hoermann4/dtool/TRIAL_DATASETS/',
                'remote_dataset': None,
            }
        }
    )
    fp_files = wfg.push_infiles(fp)
    wf = wfg.build_wf()
    wf_list.append(wf)
    
# In[]:
    
# 20211007: So far only queued first 5 workflows [:5]
# 20211011: Corrected nesting, queued [5:10]
# 20211014: confirmed 5:10 running, queued [10:], requeued [:5]
