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


# In[]:

# from '2020-07-29-sds-on-au-111-indenter-passivation'
probe_input_datasets = [
 {'nmolecules': 525,
  'uuid': '55a882d7-ae73-4839-9e0b-8a1074f06a4c',
  'concentration': 6.75},
 {'nmolecules': 503,
  'uuid': 'b011d121-23e3-43bb-a76f-315e2b374c4b',
  'concentration': 6.5},
 {'nmolecules': 481,
  'uuid': '8623845e-b772-4e01-93c9-a8c183e04355',
  'concentration': 6.0},
 {'nmolecules': 459,
  'uuid': '8ecf8bad-abea-4447-9649-1c3762703861',
  'concentration': 5.75},
 {'nmolecules': 438,
  'uuid': '6099b0bb-8a15-453a-aaaf-3d3a8be00f2f',
  'concentration': 5.5},
 {'nmolecules': 416,
  'uuid': '790fb7d0-1c8d-4a1f-aa54-e87780f9e23f',
  'concentration': 5.25},
 {'nmolecules': 394,
  'uuid': '2fba1825-b4c2-4b10-b9fd-1654223b0c5a',
  'concentration': 5.0},
 {'nmolecules': 372,
  'uuid': 'f1599301-b69c-4408-a374-80081464c6a8',
  'concentration': 4.75},
 {'nmolecules': 350,
  'uuid': '6ecaebe5-3c2f-4ba7-9d87-ce6d372d8b6d',
  'concentration': 4.5},
 {'nmolecules': 328,
  'uuid': 'f301b376-9420-4061-9fd5-51fe966f7dcc',
  'concentration': 4.25},
 {'nmolecules': 306,
  'uuid': '2ebfabd0-3038-4e59-8561-adeb1261e7b1',
  'concentration': 4.0},
 {'nmolecules': 284,
  'uuid': 'e7bde2b4-d688-45a5-b8ca-1b458336e18c',
  'concentration': 3.5},
 {'nmolecules': 263,
  'uuid': 'a64af15a-5b3e-4e5b-a11b-fc2ee0a42566',
  'concentration': 3.25},
 {'nmolecules': 241,
  'uuid': 'b789ebc7-daec-488b-ba8f-e1c9b2d8fb47',
  'concentration': 3.0},
 {'nmolecules': 219,
  'uuid': '5b45ef1f-d24b-4a86-ab3c-3f3063b4def2',
  'concentration': 2.75},
 {'nmolecules': 197,
  'uuid': '1bc8bb4a-f4cf-4e4f-96ee-208b01bc3d02',
  'concentration': 2.5},
 {'nmolecules': 175,
  'uuid': '7e128862-4221-4554-bc27-22812a6047ae',
  'concentration': 2.25},
 {'nmolecules': 153,
  'uuid': '86d2a465-61b8-4f1d-b13b-912c8f1f814b',
  'concentration': 2.0},
 {'nmolecules': 131,
  'uuid': '974b41b2-de1c-421c-897b-7e091facff3a',
  'concentration': 1.75},
 {'nmolecules': 109,
  'uuid': '02c578b1-b331-42cf-8aef-4e3dcd0b4c77',
  'concentration': 1.5},
 {'nmolecules': 88,
  'uuid': 'a72b124b-c5aa-43d8-900b-f6b6ddc05d39',
  'concentration': 1.0},
 {'nmolecules': 66,
  'uuid': '30b97009-7d73-4e65-aa4a-04e1dc4cb2d2',
  'concentration': 0.75},
 {'nmolecules': 44,
  'uuid': '9835cce9-b7d0-4e1f-9bdf-fd9767fea72c',
  'concentration': 0.5},
 {'nmolecules': 22,
  'uuid': '43a09eb9-a27b-42fb-a424-66d2cdbdf605',
  'concentration': 0.25}
]
    
# from '2021-10-06-sds-on-au-111-substrate-passivation
substrate_input_datasets = [
 {'nmolecules': 675,
  'concentration': 3.0,
  'shape': 'monolayer',
  'uuid': '115ac46b-c3b4-4d1e-920f-893164a971f2'},
 {'nmolecules': 619,
  'concentration': 2.75,
  'shape': 'monolayer',
  'uuid': 'e2e6d573-0e63-464b-9107-053aae876327'},
 {'nmolecules': 562,
  'concentration': 2.5,
  'shape': 'monolayer',
  'uuid': 'd3e139ae-8c57-4953-beef-c1c2de68ddf0'},
 {'nmolecules': 506,
  'concentration': 2.25,
  'shape': 'monolayer',
  'uuid': '5c37e1b8-9c51-43e6-b3ff-e22906e8399a'},
 {'nmolecules': 450,
  'concentration': 2.0,
  'shape': 'monolayer',
  'uuid': '51bb25e1-0568-43e3-b212-dde0f24ecf85'},
 {'nmolecules': 394,
  'concentration': 1.75,
  'shape': 'monolayer',
  'uuid': 'cd8193d6-d274-4284-92f0-7c2515ac6d1b'},
 {'nmolecules': 338,
  'concentration': 1.5,
  'shape': 'monolayer',
  'uuid': 'b29dc69f-1a5d-4bd7-9da6-68c13844d544'},
 {'nmolecules': 281,
  'concentration': 1.25,
  'shape': 'monolayer',
  'uuid': 'b8aafb69-92ce-4a07-95bc-690e0a1a3770'},
 {'nmolecules': 225,
  'concentration': 1.0,
  'shape': 'monolayer',
  'uuid': '3e6cecc7-de83-4c29-bda8-83405078a126'},
 {'nmolecules': 169,
  'concentration': 0.75,
  'shape': 'monolayer',
  'uuid': 'ba5718b0-71a6-4e72-93a1-32fcbb1d611b'},
 {'nmolecules': 112,
  'concentration': 0.5,
  'shape': 'monolayer',
  'uuid': 'ac8a097c-d6ac-4a2b-8a4c-51bad7f5af59'},
 {'nmolecules': 56,
  'concentration': 0.25,
  'shape': 'monolayer',
  'uuid': 'c2e8ff45-47eb-49fe-8a4e-ae5876f83288'}
]
    
# from '2021-02-26-sds-on-au-111-probe-and-substrate-conversion'
concentration_probe_input_dataset_dict = {
    d['concentration']: d for d in probe_input_datasets}

concentration_substrate_input_dataset_dict = {
    d['concentration']: d for d in substrate_input_datasets
}

probe_on_substrate_input_datasets = [
     {'concentration': concentration,
      'shape': 'monolayer',
      'substrate_uuid': substarte_dataset['uuid'],
      'probe_uuid': concentration_probe_input_dataset_dict[concentration]['uuid']
     } for concentration, substarte_dataset in concentration_substrate_input_dataset_dict.items()
       if concentration in concentration_probe_input_dataset_dict
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
    } for x in np.arange(0, 50.0, 25.0) for y in np.arange(0, 50.0, 25.0) # for now, four configurations per concentration
]

parameter_dict_list = [{**d, **p} for p in parameter_sets for d in probe_on_substrate_input_datasets]
# In[20]:

# SDS on Au(111)
import jlhpy.utilities.wf.mixin.mixin_wf_storage as mixin_wf_storage
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction

from jlhpy.utilities.wf.mappings import psfgen_mappings_template_context

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})


# In[25]:
    
project_id = '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach'

wf_list = []
for p in parameter_dict_list:
    wfg = ProbeOnSubstrateMergeConversionMinimizationEquilibrationApproachAndFrameExtraction(
        project_id=project_id,
        
        files_in_info={
            'substrate_data_file': {
                'query': {
                    'uuid': p['substrate_uuid'],
                    'base_uri':'s3://frct-simdata', # assure to get s3 entry, not outdated ecs
                },
                'file_name': 'default.gro',
                'metadata_dtool_source_key': 'system->substrate',
                'metadata_fw_dest_key': 'metadata->system->substrate',
                'metadata_fw_source_key': 'metadata->system->substrate',
            },
            'probe_data_file': {
                'query': {
                    'uuid': p['probe_uuid'],
                    'base_uri':'s3://frct-simdata', # assure to get s3 entry, not outdated ecs
                },
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
        mode='production',
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
                
                'max_restarts': 20,
            },
            'filter_netcdf': {
                'group': 'indenter',
            },
            'frame_extraction': {
                'first_distance_to_extract': 50.0,
                'last_distance_to_extract': 0.0,
                'distance_interval': -1.0,
                'time_step': 2.0, # this should be extractable from somewhere else, but not the case
            },
            'dtool_push': {
                'dtool_target': f'/p/project/hfr21/hoermann4/dtool/PRODUCTION/{project_id}',
                'remote_dataset': None,
            }
        }
    )
    fp_files = wfg.push_infiles(fp)
    wf = wfg.build_wf()
    wf_list.append(wf)
    
# In[]:

# dump gernerated workflows to file

for i, wf in enumerate(wf_list):
    wf.to_file("wf_{:03d}.json".format(i), indent=4)
    
# In[]

# 2022-02-12: queued wf_list[0:10]
# 2022-02-21: failed for good: concentration 0.75, 3.0, both x_, y_shiift 0,0 requeued wf_list[8], wf_list[0]
# 2022-02-21: also queued 10:20
