# In[20]:
import os.path
import datetime
# FireWorks functionality
from fireworks import LaunchPad
from fireworks.utilities.filepad import FilePad

from fireworks.utilities.dagflow import DAGFlow, plot_wf


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


# c, substrate (monolayer), probe uuid
probe_on_monolayer_input_datasets = [
    (0.0025,
      'e1766c11-ec23-488e-adfe-cbefc630ac68',
      '43a09eb9-a27b-42fb-a424-66d2cdbdf605'),
     (0.005,
      '13da3027-2a2f-45e7-bd8d-73bee135f24f',
      '9835cce9-b7d0-4e1f-9bdf-fd9767fea72c'),
     (0.0075,
      '0537047b-04c0-42b7-8aac-8de900c9e357',
      '30b97009-7d73-4e65-aa4a-04e1dc4cb2d2'),
     (0.01,
      'e11c3cb7-f53e-4024-af1c-dadbc8a01119',
      'a72b124b-c5aa-43d8-900b-f6b6ddc05d39'),
     (0.0125,
      'cda86934-50f1-4c7d-bdb5-3782c9f39a5a',
      '02c578b1-b331-42cf-8aef-4e3dcd0b4c77'),
     (0.015,
      'fe9bfb8d-d671-4c2d-bb1e-55dc1bfeec93',
      '974b41b2-de1c-421c-897b-7e091facff3a'),
     (0.0175,
      '90dbac16-9f05-4610-b765-484198116042',
      '86d2a465-61b8-4f1d-b13b-912c8f1f814b'),
     (0.02,
      'f0648f54-9a5d-488c-a913-ea53b88c99ce',
      '7e128862-4221-4554-bc27-22812a6047ae'),
     (0.0225,
      '9fb66e67-d08d-4686-972a-078bae8ef723',
      '1bc8bb4a-f4cf-4e4f-96ee-208b01bc3d02'),
     (0.025,
      'a9245caa-1439-4515-926f-c35b0476df44',
      '5b45ef1f-d24b-4a86-ab3c-3f3063b4def2'),
     (0.0275,
      'e96ff87c-7880-4d76-810e-e1a468d6b872',
      'b789ebc7-daec-488b-ba8f-e1c9b2d8fb47')]

probe_on_hemicylinders_input_datasets = [
     (0.005,
      '88c14189-f072-4df0-a04b-57bf27760b9d',
      '9835cce9-b7d0-4e1f-9bdf-fd9767fea72c'),
     (0.0075,
      'fcc304df-219a-4170-a6e0-bee06eed14e2',
      '30b97009-7d73-4e65-aa4a-04e1dc4cb2d2'),
     (0.01,
      '0899dd47-5659-408c-8dc1-253980adc975',
      'a72b124b-c5aa-43d8-900b-f6b6ddc05d39'),
     (0.0125,
      '01339270-76df-40c2-bec6-c69072f5a5f7',
      '02c578b1-b331-42cf-8aef-4e3dcd0b4c77'),
     (0.015,
      'b14873d7-0bba-4c2d-9915-ac9ee99f43c7',
      '974b41b2-de1c-421c-897b-7e091facff3a'),
     (0.0175,
      'b5d0bbfe-9c69-4dfd-9189-417bfa367882',
      '86d2a465-61b8-4f1d-b13b-912c8f1f814b')]

probe_on_substrate_input_datasets = [*probe_on_monolayer_input_datasets, *probe_on_hemicylinders_input_datasets]
# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrateMergeConversionMinimizationAndEquilibration

from jlhpy.utilities.wf.mappings import psfgen_mappings_template_context

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})


# In[25]:
    
project_id = '2021-02-25-sds-on-au-111-probe-and-substrate-conversion'

wf_list = []
for c, substrate_uuid, probe_uuid in probe_on_substrate_input_datasets:
    wfg = ProbeOnSubstrateMergeConversionMinimizationAndEquilibration(
        project_id=project_id,
        
        files_in_info={
            'substrate_data_file': {
                'query': {'uuid': substrate_uuid},
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
        machine='juwels',
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
                    'index': 1,
                },
                'tail_atom': {
                    'name': 'C12',
                    'index': 39,
                },
                'aggregates': {
                    'shape': None,
                },
                'surface_concentration': c
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
                'x_shift': 25.0,  # hemicylinders repeat every 50 Ang
                'y_shift': 0.0,
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
                    'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the ore of the indenter
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
            'dtool_push': {
                'dtool_target': '/p/project/chka18/hoermann4/dtool/DATASETS',
                'remote_dataset': None,
            }
        }
    )
    
    fp_files = wfg.push_infiles(fp)
    wf = wfg.build_wf()
    wf_list.append(wf)