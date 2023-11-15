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

# In[]:

index_file_input_datasets = [
 {'nmolecules': 916,
  'x_shift': 50.0,
  'y_shift': 50.0,
  'uuid': '608480be-e8a1-4485-b058-8e173b7d91b4'},
 {'nmolecules': 916,
  'x_shift': 50.0,
  'y_shift': 25.0,
  'uuid': 'f6a0e6f5-73c3-46c0-b175-0cccf3faab05'},
 {'nmolecules': 916,
  'x_shift': 50.0,
  'y_shift': 0.0,
  'uuid': 'b5ff6f76-8e39-4fe5-85c3-dfbc6a4f7d7f'},
 {'nmolecules': 916,
  'x_shift': 50.0,
  'y_shift': -50.0,
  'uuid': '1e7d1d6f-0325-47bf-aa77-1296b6ac5b9d'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'uuid': 'fa371636-acdc-4619-b7ba-ff90341fe690'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 25.0,
  'uuid': '9ac48ef6-fae6-4212-a8f2-445a8136bb8a'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 0.0,
  'uuid': '494c0567-2556-4dd6-b155-6d70df8b7c37'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': -25.0,
  'uuid': '43d9cdd7-2216-4567-9894-30a1123dc2d5'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': -50.0,
  'uuid': '2fb51ed8-0e11-40a3-86f7-decbf823ab8f'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 50.0,
  'uuid': '92ec5305-5a4c-41d1-abe9-ff800773c884'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'b19dc103-507c-4eff-b05b-681c5adb784c'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'uuid': '64e7243b-99fd-434d-b83d-dcd5122518f7'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': 50.0,
  'uuid': 'e954b547-e11d-443e-86d8-004f48d3f20c'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': 25.0,
  'uuid': 'e3c0f64a-7384-4545-9da2-ed49c9b26d21'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': 0.0,
  'uuid': 'b76d429e-c961-4b82-8174-cb3ae689e2ff'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'uuid': '169ff635-22b0-423d-9dee-a85a321a79bf'},
 {'nmolecules': 916,
  'x_shift': -25.0,
  'y_shift': 50.0,
  'uuid': 'ded34b6c-1b0b-4d3b-a31c-a834c952830b'},
 {'nmolecules': 916,
  'x_shift': -25.0,
  'y_shift': 25.0,
  'uuid': 'f7399397-dd96-49d1-85cf-459bfa3404a4'},
 {'nmolecules': 916,
  'x_shift': -25.0,
  'y_shift': -25.0,
  'uuid': '6cc02cbb-29d1-425b-8bca-34eff6a8b910'},
 {'nmolecules': 916,
  'x_shift': -25.0,
  'y_shift': -50.0,
  'uuid': 'c4f94979-f778-4d84-b779-d3de7be4f8c5'}
]

# y = -25.0, x = 0.0 only
probe_on_substrate_input_datasets = [
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 45.0,
  'uuid': '6268f844-e4c0-40cd-adad-ee0cdfa53a71'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 35.0,
  'uuid': '2b1e9079-8f95-44eb-89ba-c03b9bec440b'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 40.0,
  'uuid': '5617a3f9-644d-46a3-a8cd-7dee6f48c0dc'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 5.0,
  'uuid': 'd9a7d9a9-606f-4815-81a6-7e2f6c1f26d6'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 20.0,
  'uuid': '9c100916-3686-404e-95aa-2eb0273934f3'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 0.0,
  'uuid': '620ae5db-9e36-4af4-8aaa-5b817b60b583'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 10.0,
  'uuid': 'e5d12d2f-4db9-4636-bf28-33f1d59e4e7f'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 15.0,
  'uuid': 'd06373e7-f9ab-4fcc-b326-d24db817b861'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 30.0,
  'uuid': '2b09157a-f333-4425-a076-ae4ba78cedda'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 25.0,
  'uuid': 'b67ed3ac-df55-449f-8c0f-22be948ab01e'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 50.0,
  'uuid': '59563459-f62e-46ce-867f-6369ac155dd2'}
 ]

index_file_input_datasets_index_map = { (d["x_shift"], d["y_shift"]): i for i, d in enumerate(index_file_input_datasets) }

for d in probe_on_substrate_input_datasets:
    d['index_file_uuid'] = index_file_input_datasets[ index_file_input_datasets_index_map[(d['x_shift'],d['y_shift'])]]['uuid']
    
probe_on_substrate_input_datasets_index_map = { (d["x_shift"], d["y_shift"], d["distance"]): i for i, d in enumerate(probe_on_substrate_input_datasets) }
# In[29]
# parameters

parameter_sets = [
    {
        'direction_of_linear_movement': d,
        'constant_indenter_velocity': -1.0e-5, # 1 m / s
        'steps': 1500000, # 3 nm sliding
        'netcdf_frequency': 1000,
        'thermo_frequency': 1000,
        'thermo_average_frequency': 1000,
        'restart_frequency': 1000,
    } for d in range(2)
]

parameter_dict_list = [{**d, **p} for p in parameter_sets for d in probe_on_substrate_input_datasets]
# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_lateral_sliding import ProbeOnSubstrateLateralSliding

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# index = probe_on_substrate_input_datasets_index_map[0,0,25.0]
# In[25]:
    
project_id = '2022-01-21-sds-on-au-111-probe-on-substrate-lateral-sliding'

wf_list = []
# for c, substrate_uuid, probe_uuid in probe_on_substrate_input_datasets:
# c = 0.03
for p in parameter_dict_list:
    wfg = ProbeOnSubstrateLateralSliding(
        project_id=project_id,
        
        files_in_info={
            'data_file': {
                'query': {'uuid': p['uuid']},
                'file_name': 'default.lammps',
                'metadata_dtool_source_key': 'step_specific',
                'metadata_fw_dest_key': 'metadata->step_specific',
                'metadata_fw_source_key': 'metadata->step_specific',
            },
            'index_file': {
                'query': {'uuid': p['index_file_uuid']},
                'file_name': 'groups.ndx',
                'metadata_dtool_source_key': 'system',
                'metadata_fw_dest_key': 'metadata->system',
                'metadata_fw_source_key': 'metadata->system',
            }
        },
        integrate_push=True,
        description="SDS on Au(111) probe on substrate lateral sliding",
        owners=[{
            'name': 'Johannes Laurin Hörmann',
            'email': 'johannes.hoermann@imtek.uni-freiburg.de',
            'username': 'fr_jh1130',
            'orcid': '0000-0001-5867-695X'
        }],
        infile_prefix=prefix,
        machine='juwels',
        mode='production',
        system = {},
        step_specific={
            'probe_lateral_sliding': {
                'constant_indenter_velocity': p['constant_indenter_velocity'],
                'direction_of_linear_movement': p['direction_of_linear_movement'],
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
                
                'max_restarts': 100,
            },
            'filter_netcdf': {
                'group': 'indenter',
            },
            'dtool_push': {
                'dtool_target': '/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-12-30-sds-on-au-111-probe-on-substrate-lateral-sliding',
                'remote_dataset': None,
            }
        }
    )
    fp_files = wfg.push_infiles(fp)
    wf = wfg.build_wf()
    wf_list.append(wf)

    
# In[]:
    
#