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

probe_on_substrate_input_datasets = [{'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 30.0,
  'uuid': 'a46b58a1-b568-4350-986b-5674a8dfa3ab'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 8.0,
  'uuid': '296499ba-4b00-4b67-8968-a8674efc8379'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 50.0,
  'uuid': 'd67aba32-8851-4fb1-b6d3-93526d877ca6'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 20.0,
  'uuid': 'dfdc4997-6a7e-471b-930e-c70aa918078f'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 3.0,
  'uuid': '45875381-62ae-46ba-b9c8-a875ca831fcf'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 1.0,
  'uuid': '01a58e9f-ca0f-410c-956d-eee69dbdf1ee'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 4.0,
  'uuid': 'bc6bf2c9-c869-4c66-9418-7a37885fc302'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 45.0,
  'uuid': '523fd601-43ca-4b19-b5ef-698a9e5c137a'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 10.0,
  'uuid': '736d647e-c904-4540-906f-b7b97d6817a1'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 9.0,
  'uuid': 'fdcc0dbd-ed2b-4340-938e-0a4ae592aa10'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 15.0,
  'uuid': '576ce98e-4f44-4f2e-96dd-23cf84b168ac'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 0.0,
  'uuid': '29afec61-1ca6-42bb-8470-13dba803a29a'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 6.0,
  'uuid': '9789c33c-8f57-40f6-b44f-fe90f28e3b16'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 25.0,
  'uuid': '922a9f9a-df54-40aa-9f6b-75abf8fb2095'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 35.0,
  'uuid': '15f345fe-b967-43ef-adb7-da63ce141cff'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 40.0,
  'uuid': '3fa42fda-15a1-4385-b108-89e146169288'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 5.0,
  'uuid': '5455ebda-c2bb-4f83-a830-442d0179726f'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 2.0,
  'uuid': 'e691113f-7db7-48ef-9c26-1cb685564552'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 7.0,
  'uuid': 'c72cbaf4-413a-4f42-b1e6-469039b5e9e7'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 0.0,
  'uuid': '3f92c1c8-23ae-43b6-ae52-359f576f7d08'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 45.0,
  'uuid': '856b5733-26b5-4270-9022-2a1748935231'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 9.0,
  'uuid': 'c2735cc3-a970-4dbe-8d18-9dbeb0133833'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 4.0,
  'uuid': 'a6013642-3756-4217-a59b-fa14c95f0d00'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 20.0,
  'uuid': '25d0599c-5a2d-454a-8c47-484213c49ab9'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 2.0,
  'uuid': 'edc39175-626c-40fb-a944-0e8d7a8257c8'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 3.0,
  'uuid': '2c921d38-80b0-42aa-9ea5-fd0a24bb3966'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 1.0,
  'uuid': 'e4ed7184-1da8-41f3-ad69-4ad667971236'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 50.0,
  'uuid': '183a2c1c-3a1b-4022-ab60-843ee25b8e68'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 5.0,
  'uuid': '4dc9117e-b0a3-42b9-a54f-537ef86b92fb'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 30.0,
  'uuid': '8c0e983c-58e5-472a-bfcd-dbb226e240ae'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 15.0,
  'uuid': '4f16424d-2cf2-4862-a455-874257f765da'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 8.0,
  'uuid': '6aa4acb2-cc83-46b5-a4bd-7cbcadfdb1ac'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 6.0,
  'uuid': '4dd0cc5d-83c7-4d31-b576-41e66870b5c4'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 35.0,
  'uuid': '8f22bbe2-49fb-4c41-bf1b-d7bce28d849e'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 10.0,
  'uuid': '6ff33b84-7321-4211-9ff4-24d720f7feb3'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 40.0,
  'uuid': '5aaf00a8-531e-4aab-9861-efc4db0ffaa8'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 7.0,
  'uuid': 'eee6f0f4-77f0-4ae6-ad7b-59f10827d150'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 25.0,
  'uuid': 'caa2bdb5-a009-4156-97b4-6d14fcc53ab3'}]

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
    
project_id = '2022-03-31-sds-on-au-111-probe-on-substrate-lateral-sliding'

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
                'dtool_target': '/p/project/hfr21/hoermann4/dtool/PRODUCTION/2022-03-31-sds-on-au-111-probe-on-substrate-lateral-sliding',
                'remote_dataset': None,
            }
        }
    )
    fp_files = wfg.push_infiles(fp)
    wf = wfg.build_wf()
    wf_list.append(wf)

    
# In[]:
    
# Not yet submitted
