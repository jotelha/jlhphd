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

# from '2021-02-05-sds-on-au-111-probe-and-substrate-approach' probe on monolayer approach at 1 m / s
index_file_input_datasets = [
 {'nmolecules': 156,
  'concentration': 0.005,
  'uuid': '477ae425-9a75-4977-92fd-e786d829f525'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'uuid': '0e6af9b0-c5a4-451f-bfb7-2f2b37727048'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'uuid': '6a602d01-722d-4ecf-9894-83b987d5a8bd'},
 #{'nmolecules': 390,
 # 'concentration': 0.0125,
 # 'uuid': '74ede4bb-e877-406c-93cd-b4136b100b35'},
 #{'nmolecules': 625,
 # 'concentration': 0.02,
 # 'uuid': '623ee87d-d25c-401f-a7a9-e55bda3c9652'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'uuid': '833b057f-e934-4fe1-81de-e4bb180a3532'}
]
 
# from '2022-02-11-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration', except 390 and 626
probe_on_substrate_input_datasets = [
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 50.0,
  'uuid': 'eaf1e7f0-8faf-4dd5-b2b9-a52d3bdd86b2'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 3.0,
  'uuid': '90045fd1-97a4-4774-b567-714e43211a9a'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 8.0,
  'uuid': '75b82fbd-cdf5-4097-91d7-7ebf7a57c1e1'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 20.0,
  'uuid': 'ee0a4191-945c-4083-be5c-933ee536fdba'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 0.0,
  'uuid': 'a2681c94-1d6d-4ad9-989c-698d209d1412'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 30.0,
  'uuid': '47c3b1f9-598c-445d-ad38-50dd213b2d59'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 7.0,
  'uuid': 'be61ccb4-7016-4a53-848b-816a765166fe'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 6.0,
  'uuid': '19e4e660-ce7a-4bbc-8f73-ca05b7cef1ff'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 45.0,
  'uuid': 'f6d82fd6-247e-449e-b71e-8afb3511997b'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 9.0,
  'uuid': 'd95c7d81-f644-4c9b-9d06-bf91295b5a4c'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 2.0,
  'uuid': '23c1c550-c404-4584-ac9c-1916f1b8f18f'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 4.0,
  'uuid': '2ab30db8-238f-4c10-842d-de1392fd790a'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 25.0,
  'uuid': 'ddacc0ac-4225-48b3-a61c-c2303abc5c56'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 10.0,
  'uuid': '15b143ac-8ce6-455d-861c-8c7f0560491e'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 5.0,
  'uuid': '6be606dd-487d-4e87-815b-32f170a34983'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 40.0,
  'uuid': 'aa6bda03-9897-4179-8d4e-90aa3ac9b848'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 35.0,
  'uuid': '0335bd6f-03be-43b2-8b9c-838746d13ae6'},
 {'nmolecules': 156,
  'concentration': 0.005,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 1.0,
  'uuid': 'd3d779f9-9ab3-46d4-9516-ed81dd2d42b6'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 10.0,
  'uuid': 'fa1ae03e-8965-46de-b214-593053e31b23'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 25.0,
  'uuid': 'dec9ebd6-89ea-4bed-93dc-b8fe2e09bdd7'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 40.0,
  'uuid': '56acfeba-50da-433e-9ee3-ea576a8c2f48'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 15.0,
  'uuid': '6a650628-cc5e-4e6f-8ff3-315bbe74f168'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 2.0,
  'uuid': '2dfae261-1dd2-4cd9-97a6-9bc517409991'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 9.0,
  'uuid': '2bf4c327-5a26-49c5-8904-384001e4d3ee'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 8.0,
  'uuid': '18ac8bb3-ead4-45a6-ab4f-ce99745b4b59'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 50.0,
  'uuid': '7a48051a-ac45-49cd-97ca-d12004c37508'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 4.0,
  'uuid': '24d37dfd-fd99-494e-ab29-f38a635f3513'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 1.0,
  'uuid': 'a30b0909-01b8-4abf-a49e-6298fc6d1c2b'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 5.0,
  'uuid': 'a20201a9-fd3b-40ae-9f72-41065268b0e7'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 6.0,
  'uuid': '2766f8db-489b-4d81-aa73-c74535c4334c'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 20.0,
  'uuid': '7a552f4b-13c9-4759-9cd4-7bb754c6b4d6'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 35.0,
  'uuid': 'eeb06c73-83da-4fca-8db1-38d805f66774'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 7.0,
  'uuid': 'be1dc00a-96aa-475e-a742-6508fc293687'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 45.0,
  'uuid': '7e4c8adf-b55a-4423-8fc2-dd6c259c56cd'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 3.0,
  'uuid': '11ea1fe1-5448-4809-9001-fa1fc2497b17'},
 {'nmolecules': 235,
  'concentration': 0.0075,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 30.0,
  'uuid': 'ede7e482-ce14-4084-a33b-18be1baa0bb0'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 30.0,
  'uuid': '319fd289-8762-48df-b0c1-81e7c2f18f64'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 3.0,
  'uuid': '409d17dd-e85d-4b64-a30a-fccefd6f5cea'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 50.0,
  'uuid': '93b3c1b9-ed15-4f5e-9366-c56abc337df7'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 15.0,
  'uuid': 'df7aaa84-0f19-4db9-96db-b93bf7907995'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 10.0,
  'uuid': '96445123-be12-4ba7-ae00-66f2c63c6c5c'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 0.0,
  'uuid': 'e12f5ade-7623-4bf2-a78b-894c6c905f63'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 4.0,
  'uuid': 'adc46af4-0c0e-4193-bdfe-49f145a74aec'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 9.0,
  'uuid': '9a494a09-09f2-44eb-8266-60d5c9ee2734'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 35.0,
  'uuid': '53a7c7c2-c26e-4591-a7ca-8c8b2a506fe8'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 7.0,
  'uuid': 'c59ae2a9-2bf3-41d4-a95b-d3108dda10bb'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 6.0,
  'uuid': 'ea5dd168-34aa-43b0-aca0-417a8bd2d7f8'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 1.0,
  'uuid': '22885349-522d-4945-ad40-de91329453d6'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 45.0,
  'uuid': '1fd6ac1f-257f-4b9f-bcec-487397c30687'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 5.0,
  'uuid': '472d06c3-5121-4de3-a3f8-59e0b170d9bb'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 2.0,
  'uuid': '51848d67-1931-4b95-a264-a84327d2449a'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 40.0,
  'uuid': '50ed58c7-432e-4d1b-b510-253c86693583'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 25.0,
  'uuid': '2406655f-8cd0-4e5c-a984-0ee3a2f2b68c'},
 {'nmolecules': 313,
  'concentration': 0.01,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 8.0,
  'uuid': '7139536d-cb96-41ae-8ac8-cd6be99d850c'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 45.0,
  'uuid': '8c8ed8cc-00d4-4cc4-bd47-307cfa0a7ecb'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 8.0,
  'uuid': '86cb67c8-0902-44e6-af14-2e198025f66b'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 50.0,
  'uuid': 'a7d01232-8fbc-4f4c-b112-874f3a9a58e3'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 9.0,
  'uuid': 'd8615d6c-80bd-4ade-95b5-0b1ba0825273'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 7.0,
  'uuid': '927fb2be-8dbb-47bd-8696-000769fad94c'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 0.0,
  'uuid': '581bafa4-5888-4653-a787-56cb00ca8d41'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 5.0,
  'uuid': 'd0c9b6f7-7d73-42d5-a202-1ceade5e51a9'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 3.0,
  'uuid': '7d93add8-e1e5-425e-a80a-9fa4c7c31c96'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 20.0,
  'uuid': 'b9aab2db-630f-4e14-9c25-9de1baca41f2'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 10.0,
  'uuid': 'fc17023e-5f17-4dec-a437-97a02c485ff1'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 2.0,
  'uuid': '5322505e-187e-4341-b0da-7775c0dd43ec'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 35.0,
  'uuid': 'f81fc376-a4b0-4725-b39f-9f92976c1952'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 25.0,
  'uuid': '05f92518-d7ea-42ab-882b-074dd7461419'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 15.0,
  'uuid': '23407813-2dfd-482f-b0bf-daff9229fee1'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 4.0,
  'uuid': '57eda77c-eac7-4a3d-a401-725762f8c910'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 40.0,
  'uuid': '2c731e01-d520-4a81-a5d0-ef4eab69ac22'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 30.0,
  'uuid': '87407c78-cb71-45b6-afba-68c2e1e52ade'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 1.0,
  'uuid': '390097aa-08ed-4805-8150-faccbe01556c'}]


index_file_input_datasets_index_map = { d["nmolecules"]: i for i, d in enumerate(index_file_input_datasets) }

for d in probe_on_substrate_input_datasets:
    d['index_file_uuid'] = index_file_input_datasets[index_file_input_datasets_index_map[d["nmolecules"]]]['uuid']
    
probe_on_substrate_input_datasets_index_map = {(d["nmolecules"], d["distance"]): i for i, d in enumerate(probe_on_substrate_input_datasets)}
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
    } for d in range(1,2)
]

parameter_dict_list = [{**d, **p} for p in parameter_sets for d in probe_on_substrate_input_datasets]
# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_lateral_sliding import ProbeOnSubstrateLateralSliding

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# index = probe_on_substrate_input_datasets_index_map[0,0,25.0]
# In[25]:
    
project_id = '2022-04-19-sds-on-au-111-probe-on-substrate-lateral-sliding'

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
    
    
# In[]:
    