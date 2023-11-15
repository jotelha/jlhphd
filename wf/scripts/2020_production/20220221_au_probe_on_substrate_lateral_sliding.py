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
 {'nmolecules': 390,
  'concentration': 0.0125,
  'uuid': '74ede4bb-e877-406c-93cd-b4136b100b35'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'uuid': '623ee87d-d25c-401f-a7a9-e55bda3c9652'},
 {'nmolecules': 703,
  'concentration': 0.0225,
  'uuid': '833b057f-e934-4fe1-81de-e4bb180a3532'}
]
 
# from '2022-02-11-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration', nmolecules 390 and 625 only
probe_on_substrate_input_datasets = [
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 40.0,
  'uuid': '4865e57c-914c-4f4d-b4f3-71b1a8a77764'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 5.0,
  'uuid': '104d184f-a8e2-4d0b-837a-bd1a1771afcf'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 8.0,
  'uuid': '710cf7c0-934c-487d-b728-7e78411b19f7'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 1.0,
  'uuid': '5b128143-ecda-414a-9ed6-c4fdb08107ce'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 4.0,
  'uuid': 'c010b838-02fe-4273-9144-63142d84fae5'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 45.0,
  'uuid': 'd6c27d18-37d1-481e-96e9-34d329928426'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 50.0,
  'uuid': '29389e36-714f-42ed-8147-fde238c8c9db'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 30.0,
  'uuid': 'ad49f5c3-5269-46f1-8f49-fdffe699f898'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 3.0,
  'uuid': 'b3116f2e-1cc6-40ec-acdc-eb782300f271'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 20.0,
  'uuid': '6d57c5a4-56e4-405a-890c-3dac1eb556dd'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 0.0,
  'uuid': 'e7ad8eec-a02f-4c60-96e7-4fd2b46e0240'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 10.0,
  'uuid': 'ee0b5b52-c889-49b3-acd3-e3cdb1c61ba9'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 35.0,
  'uuid': '3e862ae5-1e9a-4b1e-ac98-49a89459584e'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 25.0,
  'uuid': 'b376687a-070c-43c4-9d9c-ff07cc9ed64b'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 7.0,
  'uuid': '1fcaca92-09e6-440d-ac95-b8c1f34fb309'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 15.0,
  'uuid': '8fb65fa5-bb4e-4424-9ec1-6c892b35ac6a'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 9.0,
  'uuid': '3b566fbf-46cd-4529-bc8c-23734a9fb63d'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 2.0,
  'uuid': 'bd77a5ae-b9d7-4797-9302-6db4d758f541'},
 {'nmolecules': 390,
  'concentration': 0.0125,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 6.0,
  'uuid': 'aa4ac8f1-3744-4a90-941d-dfdd7df7e325'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 7.0,
  'uuid': 'e747da0c-f0ec-40ae-8760-b1aed9ea555d'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 30.0,
  'uuid': '4837d3c6-4c3b-4b6f-8075-f722a7ed2915'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 15.0,
  'uuid': '2f4935d4-2bbe-43c0-baba-c48ddc3024c9'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 50.0,
  'uuid': '4a6b1ecf-632a-41e5-9986-5c788bf96226'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 20.0,
  'uuid': '07aaa1bc-f17a-4294-b4eb-ef3ecf6aadde'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 10.0,
  'uuid': '9a8b9b90-db0c-488c-b9a8-8be3822366ee'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 6.0,
  'uuid': 'f3ef5518-c516-478f-a218-98e8c53109b9'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 9.0,
  'uuid': 'd15507bb-117e-4a07-90d5-24d9d233cfd8'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 2.0,
  'uuid': '989728c0-70eb-4a8f-8404-d5b6ac9af089'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 8.0,
  'uuid': '4a955733-127e-4e06-8468-ddf116d37f7e'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 4.0,
  'uuid': 'c0decbd3-4420-4833-9aad-4e7f83630d51'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 0.0,
  'uuid': '9bdcb61e-c146-4b84-920f-33ec0912228f'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 3.0,
  'uuid': '565298f2-51d3-456e-808f-cafd1177e358'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 35.0,
  'uuid': 'c72be68a-d39d-48c8-bee9-8a88c14f9c0b'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 45.0,
  'uuid': 'ab86d606-abff-428d-8ae4-53945b2764b6'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 1.0,
  'uuid': '97e7b713-9285-462c-bd52-9cc96762c11f'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 25.0,
  'uuid': '5bb839e2-4a06-4764-80f5-0368e43fb6a4'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 5.0,
  'uuid': 'f0f2c501-6d0f-4ccf-a755-ac48eebd3ae7'},
 {'nmolecules': 625,
  'concentration': 0.02,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 40.0,
  'uuid': '8a733b2a-3968-416a-ae40-df570869c163'}]


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
    
project_id = '2022-02-21-sds-on-au-111-probe-on-substrate-lateral-sliding'

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
    
# 2022-02-21: appended 0:10
# 2022-03-08: appended 10:40
# 2022-03-30: appended 40: