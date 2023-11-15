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

# DPD equilibration from '2021-12-09-sds-on-au-111-probe-and-substrate-merge-and-approach' on hemicylinder flanks
index_file_input_datasets = [{'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': 62.5,
  'uuid': '2fa35618-17f2-41a3-804b-92e230b6dd0e'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': 37.5,
  'uuid': '0f0c2a3c-a5bd-4461-a831-c5760b05cc8d'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -37.5,
  'uuid': 'cee41aaf-0b89-452f-a257-b51ca8640ce3'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'uuid': '1b962980-3805-4e08-9aba-afaa28771db5'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -37.5,
  'uuid': 'ae7a46bb-7eeb-4e6d-bb48-b87bafc04cd3'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 62.5,
  'uuid': '06c1c906-67cf-4a33-b0c5-adc04a212358'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'uuid': 'd43c4af8-74f6-4742-9f54-ab6c0ad192fb'}
]
 

# from "2022-02-12-sds-on-au-111-probe-and-substrate-approach-frame-extraction" on hemicylinder flanks
probe_on_substrate_input_datasets = [
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 8.0,
  'uuid': 'a69e846b-5722-493a-8536-e4f70aba2305'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 15.0,
  'uuid': '8774f478-2651-44ab-bed2-06f66bdb9a69'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 3.0,
  'uuid': 'fffa1a4e-774b-4445-a98d-12fc5d943f16'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 0.0,
  'uuid': '4532a5ec-c55f-4e89-b68f-e219096302a4'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 1.0,
  'uuid': 'dc3c4b08-27a3-4674-b41c-119c50835a22'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 5.0,
  'uuid': '407df8bd-aad6-450f-81bc-04f4eff76fe7'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 9.0,
  'uuid': 'a0f35ea9-601d-4f2d-b8e6-8c1704f182ed'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 4.0,
  'uuid': 'f2b9c47c-85b5-4236-84ce-481073b2fc66'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 2.0,
  'uuid': 'c7d6d88e-e176-4407-afd6-84673a2d9253'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 6.0,
  'uuid': 'de88823f-6181-499c-b7da-090e85564ced'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 7.0,
  'uuid': 'be970445-c5b8-4d81-add0-8819ac8460ed'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 25.0,
  'uuid': '1e6f1e95-aa22-457d-bd41-51e650ae323f'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 50.0,
  'uuid': '583f0d20-525b-4f91-b2e0-d0775ab815c2'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 40.0,
  'uuid': '4f3c836e-7421-424c-9f81-ac8380fd13d4'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 30.0,
  'uuid': 'c76a3d2d-1063-49dc-9c03-91a1f8238cfe'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 20.0,
  'uuid': '666769b8-c11d-4525-9933-ad73e22d5dec'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 45.0,
  'uuid': '48fd45ab-76a8-4055-a6c5-c169e69b87a0'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 10.0,
  'uuid': '667b5b73-33d6-4e1f-bfc2-35b4643fcfff'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 12.5,
  'distance': 35.0,
  'uuid': '82585374-93ac-4e8f-a8c0-87d961c18a3b'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 20.0,
  'uuid': '44bae5ba-da73-4941-9a55-84ba27fcbb2e'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 2.0,
  'uuid': '84dec808-be05-4a35-bdec-8d6987a95ea0'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 9.0,
  'uuid': 'f37a65a6-a53f-49c8-9973-e82c38125a52'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 35.0,
  'uuid': '9606da42-e403-4b46-b087-e33cc0a4d091'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 5.0,
  'uuid': 'a4394a73-a763-4071-b6fc-456cad09f07c'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 8.0,
  'uuid': '8f4c41f9-4795-4039-b59c-013e4a2f7cee'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 4.0,
  'uuid': '15cb0782-39fc-44f9-86d7-dc73856a100d'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 45.0,
  'uuid': '0a63235f-e20b-40f5-b29f-a9617423a02d'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 7.0,
  'uuid': 'c456fdfb-47a2-49aa-b506-28cd68191f96'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 3.0,
  'uuid': 'ec48b361-f8be-41f6-8c5a-bb521d5cef73'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 30.0,
  'uuid': 'db9e653b-02b2-40ba-9058-2cdb4e253ba1'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 15.0,
  'uuid': '49b6f4ff-865b-4e02-b6db-0a6c81eceb24'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 10.0,
  'uuid': 'b29fdaa6-f4e5-4594-b98b-ca4cdefc73b9'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 1.0,
  'uuid': 'a49f8ad6-91f6-49a6-bf6b-99e75d8dd9c9'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 25.0,
  'uuid': '16283acd-49e2-413d-9f3e-9bb1b937ef27'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 6.0,
  'uuid': '8f1e387a-36b8-4735-aedb-4e507e4d0731'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 50.0,
  'uuid': '3c169663-ba5e-4ef2-985f-d7e91a591919'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 0.0,
  'uuid': 'fc535dc1-d144-4c60-b84d-38b0fbb185c4'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 37.5,
  'distance': 40.0,
  'uuid': '6d718e1b-cf2f-4206-b0ef-354367ac1104'}
]

# In[]:
index_file_input_datasets_index_map = {d["nmolecules"]: i for i, d in enumerate(index_file_input_datasets)}

for d in probe_on_substrate_input_datasets:
    d['index_file_uuid'] = index_file_input_datasets[ index_file_input_datasets_index_map[d['nmolecules']]]['uuid']
    
probe_on_substrate_input_datasets_index_map = { (d["nmolecules"], d["distance"]): i for i, d in enumerate(probe_on_substrate_input_datasets) }

parameter_dict_list = [{**d} for d in probe_on_substrate_input_datasets]
# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_lateral_sliding import WrapJoinAndDPDEquilibration

from jlhpy.utilities.wf.mappings import sds_lammps_type_atom_name_mapping
# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# index = probe_on_substrate_input_datasets_index_map[0,0,25.0]
# In[25]:
    
project_id = '2022-02-19-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration'

wf_list = []

for p in parameter_dict_list:
    wfg = WrapJoinAndDPDEquilibration(
        project_id=project_id,
        
        files_in_info={
            'data_file': {
                'query': {
                    'uuid': p['uuid'],
                    'base_uri':'s3://frct-simdata', # assure to get s3 entry, not outdated ecs
                },
                'file_name': 'default.lammps',
                'metadata_dtool_source_key': 'step_specific',
                'metadata_fw_dest_key': 'metadata->step_specific',
                'metadata_fw_source_key': 'metadata->step_specific',
            },
            'index_file': {
                'query': {
                    'uuid': p['index_file_uuid'],
                    'base_uri':'s3://frct-simdata', # assure to get s3 entry, not outdated ecs
                },
                'file_name': 'groups.ndx',
                'metadata_dtool_source_key': 'system',
                'metadata_fw_dest_key': 'metadata->system',
                'metadata_fw_source_key': 'metadata->system',
            }
        },
        integrate_push=True,
        description="SDS on Au(111) probe on substrate wrap-join and DPD equilibration after frame extraction",
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
            'wrap_join' : {
                'type_name_mapping': sds_lammps_type_atom_name_mapping
            },
            'equilibration': {
                'dpd': {
                    'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                    'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the core of the indenter
                    'temperature': 298.0,
                    'steps': 10000,
                    'netcdf_frequency': 100,
                    'thermo_frequency': 100,
                    'thermo_average_frequency': 100,
                    'restart_frequency': 100,
                    
                    'ewald_accuracy': 1.0e-4,
                    'coulomb_cutoff': 8.0,
                    'neigh_delay': 2,
                    'neigh_every': 1,
                    'neigh_check': True,
                    'skin_distance': 3.0,
                    
                    'max_restarts': 5,
                },
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
    
# for wf in wf_list:
#    lp.add_wf(wf)

# 2022-02-18: all queued