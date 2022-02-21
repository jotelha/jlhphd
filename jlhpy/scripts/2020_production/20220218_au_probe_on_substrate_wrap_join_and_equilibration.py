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

# DPD equilibration from '2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach' on and between hemicylinders
index_file_input_datasets = [
 {'nmolecules': 916,
  'x_shift': 50.0,
  'y_shift': 50.0,
  'uuid': '015048d4-ca53-4683-a1fd-9fa34e596f85'},
 {'nmolecules': 916,
  'x_shift': 50.0,
  'y_shift': 25.0,
  'uuid': 'ead6d394-9f97-478a-b45b-392b09ea2ff9'},
 {'nmolecules': 916,
  'x_shift': 50.0,
  'y_shift': 0.0,
  'uuid': 'dc682df5-12f3-4a5f-998e-d9932983d889'},
 {'nmolecules': 916,
  'x_shift': 50.0,
  'y_shift': -25.0,
  'uuid': 'f4d4222e-5222-4f70-b055-7bb9f43f8d50'},
 {'nmolecules': 916,
  'x_shift': 50.0,
  'y_shift': -50.0,
  'uuid': 'b66eea97-d276-4814-95b2-f88b4e6dfbb7'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'uuid': '31492349-245d-4287-b39e-e53b34fa8236'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 25.0,
  'uuid': '66e85886-cb11-4ddd-b51f-e62fa6f1ec16'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 0.0,
  'uuid': '1d185094-2dd1-4930-b2cf-d92ad64290ac'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': -25.0,
  'uuid': '4e3eae9e-6ea0-42bd-a9bd-218983dbab4e'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': -50.0,
  'uuid': 'd607ea9e-83df-4aa0-afa1-e408258f8a97'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 50.0,
  'uuid': 'ceebbc45-7360-4732-ae2f-810ebf8674d1'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '704d59f5-6856-4a5b-9ba6-1b2673b86486'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'uuid': 'cdc95c01-f3c3-4400-8f45-339c065ca0e8'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -50.0,
  'uuid': '12601538-8a94-49db-afef-e65706a9f712'},
 {'nmolecules': 916,
  'x_shift': -25.0,
  'y_shift': 50.0,
  'uuid': '762f9596-5de5-405e-8507-d7c605985321'},
 {'nmolecules': 916,
  'x_shift': -25.0,
  'y_shift': 25.0,
  'uuid': '2890f7fd-e938-48b1-aaae-f7c6db063dba'},
 {'nmolecules': 916,
  'x_shift': -25.0,
  'y_shift': 0.0,
  'uuid': '5b5f0af5-473e-4f07-a890-1c6d34904b93'},
 {'nmolecules': 916,
  'x_shift': -25.0,
  'y_shift': -25.0,
  'uuid': '6fb7c45f-210f-4de5-8b1f-7527d5a106a9'},
 {'nmolecules': 916,
  'x_shift': -25.0,
  'y_shift': -50.0,
  'uuid': '9e468f7f-d27c-4e96-9a83-0b933481c4bf'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': 50.0,
  'uuid': 'a2e83a60-99e4-4ee9-a52f-c051858f7bfb'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': 25.0,
  'uuid': 'f22269f3-6e3b-481a-b3c3-fbafa2fb1804'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': 0.0,
  'uuid': 'a9101e05-7076-4dee-9b90-70484b7d2f12'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -25.0,
  'uuid': 'e419da6c-23ec-4d3b-bf33-0754fda8c5a2'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'uuid': 'cdc04b4d-7197-4788-aeba-f85d57b31664'}]

# from "2022-02-11-sds-on-au-111-probe-and-substrate-approach-frame-extraction" on and between hemicylinders
probe_on_substrate_input_datasets = [
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 25.0,
  'uuid': '5a6d6b94-f152-477b-aec1-118d5b9b3db5'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 7.0,
  'uuid': 'db71ff66-8048-4ad4-8c73-0c0d77198e82'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 40.0,
  'uuid': 'dd9846cb-47e8-4af7-b2be-7d3123522cb3'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 10.0,
  'uuid': 'd1f3b9c2-65d1-4874-bcda-4e4b33d699ae'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 35.0,
  'uuid': '3c3d7ba7-e036-4d54-b3f3-f3cbedc10599'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 6.0,
  'uuid': 'aa74a7be-d175-4287-971c-dae46a3a8646'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 8.0,
  'uuid': 'f70cd093-f54b-48cc-bc6e-1b8e7bfba709'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 15.0,
  'uuid': '45f73aba-2bd0-4d18-9a54-c158dd8f1acc'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 30.0,
  'uuid': 'd519b2fc-8d33-4a95-a1f9-36735e299181'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 5.0,
  'uuid': '8fefe61d-4b73-4edb-974e-aef21d5673d0'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 50.0,
  'uuid': 'b8308837-d8dc-4e6e-a111-ea25591d663b'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 1.0,
  'uuid': 'f72cb792-ec89-419f-a8ed-f356b2a6a497'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 3.0,
  'uuid': '40be008c-33a1-4f74-85ee-06ea5aadb78b'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 2.0,
  'uuid': 'ba547df2-7507-4c19-86c5-671d9cf673ff'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 20.0,
  'uuid': 'd053063d-d6df-4ef3-a25d-7dbcb6a8b395'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 4.0,
  'uuid': '54b5e64b-a06d-4276-80d1-68abc530e50f'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 9.0,
  'uuid': 'bd3d5234-ca35-4aae-841b-351cf2eb3660'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 45.0,
  'uuid': 'fdb7690a-4f2f-4730-9d77-26a0cadd3c8f'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': -25.0,
  'distance': 0.0,
  'uuid': '8618854c-28c4-449d-8c74-116f479b7467'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 7.0,
  'uuid': '07eaf01a-00b4-4a6f-b5cf-e8c8ff807a22'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 2.0,
  'uuid': '69ab5719-9618-4800-9ada-00cdf40490fe'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 5.0,
  'uuid': '29fb15bc-2942-4082-a90d-6f97ade241bf'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 40.0,
  'uuid': 'de0b20a5-b085-428d-8f18-3ab8e2c39181'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 35.0,
  'uuid': 'c0227b1b-69d6-4c35-8457-c66a58848269'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 25.0,
  'uuid': 'd57bc66b-0cef-4ce4-97e9-457e943f8180'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 6.0,
  'uuid': '355e3626-cb3c-41c6-bded-4afe25b5308f'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 0.0,
  'uuid': '0433ddeb-5ec1-47ff-aa37-00c7854dde23'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 15.0,
  'uuid': '7d38debf-5c80-41d2-8060-7dc3559fb6d0'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 9.0,
  'uuid': 'd6d78a69-e023-4c2b-99e2-7333fa971d0c'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 10.0,
  'uuid': '32205129-1794-4acf-b235-7846bc114726'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 45.0,
  'uuid': '3d4260b1-7f88-4de2-90a2-a2743a20b064'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 4.0,
  'uuid': '61deeec1-00a5-4df2-902a-2b06d13f886e'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 1.0,
  'uuid': '97034880-c187-4418-8161-a338b1b50578'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 3.0,
  'uuid': '803238bc-8b96-41c7-9e4f-4fac525c1c78'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 20.0,
  'uuid': 'cee03893-74d2-40ac-b545-b13b7b1ae139'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 50.0,
  'uuid': '17f777f5-3f40-4bb2-bea5-3d4f29b04ab6'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 8.0,
  'uuid': '80a8a34a-44d4-43d8-94af-d73af8326aaa'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 30.0,
  'uuid': '26ca7ad6-cdff-499a-ad05-0b31ac637572'}
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
    
project_id = '2022-02-18-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration'

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