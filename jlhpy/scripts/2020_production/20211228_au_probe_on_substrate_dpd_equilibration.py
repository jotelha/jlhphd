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

# DPD equilibration
index_file_input_datasets = [{'nmolecules': 916,
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

    
probe_on_substrate_input_datasets = [{'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'distance': 20.0,
  'uuid': 'be4c0669-5000-4854-bd41-80e38f553153'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'distance': 50.0,
  'uuid': '16510e57-7c80-4758-9a29-4d280bda0b4c'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'distance': 45.0,
  'uuid': '16458e5b-0188-41fb-b69a-f4cef50dbd00'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'distance': 30.0,
  'uuid': '165521a0-b72a-4fdb-9689-c5edf804028d'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'distance': 15.0,
  'uuid': '7782f6f5-122b-454e-8425-733769bd8079'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'distance': 10.0,
  'uuid': '14d5bd15-13ba-47d5-b94d-6c833ace8324'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'distance': 5.0,
  'uuid': '25a32860-724f-4c12-ac75-aaae79e95e92'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'distance': 25.0,
  'uuid': '8b8f7548-9673-4a72-9f62-5d406f7cda58'},
 {'nmolecules': 916,
  'x_shift': -50.0,
  'y_shift': -50.0,
  'distance': 40.0,
  'uuid': '93e0787d-1450-436d-a534-5ff89c771c90'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 45.0,
  'uuid': 'ad551593-4c46-44cd-b492-24509371545b'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 20.0,
  'uuid': 'a3869a22-5202-4551-b2a2-b2ea5e24d569'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 35.0,
  'uuid': '5df834df-b134-465a-b28b-54acece8186b'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 25.0,
  'uuid': '6cebc80c-71e3-4e69-bf4b-a508f8a0a59e'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 0.0,
  'uuid': '2d05e0be-99f2-4d63-a779-4716e227873d'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 40.0,
  'uuid': '4fb8ad8e-7129-40de-9b05-92d66abb2116'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 30.0,
  'uuid': '6ed943a0-30ce-4c98-9333-6ffdad2de9d3'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 15.0,
  'uuid': '2713ddde-958e-4307-8f35-5c1400f8b657'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 5.0,
  'uuid': '4b46368f-8055-4a6e-a7e6-79b622442860'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 10.0,
  'uuid': '70c7bfd1-0181-4e36-a34d-760332b5aefa'},
 {'nmolecules': 916,
  'x_shift': 25.0,
  'y_shift': 50.0,
  'distance': 50.0,
  'uuid': '487148c8-c569-481a-b654-8275c16fef77'},
 {'nmolecules': 916,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'distance': 25.0,
  'uuid': '221f354c-dc80-4e3c-88ac-8c1a743bff9c'}]

index_file_input_datasets_index_map = { (d["x_shift"], d["y_shift"]): i for i, d in enumerate(index_file_input_datasets) }

for d in probe_on_substrate_input_datasets:
    d['index_file_uuid'] = index_file_input_datasets[ index_file_input_datasets_index_map[(d['x_shift'],d['y_shift'])]]['uuid']
    
probe_on_substrate_input_datasets_index_map = { (d["x_shift"], d["y_shift"], d["distance"]): i for i, d in enumerate(probe_on_substrate_input_datasets) }

parameter_dict_list = [{**d} for d in probe_on_substrate_input_datasets]
# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.sub_wf_190_lammps_equilibration_dpd import LAMMPSEquilibrationDPD

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

index = probe_on_substrate_input_datasets_index_map[0,0,25.0]
# In[25]:
    
project_id = '2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration'

wf_list = []
# for c, substrate_uuid, probe_uuid in probe_on_substrate_input_datasets:
# c = 0.03
for p in parameter_dict_list[:-1]:
    wfg = LAMMPSEquilibrationDPD(
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
                'dtool_target': '/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-12-27-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration',
                'remote_dataset': None,
            }
        }
    )
    fp_files = wfg.push_infiles(fp)
    wf = wfg.build_wf()
    wf_list.append(wf)
    
# In[]:
 
    
# for wf in wf_list:
#    lp.add_wf(wf)

# 2021-12-28: submitted probe_on_substrate_input_datasets_index_map[0,0,25.0], index 20
# 2021-12-29: submitted probe_on_substrate_input_datasets_index_map[25.0,50.0,50.0], index 19 / -1
#   index 18 / -2, running
#   index 17 / -3, running
# index 0:-3, appended
