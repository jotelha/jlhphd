# In[20]:
import os.path
import datetime
# FireWorks functionality
from fireworks import LaunchPad
from fireworks.utilities.filepad import FilePad

from fireworks.utilities.dagflow import DAGFlow, plot_wf
# sample for plotting fraph:
#    plot_wf(wf_list[0], view='combined', labels=True, target='wf.png', bbox=(2400,2400))


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

# In[010]


probe_on_substrate_input_datasets = [
      {'concentration': 0.0175,
       'shape': 'hemicylinders',
       'uuid': 'f4154128-47d1-4378-979d-b6f49d45b117'},
      {'concentration': 0.0125,
       'shape': 'hemicylinders',
       'uuid': 'eea1c50c-5d4d-4261-8ea7-492f328af3bf'},
      {'concentration': 0.01,
       'shape': 'hemicylinders',
       'uuid': '357c6a47-866b-4645-b9b3-6b623b4fd361'},
      {'concentration': 0.0075,
       'shape': 'hemicylinders',
       'uuid': 'c17b298c-5e8f-4d34-8d4c-67462709386b'},
      {'concentration': 0.005,
       'shape': 'hemicylinders',
       'uuid': '8b40904c-e772-436a-8ae1-72e0f646f01f'},
     # {'concentration': 0.0275,
     #  'shape': 'monolayer',
     #  'uuid': 'f29d9189-4bb9-4a13-849e-75bdd7eaf347'},
     # {'concentration': 0.0225,
     #  'shape': 'monolayer',
     #  'uuid': '05dd441b-4ef5-4267-b9da-c2b82f7d6db9'},
     # {'concentration': 0.02,
     #  'shape': 'monolayer',
     #  'uuid': '319e10ef-124b-4f2f-9566-15d59c18d4b4'},
     # {'concentration': 0.015,
     #  'shape': 'monolayer',
     #  'uuid': '6ae4c3bb-9222-441e-aaa5-c5f0e5605d8a'},
     # {'concentration': 0.0125,
     #  'shape': 'monolayer',
     #  'uuid': '92383040-64bb-44fc-bc02-173944883756'},
     # {'concentration': 0.01,
     #  'shape': 'monolayer',
     #  'uuid': '5276283a-5f9a-4592-a9cc-a4866df512ba'},
     # {'concentration': 0.0075,
     #  'shape': 'monolayer',
     #  'uuid': '6f309e3f-648a-4dec-a700-f336b2f59de2'},
     # {'concentration': 0.005,
     #  'shape': 'monolayer',
     #  'uuid': '4c140e7e-8e9e-4382-abab-c53a0a4dad8e'},
     # {'concentration': 0.0025,
     #  'shape': 'monolayer',
     #  'uuid': '04fb6760-8189-487a-9b47-88b53b82a284'}
]
    
# parameters

parameter_sets = [
    # {
    #     'constant_indenter_velocity': -1.0e-4, # 10 m / s
    #     'steps': 250000,
    #     'netcdf_frequency': 100,
    #     'thermo_frequency': 100,
    #     'thermo_average_frequency': 100,
    #     'restart_frequency': 100,
    # },
    {
        'constant_indenter_velocity': -1.0e-5, # 1 m / s
        'steps': 2500000,
        'netcdf_frequency': 1000,
        'thermo_frequency': 1000,
        'thermo_average_frequency': 1000,
        'restart_frequency': 1000,
    }
]

parameter_dict_list = [{**d, **p} for p in parameter_sets for d in probe_on_substrate_input_datasets]

# In[020]


from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrateNormalApproach


project_id = '2021-02-05-sds-on-au-111-probe-and-substrate-approach'

wf_list = []

for p in parameter_dict_list:
    wfg = ProbeOnSubstrateNormalApproach(
        project_id=project_id,
        
        files_in_info={
            'data_file': {
                'query': {'uuid': p['uuid']},
                'file_name': 'default.lammps',
                'metadata_dtool_source_key': 'system',
                'metadata_fw_dest_key': 'metadata->system',
                'metadata_fw_source_key': 'metadata->system',
            },
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
            'concentration': p['concentration'],
            'shape': p['shape'],
        },
        step_specific={
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
            'dtool_push': {
                'dtool_target': '/p/project/chfr13/hoermann4/dtool/DATASETS',
                'remote_dataset': None,
            }
        }
    )
    
    fp_files = wfg.push_infiles(fp)
    wf = wfg.build_wf()
    wf_list.append(wf)
