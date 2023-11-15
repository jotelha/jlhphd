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

# from '2021-02-05-sds-on-au-111-probe-and-substrate-approach', monolayer approach at -1e-5 m / s
probe_on_substrate_input_datasets = [
     {'nmolecules': 156,
      'concentration': 0.005,
      'uuid': '477ae425-9a75-4977-92fd-e786d829f525'},
     {'nmolecules': 235,
      'concentration': 0.0075,
      'uuid': '0e6af9b0-c5a4-451f-bfb7-2f2b37727048'},
     {'nmolecules': 313,
      'concentration': 0.01,
      'uuid': '7c4656b4-ea80-4910-8321-1420c711e40b'},
     {'nmolecules': 390,
      'concentration': 0.0125,
      'uuid': '74ede4bb-e877-406c-93cd-b4136b100b35'},
     {'nmolecules': 625,
      'concentration': 0.02,
      'uuid': '623ee87d-d25c-401f-a7a9-e55bda3c9652'},
     {'nmolecules': 703,
      'concentration': 0.0225,
      'uuid': '833b057f-e934-4fe1-81de-e4bb180a3532'},
]
    
# parameters

# In[020]

from jlhpy.utilities.wf.probe_on_substrate.sub_wf_150_lammps_trajectory_frame_extraction import LAMMPSTrajectoryFrameExtraction

project_id = '2022-02-10-sds-on-au-111-probe-and-substrate-approach-frame-extraction'

wf_list = []

for p in probe_on_substrate_input_datasets:
    wfg = LAMMPSTrajectoryFrameExtraction(
        project_id=project_id,
    
        files_in_info={
            'data_file': {
                'query': {'uuid': p['uuid']},
                'file_name': 'default.lammps',
                'metadata_dtool_source_key': 'system',
                'metadata_fw_dest_key': 'metadata->system',
                'metadata_fw_source_key': 'metadata->system',
            },
            'trajectory_file': {
                'query': {'uuid': p['uuid']},
                'file_name': 'joint.default.nc',
                'metadata_dtool_source_key': 'step_specific',
                'metadata_fw_dest_key': 'metadata->step_specific',
                'metadata_fw_source_key': 'metadata->step_specific',
            },
        },
    
        integrate_push=True,
        description="SDS on Au(111) substrate and probe frame extraction",
        owners=[{
            'name': 'Johannes Laurin Hörmann',
            'email': 'johannes.hoermann@imtek.uni-freiburg.de',
            'username': 'fr_jh1130',
            'orcid': '0000-0001-5867-695X'
        }],
        infile_prefix=prefix,
        machine='juwels',
        mode='production',
        system={},
        step_specific={
            'frame_extraction': {
                'first_distance_to_extract': 50.0,
                'last_distance_to_extract': 0.0,
                'distance_interval': -1.0,
                'time_step': 2.0, # this should be extractable from somewhere else, but not the case
            },
            'dtool_push': {
                'dtool_target': '/p/project/hfr21/hoermann4/dtool/PRODUCTION/2022-02-10-sds-on-au-111-probe-and-substrate-approach-frame-extraction',
                'remote_dataset': p,
                # ATTENTION: not including the names in the source datasets results in the feld being replaced by the wrong name when pulling the README
            },
            'merge': { # from 2020-12-23-sds-on-au-111-probe-and-substrate-conversion, z_dist needed
                'tol': 2.0,
                'z_dist': 50.0,
                'x_shift': 0.0,
                'y_shift': 0.0,
            },
            'probe_normal_approach': {},
        }
    )
    wf = wfg.build_wf()
    wf_list.append(wf)

# In[045]

