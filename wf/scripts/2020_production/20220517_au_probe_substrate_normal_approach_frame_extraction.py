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

# from '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach', on monolayers approach at -1e-5 m / s
probe_on_substrate_input_datasets = [
 {'concentration': 0.5,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '9244b291-40d0-4e9f-b985-6b2928d7b53c'},
 {'concentration': 0.75,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'c120adb1-4456-4be8-bcb8-fa6208f688a5'},
 {'concentration': 1.0,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'b3def621-636e-4d5c-a114-9aa3c7162272'},
 {'concentration': 1.5,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '64d57620-d6ff-4e39-bf3c-c7cdac01f656'},
 {'concentration': 1.75,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '24e8c4ac-7f16-4d97-8400-e4154bbded9a'},
 {'concentration': 2.0,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '5e844a04-fb3b-416e-8bce-316bc29421a0'},
 {'concentration': 2.25,
  'x_shift': 0.0,
  'y_shift': 25.0,
  'uuid': '1a5146ba-37d4-47d3-8fce-e22beca2c5a1'},
 {'concentration': 2.75,
  'x_shift': 0.0,
  'y_shift': 25.0,
  'uuid': 'fa39688d-ff02-4a39-85c0-ce1bf7cf73d3'},
 {'concentration': 2.25,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '7c16ea40-1d92-43c3-a807-254560078e2b'}]
# parameters

# In[020]

from jlhpy.utilities.wf.probe_on_substrate.sub_wf_150_lammps_trajectory_frame_extraction import LAMMPSTrajectoryFrameExtraction

project_id = '2022-05-17-sds-on-au-111-probe-and-substrate-approach-frame-extraction'

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
                'dtool_target': f'/p/project/hfr21/hoermann4/dtool/PRODUCTION/{project_id}',
                'remote_dataset': p,
                # ATTENTION: not including the names in the source datasets results in the field being replaced by the wrong name when pulling the README
            },
            'merge': {},
            'probe_normal_approach': {},
        }
    )
    wf = wfg.build_wf()
    wf_list.append(wf)

# In[045]

# All queued