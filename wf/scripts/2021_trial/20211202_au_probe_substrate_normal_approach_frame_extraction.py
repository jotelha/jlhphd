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

# from '2021-02-26-sds-on-au-111-probe-and-substrate-conversion'
probe_on_substrate_input_datasets = [
     {
        'uri': 'file:///p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-16-154936-probeonsubst--ionminimizationequilibrationandapproach',
        'uuid': '2fb51ed8-0e11-40a3-86f7-decbf823ab8f',
     }
]

    
# parameters

# In[020]

from jlhpy.utilities.wf.building_blocks.sub_wf_lammps_trajectory_frame_extraction import LAMMPSTrajectoryFrameExtraction

project_id = '2021-12-02-sds-on-au-111-probe-and-substrate-approach-frame-extraction-trial'

wf_list = []

p = probe_on_substrate_input_datasets[0]
wfg = LAMMPSTrajectoryFrameExtraction(
    project_id=project_id,

    files_in_info={
        'data_file': {
            #'query': {'uuid': p['uuid']},
            'uri': p['uri'],
            'file_name': 'default.lammps',
            'metadata_dtool_source_key': 'system',
            'metadata_fw_dest_key': 'metadata->system',
            'metadata_fw_source_key': 'metadata->system',
        },
        'trajectory_file': {
            #'query': {'uuid': p['uuid']},
            'uri': p['uri'],
            'file_name': 'joint.default.nc',
            'metadata_dtool_source_key': 'step_specific',
            'metadata_fw_dest_key': 'metadata->step_specific',
            'metadata_fw_source_key': 'metadata->step_specific',
        },
    },

    integrate_push=True,
    description="SDS on Au(111) substrate and probe frame extraction trial",
    owners=[{
        'name': 'Johannes Laurin Hörmann',
        'email': 'johannes.hoermann@imtek.uni-freiburg.de',
        'username': 'fr_jh1130',
        'orcid': '0000-0001-5867-695X'
    }],
    infile_prefix=prefix,
    machine='juwels',
    mode='trial',
    system={},
    step_specific={
        'frame_extraction': {
            #'first_frame_to_extract': 0 ,
            #'last_frame_to_extract': 2501,
            #'every_nth_frame_to_extract': 500,
            'first_distance_to_extract': 50.0,
            'last_distance_to_extract': 0.0,
            'distance_interval': -5.0,
            'time_step': 2.0, # this should be xtractable from somewhere else, but not the case
        },
        'dtool_push': {
            'dtool_target': '/p/project/hfr21/hoermann4/dtool/TRIAL_DATASETS',
            'remote_dataset': probe_on_substrate_input_datasets,
        },
        'merge': {},
        'probe_normal_approach': {},
    }
)

# In[045]

# fp_files = wfg.push_infiles(fp)
wf = wfg.build_wf()

# In[050]
