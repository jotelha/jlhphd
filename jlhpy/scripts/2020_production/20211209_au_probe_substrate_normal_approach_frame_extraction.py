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
# probe_on_substrate_input_datasets = [
     #{
     #   'uri': 'file:///p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-16-154936-probeonsubst--ionminimizationequilibrationandapproach',
     #   'uuid': '2fb51ed8-0e11-40a3-86f7-decbf823ab8f',
     #}
     
# ]

# dtool ls -v . | grep -A1 'file:' | grep -A1 -f ~/tmp/20211209.txt | xargs -n 5 echo | awk '{print $1, $4}'
datasets = """
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-39-05-810916-probeonsubst--ionminimizationequilibrationandapproach 608480be-e8a1-4485-b058-8e173b7d91b4
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-54-744236-probeonsubst--ionminimizationequilibrationandapproach b5ff6f76-8e39-4fe5-85c3-dfbc6a4f7d7f
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-11-10-50-13-324885-probeonsubst--ionminimizationequilibrationandapproach c4f94979-f778-4d84-b779-d3de7be4f8c5
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-27-168130-probeonsubst--ionminimizationequilibrationandapproach 494c0567-2556-4dd6-b155-6d70df8b7c37
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-37-10-750783-probeonsubst--ionminimizationequilibrationandapproach e3c0f64a-7384-4545-9da2-ed49c9b26d21
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-37-59-305354-probeonsubst--ionminimizationequilibrationandapproach b19dc103-507c-4eff-b05b-681c5adb784c
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-10-696766-probeonsubst--ionminimizationequilibrationandapproach 92ec5305-5a4c-41d1-abe9-ff800773c884
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-11-10-50-14-264605-probeonsubst--ionminimizationequilibrationandapproach f7399397-dd96-49d1-85cf-459bfa3404a4
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-21-703138-probeonsubst--ionminimizationequilibrationandapproach 43d9cdd7-2216-4567-9894-30a1123dc2d5
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-11-10-50-13-635304-probeonsubst--ionminimizationequilibrationandapproach 6cc02cbb-29d1-425b-8bca-34eff6a8b910
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-37-05-309454-probeonsubst--ionminimizationequilibrationandapproach b76d429e-c961-4b82-8174-cb3ae689e2ff
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-11-10-50-14-567563-probeonsubst--ionminimizationequilibrationandapproach ded34b6c-1b0b-4d3b-a31c-a834c952830b
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-39-00-276898-probeonsubst--ionminimizationequilibrationandapproach f6a0e6f5-73c3-46c0-b175-0cccf3faab05
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-32-480158-probeonsubst--ionminimizationequilibrationandapproach 9ac48ef6-fae6-4212-a8f2-445a8136bb8a
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-43-896092-probeonsubst--ionminimizationequilibrationandapproach 1e7d1d6f-0325-47bf-aa77-1296b6ac5b9d
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-37-53-836687-probeonsubst--ionminimizationequilibrationandapproach 64e7243b-99fd-434d-b83d-dcd5122518f7
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-37-15-995751-probeonsubst--ionminimizationequilibrationandapproach e954b547-e11d-443e-86d8-004f48d3f20c
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-36-54-196586-probeonsubst--ionminimizationequilibrationandapproach 169ff635-22b0-423d-9dee-a85a321a79bf
file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-38-044991-probeonsubst--ionminimizationequilibrationandapproach fa371636-acdc-4619-b7ba-ff90341fe690
"""
# 2021-12-09, omit last entry
# file://jwlogin05.juwels/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-10-07-sds-on-au-111-probe-and-substrate-merge-and-approach/2021-10-14-21-38-16-154936-probeonsubst--ionminimizationequilibrationandapproach 2fb51ed8-0e11-40a3-86f7-decbf823ab8f
# as treated already

probe_on_substrate_input_datasets = [{"uri": line.split()[0], "uuid": line.split()[1]} for line in datasets.splitlines() if line != '']
    
# parameters

# In[020]

from jlhpy.utilities.wf.building_blocks.sub_wf_lammps_trajectory_frame_extraction import LAMMPSTrajectoryFrameExtraction

project_id = '2021-12-09-sds-on-au-111-probe-and-substrate-approach-frame-extraction'

wf_list = []

for p in probe_on_substrate_input_datasets:
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
                'distance_interval': -5.0,
                'time_step': 2.0, # this should be xtractable from somewhere else, but not the case
            },
            'dtool_push': {
                'dtool_target': '/p/project/hfr21/hoermann4/dtool/PRODUCTION/2021-12-09-sds-on-au-111-probe-and-substrate-approach-frame-extraction',
                'remote_dataset': p,
                # ATTENTION: not including the names in the source datasets results in the feld being replaced by the wrong name when pulling the README
            },
            'merge': {},
            'probe_normal_approach': {},
        }
    )
    wf = wfg.build_wf()
    wf_list.append(wf)

# In[045]

# fp_files = wfg.push_infiles(fp)

# In[050]
