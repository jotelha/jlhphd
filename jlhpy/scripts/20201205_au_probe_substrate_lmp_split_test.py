# In[20]:
import os.path
import datetime
# FireWorks functionality
from fireworks import LaunchPad
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


# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrateConversion

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# parameter_values = [{'n': n, 'm': n, 's': s } for n in N for s in ['monolayer','hemicylinders']][10:11]

# 2020-12-04-15-51-15-372843-charmm36gmx2lmp
# smb://jh1130/d399fa16-bda4-4df3-bbd8-cb1c2ac3a86d# In[25]

project_id = '2020-12-05-sds-on-au-111-probe-and-substrate-lmp-split-test'

wfg = SplitDatafile(
    project_id=project_id,
    files_in_info={ 
        'data_file': {  # 2020-12-04-15-51-15-372843-charmm36gmx2lmp
            'query': {'uuid': 'd399fa16-bda4-4df3-bbd8-cb1c2ac3a86d'},
            'file_name': 'default.data',
            'metadata_dtool_source_key': 'system',
            'metadata_fw_dest_key': 'metadata->system',
            'metadata_fw_source_key': 'metadata->system',
        },
    },
    #source_project_id="2020-11-25-au-111-150x150x150-fcc-substrate-creation",
    #source_step='FCCSubstrateCreationChainWorkflowGenerator:LAMMPSEquilibrationNPTWorkflowGenerator:push_dtool',
    #metadata_dtool_source_key='system->substrate',
    #metadata_fw_dest_key='metadata->system->substrate',
    #metadata_fw_source_key='metadata->system->substrate',

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
    mode='trial',
    system={},  # TODO: still needs empty placeholder to merge
    step_specific={
        'split_datafile': {
            'region_tolerance': 5.0,
            'shift_tolerance': 2.0,
        },
        'merge': {
            'z_dist': 50.0,
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/TRIAL_DATASETS',
            'remote_dataset': None,
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[666]

from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrateMinimizationAndEquilibration

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# parameter_values = [{'n': n, 'm': n, 's': s } for n in N for s in ['monolayer','hemicylinders']][10:11]

# 2020-12-04-15-51-15-372843-charmm36gmx2lmp
# smb://jh1130/d399fa16-bda4-4df3-bbd8-cb1c2ac3a86d

# 2020-12-05-21-57-01-779166-splitdatafile
# smb://jh1130/db9e4a21-8acd-46b0-a365-89ee3fdfe087, wrong shift

# 2020-12-05-22-10-30-724302-splitdatafile
# smb://jh1130/3adb67df-1fd5-4610-9f6d-2bc4bc8f09c6

# 2020-12-06-14-20-56-812735-probeonsubstrateconversion-splitdatafile
# smb://jh1130/112e1da9-011c-4ed9-ade8-2dd2ad604a4a

project_id = '2020-12-05-sds-on-au-111-probe-and-substrate-lmp-minimization-and-equilibration-test'

wfg = ProbeOnSubstrateMinimizationAndEquilibration(
    project_id=project_id,
    files_in_info={ 
        'data_file': {  
            'query': {'uuid': '112e1da9-011c-4ed9-ade8-2dd2ad604a4a'},
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
    machine='juwels_devel',
    mode='trial',
    system={
        'substrate': {
            'element': 'Au',
            'lmp': {
                'type': 11,
            }
        }
    },
    step_specific={
        'minimization': {
            'ftol': 1.0e-6,
            'maxiter': 10000,
            'maxeval': 10000,
            
            'ewald_accuracy': 1.0e-4,
            'coulomb_cutoff': 8.0,
            'neigh_delay': 2,
            'neigh_every': 1,
            'neigh_check': True,
            'skin_distance': 3.0,
        },
        'equilibration': {
            'nvt': {
                'initial_temperature': 1.0,
                'temperature': 298.0,
                'langevin_damping': 1000,
                'steps': 10000,
                'netcdf_frequency': 1,
                'thermo_frequency': 1,
                'thermo_average_frequency': 1,
            
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': True,
                'skin_distance': 3.0,
            },
            'npt': {
                'pressure': 1.0,
                'temperature': 298.0,
                'barostat_damping': 10000,
                'langevin_damping': 1000,
                'steps': 250000,
                'netcdf_frequency': 10,
                'thermo_frequency': 10,
                'thermo_average_frequency': 10,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': 1,
                'skin_distance': 3.0
            },
            'dpd': {
                'temperature': 298.0,
                'steps': 250000,
                'netcdf_frequency': 10,
                'thermo_frequency': 10,
                'thermo_average_frequency': 10,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': 1,
                'skin_distance': 3.0
            }
        },
        'split_datafile': {
            'region_tolerance': 5.0,
            'shift_tolerance': 2.0,
        },
        'merge': {
            'z_dist': 50.0,
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/TRIAL_DATASETS',
            'remote_dataset': None,
        }
    },
 )

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[666]
# 2020-12-06-00-21-37-083882-probeonsubstrateminizationequlibration
# smb://jh1130/86ed365e-67ed-494d-a980-a8210c5d81d2

from jlhpy.utilities.wf.probe_on_substrate.sub_wf_050_lammps_equilibration_nvt import LAMMPSEquilibrationNVT
# 

project_id = '2020-12-05-sds-on-au-111-probe-and-substrate-lmp-eq-nvt-test'

wfg = LAMMPSEquilibrationNVT(
    project_id=project_id,
    files_in_info={ 
        'data_file': {  # 2020-12-04-15-51-15-372843-charmm36gmx2lmp
            'query': {'uuid': '86ed365e-67ed-494d-a980-a8210c5d81d2'},
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
    machine='juwels_devel',
    mode='trial',
    system={
        'substrate': {
            'element': 'Au',
            'lmp': {
                'type': 11,
            }
        }
    },
    step_specific={
        'minimization': {
            'ftol': 1.0e-6,
            'maxiter': 10000,
            'maxeval': 10000,
            
            'ewald_accuracy': 1.0e-4,
            'coulomb_cutoff': 8.0,
            'neigh_delay': 2,
            'neigh_every': 1,
            'neigh_check': True,
            'skin_distance': 3.0,
        },
        'equilibration': {
            'nvt': {
                'initial_temperature': 1.0,
                'temperature': 1.0,
                'temperature': 298.0,
                'langevin_damping': 1000,
                'steps': 10000,
                'netcdf_frequency': 10,
                'thermo_frequency': 10,
                'thermo_average_frequency': 10,
            
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': True,
                'skin_distance': 3.0,
            },
        },
        'split_datafile': {
            'region_tolerance': 5.0,
            'shift_tolerance': 2.0,
        },
        'merge': {
            'z_dist': 50.0,
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/TRIAL_DATASETS',
            'remote_dataset': None,
        }
    },
)

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()

# In[6666]

from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrateConversionMinimizationAndEquilibration
from jlhpy.utilities.wf.mappings import psfgen_mappings_template_context

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# parameter_values = [{'n': n, 'm': n, 's': s } for n in N for s in ['monolayer','hemicylinders']][10:11]


# ProbeOnSubstrate:GromacsMinimizationEquilibrationRelaxationNoSolvation:GromacsNPTEquilibration:push_dtool
# c1a640be-694c-4fcb-b5f8-b998c229f7e8'

# 2020-12-04-15-51-15-372843-charmm36gmx2lmp
# smb://jh1130/d399fa16-bda4-4df3-bbd8-cb1c2ac3a86d

# 2020-12-05-21-57-01-779166-splitdatafile
# smb://jh1130/db9e4a21-8acd-46b0-a365-89ee3fdfe087, wrong shift

# 2020-12-05-22-10-30-724302-splitdatafile
# smb://jh1130/3adb67df-1fd5-4610-9f6d-2bc4bc8f09c6

# 

project_id = '2020-12-06-sds-on-au-111-probe-and-substrate-conversion-minimization-equilibration-test'

wfg = ProbeOnSubstrateConversionMinimizationAndEquilibration(
    project_id=project_id,
    files_in_info={
        'data_file': {
            'query': {'uuid': 'c1a640be-694c-4fcb-b5f8-b998c229f7e8'},
            'file_name': 'default.gro',
            'metadata_dtool_source_key': 'system',
            'metadata_fw_dest_key': 'metadata->system',
            'metadata_fw_source_key': 'metadata->system',
        },
        'tpr_file': {
            'query': {'uuid': 'c1a640be-694c-4fcb-b5f8-b998c229f7e8'},
            'file_name': 'default.tpr',
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
    machine='juwels_devel',
    mode='trial',
    system={
        'substrate': {
            'element': 'Au',
            'lmp': {
                'type': 11,
            }
        }
    },
    step_specific={
        'merge': {
            'z_dist': 50.0,
        },
        'psfgen': psfgen_mappings_template_context,
        'split_datafile': {
            'region_tolerance': 5.0,
            'shift_tolerance': 2.0,
        },
        'minimization': {
            'ftol': 1.0e-6,
            'maxiter': 10000,
            'maxeval': 10000,
            
            'ewald_accuracy': 1.0e-4,
            'coulomb_cutoff': 8.0,
            'neigh_delay': 2,
            'neigh_every': 1,
            'neigh_check': True,
            'skin_distance': 3.0,
        },
        'equilibration': {
            'nvt': {
                #'freeze_substrate_layer': 14.0,
                #'rigid_indenter_core_radius': 12.0,
                'initial_temperature': 1.0,
                'temperature': 298.0,
                'langevin_damping': 1000,
                'steps': 10000,
                'netcdf_frequency': 10,
                'thermo_frequency': 10,
                'thermo_average_frequency': 10,
            
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': True,
                'skin_distance': 3.0,
            },
            'npt': {
                'pressure': 1.0,
                'temperature': 298.0,
                'barostat_damping': 10000,
                'langevin_damping': 1000,
                'steps': 250000,
                'netcdf_frequency': 10,
                'thermo_frequency': 10,
                'thermo_average_frequency': 10,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': 1,
                'skin_distance': 3.0
            },
            'dpd': {
                'temperature': 298.0,
                'steps': 250000,
                'netcdf_frequency': 10,
                'thermo_frequency': 10,
                'thermo_average_frequency': 10,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': 1,
                'skin_distance': 3.0
            }
        },
        'dtool_push': {
            'dtool_target': '/p/project/chfr13/hoermann4/dtool/TRIAL_DATASETS',
            'remote_dataset': None,
        }
    },
 )

fp_files = wfg.push_infiles(fp)

wf = wfg.build_wf()