# In[20]:
import os.path
import datetime
# FireWorks functionality
from fireworks import LaunchPad
from fireworks.utilities.filepad import FilePad

from fireworks.utilities.dagflow import DAGFlow, plot_wf


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


# In[6666]

# All

from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_insertion import ProbeOnSubstrateMergeConversionMinimizationAndEquilibration
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

project_id = '2020-12-07-sds-on-au-111-probe-and-substrate-merge-conversion-minimization-equilibration-test'

wfg = ProbeOnSubstrateMergeConversionMinimizationAndEquilibration(
    project_id=project_id,
    files_in_info={
        'substrate_data_file': {  #  506 SDS
            'query': {'uuid': '6d5fe574-3359-4580-ae2d-eeda9ec5b926'},
            'file_name': 'default.gro',
            'metadata_dtool_source_key': 'system->substrate',
            'metadata_fw_dest_key': 'metadata->system->substrate',
            'metadata_fw_source_key': 'metadata->system->substrate',
        },
        'probe_data_file': {  # 197
            'query': {'uuid': '1bc8bb4a-f4cf-4e4f-96ee-208b01bc3d02'},
            'file_name': 'default.gro',
            'metadata_dtool_source_key': 'system->indenter',
            'metadata_fw_dest_key': 'metadata->system->indenter',
            'metadata_fw_source_key': 'metadata->system->indenter',
        }
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
    mode='trial',
    system={
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': None,
            'reference_atom': {
                'name': 'NA',
            },
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': None,
            'connector_atom': {
                'index': 2,
            },
            'head_atom': {
                'name': 'S',
                'index': 1,
            },
            'tail_atom': {
                'name': 'C12',
                'index': 39,
            },
            'aggregates': {
                'shape': None,
            }
        },
        'indenter': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
            'element': 'Au',
            'lmp': {
                'type': 11,
            }
      
        },
        'solvent': {
            'name': 'H2O',
            'resname': 'SOL',
            'reference_atom': {
                'name': 'OW',
            },
            'height': 180.0,
        }
    },
    step_specific={
        'merge': {
            'tol': 2.0,
            'z_dist': 50.0,
            'x_shift': 0.0,
            'y_shift': 0.0,
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
                'initial_temperature': 1.0,
                'temperature': 298.0,
                'langevin_damping': 1000,
                'steps': 10000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
            
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
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': 1,
                'skin_distance': 3.0
            },
            'dpd': {
                'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the ore of the indenter
                'temperature': 298.0,
                'steps': 250000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
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

# In[010]

# Minimization

from jlhpy.utilities.wf.probe_on_substrate.sub_wf_040_lammps_minimization import LAMMPSMinimization
from jlhpy.utilities.wf.mappings import psfgen_mappings_template_context

# 2020-12-06-14-20-56-812735-probeonsubstrateconversion-splitdatafile
# 112e1da9-011c-4ed9-ade8-2dd2ad604a4a

project_id = '2020-12-07-sds-on-au-111-probe-and-substrate-minimization-test'

wfg = LAMMPSMinimization(
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
    machine='juwels',
    mode='trial',
    system={
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': None,
            'reference_atom': {
                'name': 'NA',
            },
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': None,
            'connector_atom': {
                'index': 2,
            },
            'head_atom': {
                'name': 'S',
                'index': 1,
            },
            'tail_atom': {
                'name': 'C12',
                'index': 39,
            },
            'aggregates': {
                'shape': None,
            }
        },
        'indenter': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
            'element': 'Au',
            'lmp': {
                'type': 11,
            }
      
        },
        'solvent': {
            'name': 'H2O',
            'resname': 'SOL',
            'reference_atom': {
                'name': 'OW',
            },
            'height': 180.0,
        }
    },
    step_specific={
        'merge': {
            'tol': 2.0,
            'z_dist': 50.0,
            'x_shift': 0.0,
            'y_shift': 0.0,
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
                'initial_temperature': 1.0,
                'temperature': 298.0,
                'langevin_damping': 1000,
                'steps': 10000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
            
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
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': 1,
                'skin_distance': 3.0
            },
            'dpd': {
                'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the ore of the indenter
                'temperature': 298.0,
                'steps': 250000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
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


# In[020]

# minimized
# 95f5317d-503e-4be1-a51f-bfb40ddd4662

from jlhpy.utilities.wf.probe_on_substrate.sub_wf_050_lammps_equilibration_nvt import LAMMPSEquilibrationNVT
from jlhpy.utilities.wf.mappings import psfgen_mappings_template_context


project_id = '2020-12-07-sds-on-au-111-probe-and-substrate-equilibration-nvt-test'

wfg = LAMMPSEquilibrationNVT(
    project_id=project_id,
    files_in_info={
        'data_file': { 
            'query': {'uuid': '95f5317d-503e-4be1-a51f-bfb40ddd4662'},
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
    mode='trial',
    system={
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': None,
            'reference_atom': {
                'name': 'NA',
            },
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': None,
            'connector_atom': {
                'index': 2,
            },
            'head_atom': {
                'name': 'S',
                'index': 1,
            },
            'tail_atom': {
                'name': 'C12',
                'index': 39,
            },
            'aggregates': {
                'shape': None,
            }
        },
        'indenter': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
            'element': 'Au',
            'lmp': {
                'type': 11,
            }
      
        },
        'solvent': {
            'name': 'H2O',
            'resname': 'SOL',
            'reference_atom': {
                'name': 'OW',
            },
            'height': 180.0,
        }
    },
    step_specific={
        'merge': {
            'tol': 2.0,
            'z_dist': 50.0,
            'x_shift': 0.0,
            'y_shift': 0.0,
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
                'initial_temperature': 1.0,
                'temperature': 298.0,
                'langevin_damping': 1000,
                'steps': 10000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
            
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
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': 1,
                'skin_distance': 3.0
            },
            'dpd': {
                'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the ore of the indenter
                'temperature': 298.0,
                'steps': 250000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
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

# In[050]

# NVT equlibrated
# 9d2fcef6-d943-498b-8d0f-1264f068c094


from jlhpy.utilities.wf.probe_on_substrate.sub_wf_060_lammps_equilibration_npt import LAMMPSEquilibrationNPT
from jlhpy.utilities.wf.mappings import psfgen_mappings_template_context


project_id = '2020-12-07-sds-on-au-111-probe-and-substrate-equilibration-npt-test'

wfg = LAMMPSEquilibrationNPT(
    project_id=project_id,
    files_in_info={
        'data_file': { 
            'query': {'uuid': '9d2fcef6-d943-498b-8d0f-1264f068c094'},
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
    mode='trial',
    system={
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': None,
            'reference_atom': {
                'name': 'NA',
            },
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': None,
            'connector_atom': {
                'index': 2,
            },
            'head_atom': {
                'name': 'S',
                'index': 1,
            },
            'tail_atom': {
                'name': 'C12',
                'index': 39,
            },
            'aggregates': {
                'shape': None,
            }
        },
        'indenter': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
            'element': 'Au',
            'lmp': {
                'type': 11,
            }
      
        },
        'solvent': {
            'name': 'H2O',
            'resname': 'SOL',
            'reference_atom': {
                'name': 'OW',
            },
            'height': 180.0,
        }
    },
    step_specific={
        'merge': {
            'tol': 2.0,
            'z_dist': 50.0,
            'x_shift': 0.0,
            'y_shift': 0.0,
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
                'initial_temperature': 1.0,
                'temperature': 298.0,
                'langevin_damping': 1000,
                'steps': 10000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
            
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
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': 1,
                'skin_distance': 3.0
            },
            'dpd': {
                'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the ore of the indenter
                'temperature': 298.0,
                'steps': 250000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
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

# In[060]

# NVT equlibrated
# 9d2fcef6-d943-498b-8d0f-1264f068c094


from jlhpy.utilities.wf.probe_on_substrate.sub_wf_060_lammps_equilibration_npt import LAMMPSEquilibrationDPD
from jlhpy.utilities.wf.mappings import psfgen_mappings_template_context


project_id = '2020-12-07-sds-on-au-111-probe-and-substrate-equilibration-dpd-on-nvt-test'

wfg = LAMMPSEquilibrationDPD(
    project_id=project_id,
    files_in_info={
        'data_file': { 
            'query': {'uuid': '9d2fcef6-d943-498b-8d0f-1264f068c094'},
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
    mode='trial',
    system={
        'counterion': {
            'name': 'NA',
            'resname': 'NA',
            'nmolecules': None,
            'reference_atom': {
                'name': 'NA',
            },
        },
        'surfactant': {
            'name': 'SDS',
            'resname': 'SDS',
            'nmolecules': None,
            'connector_atom': {
                'index': 2,
            },
            'head_atom': {
                'name': 'S',
                'index': 1,
            },
            'tail_atom': {
                'name': 'C12',
                'index': 39,
            },
            'aggregates': {
                'shape': None,
            }
        },
        'indenter': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
        },
        'substrate': {
            'name': 'AUM',
            'resname': 'AUM',
            'reference_atom': {
                'name': 'AU',
            },
            'element': 'Au',
            'lmp': {
                'type': 11,
            }
      
        },
        'solvent': {
            'name': 'H2O',
            'resname': 'SOL',
            'reference_atom': {
                'name': 'OW',
            },
            'height': 180.0,
        }
    },
    step_specific={
        'merge': {
            'tol': 2.0,
            'z_dist': 50.0,
            'x_shift': 0.0,
            'y_shift': 0.0,
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
                'initial_temperature': 1.0,
                'temperature': 298.0,
                'langevin_damping': 1000,
                'steps': 10000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
            
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
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': 1,
                'skin_distance': 3.0
            },
            'dpd': {
                'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the ore of the indenter
                'temperature': 298.0,
                'steps': 250000,
                'netcdf_frequency': 1000,
                'thermo_frequency': 1000,
                'thermo_average_frequency': 1000,
                
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
