# -*- coding: utf-8 -*-
"""Generic GROMACS trajectory analyisis blocks."""

import datetime

from abc import ABC, abstractmethod

from fireworks import Firework
from fireworks.user_objects.firetasks.fileio_tasks import ArchiveDirTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask, PickledPyEnvTask

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator

from imteksimfw.fireworks.utilities.serialize import serialize_module_obj
import jlhpy.utilities.analysis.rdf as serial_rdf
# import jlhpy.utilities.analysis.mpi_rdf as parallel_rdf
import jlhpy.utilities.analysis.msd as serial_msd
# import jlhpy.utilities.analysis.mpi_msd as parallel_msd


class GromacsTrajectoryAnalysis(WorkflowGenerator):
    """
    Abstract base class for partial analysis worklfow.

    analysis dynamic infiles:
        no pull stub implemented

    - data_file:       default.gro
    - trajectory_file: default.xtc

    Implementation must provide rmsd_list and rdf_list.
    """

    @property
    @abstractmethod
    def rmsd_list(self) -> list:
        ...

    @property
    @abstractmethod
    def rdf_list(self) -> list:
        ...

    def main(self, fws_root=[]):
        fw_list = []

        # compute rdf
        # -----------

        step_label = self.get_step_label('analysis_rdf')

        files_in = {
            'data_file': 'default.gro',
            'trajectory_file': 'default.xtc',
        }
        files_out = {
            f['file_label']: f['file_name'] for f in self.rdf_list
        }

        func_str = serialize_module_obj(serial_rdf.atom_atom_rdf)

        fts_rdf = []
        for rdf in self.rdf_list:
            fts_rdf.append(PickledPyEnvTask(
                func=func_str,
                args=['default.gro', 'default.xtc', rdf['file_name']],
                kwargs_inputs={
                    'atom_name_a': rdf['atom_name_a'],
                    'atom_name_b': rdf['atom_name_b'],
                },
                env='mdanalysis',
                stderr_file='file.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
            ))

        fw_rdf = Firework(fts_rdf,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    **self.hpc_specs['single_core_job_queueadapter_defaults']
                },
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':     step_label,
                     **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_rdf)

        # compute rmsd
        # ------------

        step_label = self.get_step_label('analysis_rmsd')

        files_in = {
            'data_file': 'default.gro',
            'trajectory_file': 'default.xtc',
        }
        files_out = {
            f['file_label']: f['file_name'] for f in self.rmsd_list
        }

        func_str = serialize_module_obj(serial_msd.atom_rmsd)

        fts_rmsd = []
        for rmsd in self.rmsd_list:
            fts_rmsd.append(PickledPyEnvTask(
                func=func_str,
                args=['default.gro', 'default.xtc', rmsd['file_name']],
                kwargs_inputs={
                    'atom_name': rmsd['atom_name'],
                },
                env='mdanalysis',
                stderr_file='std.err',
                stdout_file='std.out',
                store_stdout=True,
                store_stderr=True,
            ))

        fw_rmsd = Firework(fts_rmsd,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    **self.hpc_specs['single_core_job_queueadapter_defaults']
                },
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':     step_label,
                     **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_rmsd)

        return fw_list, [fw_rdf, fw_rmsd], [fw_rdf, fw_rmsd]


class GromacsParallelTrajectoryAnalysis(GromacsTrajectoryAnalysis):
    """
    Abstract base class for parallel partial analysis worklfow.

    analysis dynamic infiles:
        no pull stub implemented

    - data_file:       default.gro
    - trajectory_file: default.xtc

    Implementation must provide rmsd_list and rdf_list.
    """

    def main(self, fws_root=[]):
        fw_list = []

        # compute rdf
        # -----------

        step_label = self.get_step_label('analysis_rdf')

        files_in = {
            'data_file': 'default.gro',
            'trajectory_file': 'default.xtc',
        }
        files_out = {
            f['file_label']: f['file_name'] for f in self.rdf_list
        }

        fts_rdf = []
        for rdf in self.rdf_list:
            fts_rdf.append(CmdTask(
                cmd='python',
                opt=[
                    '-m', 'mpi4py.futures',  # mpi4py.futures wrapper necessary for static process allocation
                    '-m', 'imteksimcs.mpi4py.mpi_pool_executor',
                    'imteksimcs.GROMACS.gmx_mpi_rdf.atom_atom_rdf',
                    'default.gro',
                    'default.xtc',
                    rdf['file_name'],
                    {'key': rdf['atom_name_a']},
                    {'key': rdf['atom_name_b']},
                ],
                env='mdanalysis',
                stderr_file=rdf['file_label'] + '.err',
                stdout_file=rdf['file_label'] + '.out',
                stdlog_file=rdf['file_label'] + '.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=True,
            ))

        fw_rdf = Firework(fts_rdf,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    **self.hpc_specs['single_node_job_queueadapter_defaults']
                },
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':     step_label,
                     **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_rdf)

        # compute rmsd
        # ------------

        step_label = self.get_step_label('analysis_rmsd')

        files_in = {
            'data_file': 'default.gro',
            'trajectory_file': 'default.xtc',
        }
        files_out = {
            f['file_label']: f['file_name'] for f in self.rmsd_list
        }

        fts_rmsd = []
        for rmsd in self.rmsd_list:
            fts_rmsd.append(CmdTask(
                cmd='python',
                opt=[
                    '-m', 'mpi4py.futures',  # mpi4py.futures wrapper necessary for static process allocation
                    '-m', 'imteksimcs.mpi4py.mpi_pool_executor',
                    'imteksimcs.GROMACS.gmx_mpi_msd.atom_rmsd',
                    'default.gro',
                    'default.xtc',
                    rmsd['file_name'],
                    {'key': rmsd['atom_name']},
                ],
                env='mdanalysis',
                stderr_file=rmsd['file_label'] + '.err',
                stdout_file=rmsd['file_label'] + '.out',
                stdlog_file=rmsd['file_label'] + '.log',
                store_stdout=True,
                store_stderr=True,
                fizzle_bad_rc=True,
            ))

        fw_rmsd = Firework(fts_rmsd,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_queue_category'],
                '_queueadapter': {
                    **self.hpc_specs['single_node_job_queueadapter_defaults']
                },
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':     step_label,
                     **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_rmsd)

        return fw_list, [fw_rdf, fw_rmsd], [fw_rdf, fw_rmsd]


class GromacsMinimalTrajectoryAnalysis(
        GromacsParallelTrajectoryAnalysis):
    """
    Implements partial analysis worklfow only.

    analysis dynamic infiles:
    - data_file:       default.gro
    - trajectory_file: default.xtc

    analysis fw_spec inputs:
    - metadata->system->counterion->reference_atom->name
    - metadata->system->substrate->reference_atom->name
    - metadata->system->surfactant->head_atom->name
    - metadata->system->surfactant->tail_atom->name

    anaylsis outfiles:
    - counterion_counterion_rdf: counterion_counterion_rdf.txt
        tagged as {'metadata->type': 'counterion_counterion_rdf'}
    - counterion_substrate_rdf: counterion_substrate_rdf.txt
        tagged as {'metadata->type': 'counterion_substrate_rdf'}
    - counterion_surfactant_head_rdf: counterion_surfactant_head_rdf.txt
        tagged as {'metadata->type': 'counterion_surfactant_head_rdf'}
    - counterion_surfactant_tail_rdf: counterion_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'counterion_surfactant_tail_rdf'}

    - substrate_surfactant_head_rdf: substrate_surfactant_head_rdf.txt
        tagged as {'metadata->type': 'subtrate_surfactant_head_rdf'}
    - substrate_surfactant_tail_rdf: substrate_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'subtrate_surfactant_tail_rdf'}

    - surfactant_head_surfactant_head_rdf: surfactant_head_surfactant_head_rdf.txt
            tagged as {'metadata->type': 'surfactant_head_surfactant_head_rdf'}
    - surfactant_head_surfactant_tail_rdf: surfactant_head_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'surfactant_head_surfactant_tail_rdf'}

    - surfactant_tail_surfactant_tail_rdf: surfactant_tail_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'surfactant_tail_surfactant_tail_rdf'}
    """
    @property
    def rmsd_list(self):
        return [
            {
                'file_label': 'counterion_rmsd',
                'file_name': 'counterion_rmsd.txt',
                'type_label': 'counterion_rmsd',
                'atom_name': 'metadata->system->counterion->reference_atom->name'},
            {
                'file_label': 'substrate_rmsd',
                'file_name': 'substrate_rmsd.txt',
                'type_label': 'substrate_rmsd',
                'atom_name': 'metadata->system->substrate->reference_atom->name' },
            {
                'file_label': 'surfactant_head_rmsd',
                'file_name': 'surfactant_head_rmsd.txt',
                'type_label': 'surfactant_head_rmsd',
                'atom_name': 'metadata->system->surfactant->head_atom->name' },
            {
                'file_label': 'surfactant_tail_rmsd',
                'file_name': 'surfactant_tail_rmsd.txt',
                'type_label': 'surfactant_tail_rmsd',
                'atom_name': 'metadata->system->surfactant->tail_atom->name' },
        ]

    # use regex
    #    - ([^\s:]+): ([^\s]+)\s+tagged as {'metadata->type': '([^\s']+)'}
    # and reqplacement pattern
    #    {'file_label': '$1', 'file_name': '$2', 'type_label': '$3', 'atom_name_a': a, 'atom_name_b': b},
    # on help text
    @property
    def rdf_list(self):
        return [
            {
                'file_label': 'counterion_counterion_rdf',
                'file_name': 'counterion_counterion_rdf.txt',
                'type_label': 'counterion_counterion_rdf',
                'atom_name_a': 'metadata->system->counterion->reference_atom->name',
                'atom_name_b': 'metadata->system->counterion->reference_atom->name'},
            {
                'file_label': 'counterion_substrate_rdf',
                'file_name': 'counterion_substrate_rdf.txt',
                'type_label': 'counterion_substrate_rdf',
                'atom_name_a': 'metadata->system->counterion->reference_atom->name',
                'atom_name_b': 'metadata->system->substrate->reference_atom->name' },
            {
                'file_label': 'counterion_surfactant_head_rdf',
                'file_name': 'counterion_surfactant_head_rdf.txt',
                'type_label': 'counterion_surfactant_head_rdf',
                'atom_name_a': 'metadata->system->counterion->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->head_atom->name' },
            {
                'file_label': 'counterion_surfactant_tail_rdf',
                'file_name': 'counterion_surfactant_tail_rdf.txt',
                'type_label': 'counterion_surfactant_tail_rdf',
                'atom_name_a': 'metadata->system->counterion->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->tail_atom->name' },

            {
                'file_label': 'substrate_surfactant_head_rdf',
                'file_name': 'substrate_surfactant_head_rdf.txt',
                'type_label': 'subtrate_surfactant_head_rdf',
                'atom_name_a': 'metadata->system->substrate->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->head_atom->name' },
            {
                'file_label': 'substrate_surfactant_tail_rdf',
                'file_name': 'substrate_surfactant_tail_rdf.txt',
                'type_label': 'subtrate_surfactant_tail_rdf',
                'atom_name_a': 'metadata->system->substrate->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->tail_atom->name' },

            {
                'file_label': 'surfactant_head_surfactant_head_rdf',
                'file_name': 'surfactant_head_surfactant_head_rdf.txt',
                'type_label': 'surfactant_head_surfactant_head_rdf',
                'atom_name_a': 'metadata->system->surfactant->head_atom->name',
                'atom_name_b': 'metadata->system->surfactant->head_atom->name' },
            {
                'file_label': 'surfactant_head_surfactant_tail_rdf',
                'file_name': 'surfactant_head_surfactant_tail_rdf.txt',
                'type_label': 'surfactant_head_surfactant_tail_rdf',
                'atom_name_a': 'metadata->system->surfactant->head_atom->name',
                'atom_name_b': 'metadata->system->surfactant->tail_atom->name' },

            {
                'file_label': 'surfactant_tail_surfactant_tail_rdf',
                'file_name': 'surfactant_tail_surfactant_tail_rdf.txt',
                'type_label': 'surfactant_tail_surfactant_tail_rdf',
                'atom_name_a': 'metadata->system->surfactant->tail_atom->name',
                'atom_name_b': 'metadata->system->surfactant->tail_atom->name'},
        ]


class GromacsVacuumTrajectoryAnalysis(
        GromacsParallelTrajectoryAnalysis):
    """
    Implements partial analysis worklfow only.

    analysis dynamic infiles:
        no pull stub implemented

    - data_file:       default.gro
    - trajectory_file: default.xtc

    analysis fw_spec inputs:
    - metadata->system->counterion->reference_atom->name
    - metadata->system->substrate->reference_atom->name
    - metadata->system->surfactant->head_atom->name
    - metadata->system->surfactant->tail_atom->name

    anaylsis outfiles:
    - counterion_counterion_rdf: counterion_counterion_rdf.txt
        tagged as {'metadata->type': 'counterion_counterion_rdf'}
    - counterion_substrate_rdf: counterion_substrate_rdf.txt
        tagged as {'metadata->type': 'counterion_substrate_rdf'}
    - counterion_surfactant_head_rdf: counterion_surfactant_head_rdf.txt
        tagged as {'metadata->type': 'counterion_surfactant_head_rdf'}
    - counterion_surfactant_tail_rdf: counterion_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'counterion_surfactant_tail_rdf'}

    - substrate_substrate_rdf: substrate_substrate_rdf.txt
        tagged as {'metadata->type': 'substrate_substrate_rdf'}
    - substrate_surfactant_head_rdf: substrate_surfactant_head_rdf.txt
        tagged as {'metadata->type': 'subtrate_surfactant_head_rdf'}
    - substrate_surfactant_tail_rdf: substrate_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'subtrate_surfactant_tail_rdf'}

    - surfactant_head_surfactant_head_rdf: surfactant_head_surfactant_head_rdf.txt
            tagged as {'metadata->type': 'surfactant_head_surfactant_head_rdf'}
    - surfactant_head_surfactant_tail_rdf: surfactant_head_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'surfactant_head_surfactant_tail_rdf'}

    - surfactant_tail_surfactant_tail_rdf: surfactant_tail_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'surfactant_tail_surfactant_tail_rdf'}
    """
    @property
    def rmsd_list(self):
        return [
            {
                'file_label': 'counterion_rmsd',
                'file_name': 'counterion_rmsd.txt',
                'type_label': 'counterion_rmsd',
                'atom_name': 'metadata->system->counterion->reference_atom->name'},
            {
                'file_label': 'substrate_rmsd',
                'file_name': 'substrate_rmsd.txt',
                'type_label': 'substrate_rmsd',
                'atom_name': 'metadata->system->substrate->reference_atom->name' },
            {
                'file_label': 'surfactant_head_rmsd',
                'file_name': 'surfactant_head_rmsd.txt',
                'type_label': 'surfactant_head_rmsd',
                'atom_name': 'metadata->system->surfactant->head_atom->name' },
            {
                'file_label': 'surfactant_tail_rmsd',
                'file_name': 'surfactant_tail_rmsd.txt',
                'type_label': 'surfactant_tail_rmsd',
                'atom_name': 'metadata->system->surfactant->tail_atom->name' },
        ]

    # use regex
    #    - ([^\s:]+): ([^\s]+)\s+tagged as {'metadata->type': '([^\s']+)'}
    # and reqplacement pattern
    #    {'file_label': '$1', 'file_name': '$2', 'type_label': '$3', 'atom_name_a': a, 'atom_name_b': b},
    # on help text
    @property
    def rdf_list(self):
        return [
            {
                'file_label': 'counterion_counterion_rdf',
                'file_name': 'counterion_counterion_rdf.txt',
                'type_label': 'counterion_counterion_rdf',
                'atom_name_a': 'metadata->system->counterion->reference_atom->name',
                'atom_name_b': 'metadata->system->counterion->reference_atom->name'},
            {
                'file_label': 'counterion_substrate_rdf',
                'file_name': 'counterion_substrate_rdf.txt',
                'type_label': 'counterion_substrate_rdf',
                'atom_name_a': 'metadata->system->counterion->reference_atom->name',
                'atom_name_b': 'metadata->system->substrate->reference_atom->name' },
            {
                'file_label': 'counterion_surfactant_head_rdf',
                'file_name': 'counterion_surfactant_head_rdf.txt',
                'type_label': 'counterion_surfactant_head_rdf',
                'atom_name_a': 'metadata->system->counterion->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->head_atom->name' },
            {
                'file_label': 'counterion_surfactant_tail_rdf',
                'file_name': 'counterion_surfactant_tail_rdf.txt',
                'type_label': 'counterion_surfactant_tail_rdf',
                'atom_name_a': 'metadata->system->counterion->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->tail_atom->name' },

            {
                'file_label': 'substrate_substrate_rdf',
                'file_name': 'substrate_substrate_rdf.txt',
                'type_label': 'substrate_substrate_rdf',
                'atom_name_a': 'metadata->system->substrate->reference_atom->name',
                'atom_name_b': 'metadata->system->substrate->reference_atom->name' },
            {
                'file_label': 'substrate_surfactant_head_rdf',
                'file_name': 'substrate_surfactant_head_rdf.txt',
                'type_label': 'subtrate_surfactant_head_rdf',
                'atom_name_a': 'metadata->system->substrate->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->head_atom->name' },
            {
                'file_label': 'substrate_surfactant_tail_rdf',
                'file_name': 'substrate_surfactant_tail_rdf.txt',
                'type_label': 'subtrate_surfactant_tail_rdf',
                'atom_name_a': 'metadata->system->substrate->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->tail_atom->name' },

            {
                'file_label': 'surfactant_head_surfactant_head_rdf',
                'file_name': 'surfactant_head_surfactant_head_rdf.txt',
                'type_label': 'surfactant_head_surfactant_head_rdf',
                'atom_name_a': 'metadata->system->surfactant->head_atom->name',
                'atom_name_b': 'metadata->system->surfactant->head_atom->name' },
            {
                'file_label': 'surfactant_head_surfactant_tail_rdf',
                'file_name': 'surfactant_head_surfactant_tail_rdf.txt',
                'type_label': 'surfactant_head_surfactant_tail_rdf',
                'atom_name_a': 'metadata->system->surfactant->head_atom->name',
                'atom_name_b': 'metadata->system->surfactant->tail_atom->name' },

            {
                'file_label': 'surfactant_tail_surfactant_tail_rdf',
                'file_name': 'surfactant_tail_surfactant_tail_rdf.txt',
                'type_label': 'surfactant_tail_surfactant_tail_rdf',
                'atom_name_a': 'metadata->system->surfactant->tail_atom->name',
                'atom_name_b': 'metadata->system->surfactant->tail_atom->name'},
        ]

class GromacsSolvatedTrajectoryAnalysis(
        GromacsVacuumTrajectoryAnalysis):
    """
    Implements partial analysis worklfow only.

    analysis dynamic infiles:
        no pull stub implemented

    - data_file:       default.gro
    - trajectory_file: default.xtc

    analysis fw_spec inputs:
    - metadata->system->counterion->reference_atom->name
    - metadata->system->solvent->reference_atom->name
    - metadata->system->substrate->reference_atom->name
    - metadata->system->surfactant->head_atom->name
    - metadata->system->surfactant->tail_atom->name

    anaylsis outfiles:
    - counterion_counterion_rdf: counterion_counterion_rdf.txt
        tagged as {'metadata->type': 'counterion_counterion_rdf'}
    - counterion_solvent_rdf: counterion_solvent_rdf.txt
        tagged as {'metadata->type': 'counterion_solvent_rdf'}
    - counterion_substrate_rdf: counterion_substrate_rdf.txt
        tagged as {'metadata->type': 'counterion_substrate_rdf'}
    - counterion_surfactant_head_rdf: counterion_surfactant_head_rdf.txt
        tagged as {'metadata->type': 'counterion_surfactant_head_rdf'}
    - counterion_surfactant_tail_rdf: counterion_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'counterion_surfactant_tail_rdf'}

    - solvent_solvent_rdf: solvent_solvent_rdf.txt
        tagged as {'metadata->type': 'solvent_solvent_rdf'}
    - solvent_substrate_rdf: solvent_substrate_rdf.txt
        tagged as {'metadata->type': 'solvent_substrate_rdf'}
    - solvent_surfactant_head_rdf: solvent_surfactant_head_rdf.txt
        tagged as {'metadata->type': 'solvent_surfactant_head_rdf'}
    - solvent_surfactant_tail_rdf: solvent_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'solvent_surfactant_tail_rdf'}

    - substrate_substrate_rdf: substrate_substrate_rdf.txt
        tagged as {'metadata->type': 'substrate_substrate_rdf'}
    - substrate_surfactant_head_rdf: substrate_surfactant_head_rdf.txt
        tagged as {'metadata->type': 'subtrate_surfactant_head_rdf'}
    - substrate_surfactant_tail_rdf: substrate_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'subtrate_surfactant_tail_rdf'}

    - surfactant_head_surfactant_head_rdf: surfactant_head_surfactant_head_rdf.txt
            tagged as {'metadata->type': 'surfactant_head_surfactant_head_rdf'}
    - surfactant_head_surfactant_tail_rdf: surfactant_head_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'surfactant_head_surfactant_tail_rdf'}

    - surfactant_tail_surfactant_tail_rdf: surfactant_tail_surfactant_tail_rdf.txt
        tagged as {'metadata->type': 'surfactant_tail_surfactant_tail_rdf'}
    """
    @property
    def rmsd_list(self):
        return [
            *super().rmsd_list,
            {
                'file_label': 'solvent_rmsd',
                'file_name': 'solvent_rmsd.txt',
                'type_label': 'solvent_rmsd',
                'atom_name': 'metadata->system->solvent->reference_atom->name',
            },
        ]

    # use regex
    #    - ([^\s:]+): ([^\s]+)\s+tagged as {'metadata->type': '([^\s']+)'}
    # and reqplacement pattern
    #    {'file_label': '$1', 'file_name': '$2', 'type_label': '$3', 'atom_name_a': a, 'atom_name_b': b},
    # on help text
    @property
    def rdf_list(self):
        return [
            *super().rdf_list,
            {
                'file_label': 'counterion_solvent_rdf',
                'file_name': 'counterion_solvent_rdf.txt',
                'type_label': 'counterion_solvent_rdf',
                'atom_name_a': 'metadata->system->counterion->reference_atom->name',
                'atom_name_b': 'metadata->system->solvent->reference_atom->name',
            }, {
                'file_label': 'solvent_solvent_rdf',
                'file_name': 'solvent_solvent_rdf.txt',
                'type_label': 'solvent_solvent_rdf',
                'atom_name_a': 'metadata->system->solvent->reference_atom->name',
                'atom_name_b': 'metadata->system->solvent->reference_atom->name',
            }, {
                'file_label': 'solvent_substrate_rdf',
                'file_name': 'solvent_substrate_rdf.txt',
                'type_label': 'solvent_substrate_rdf',
                'atom_name_a': 'metadata->system->solvent->reference_atom->name',
                'atom_name_b': 'metadata->system->substrate->reference_atom->name',
            }, {
                'file_label': 'solvent_surfactant_head_rdf',
                'file_name': 'solvent_surfactant_head_rdf.txt',
                'type_label': 'solvent_surfactant_head_rdf',
                'atom_name_a': 'metadata->system->solvent->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->head_atom->name'
            }, {
                'file_label': 'solvent_surfactant_tail_rdf',
                'file_name': 'solvent_surfactant_tail_rdf.txt',
                'type_label': 'solvent_surfactant_tail_rdf',
                'atom_name_a': 'metadata->system->solvent->reference_atom->name',
                'atom_name_b': 'metadata->system->surfactant->tail_atom->name'
            }
        ]


class GromacsDefaultTrajectoryAnalysis(GromacsMinimalTrajectoryAnalysis):
    pass
