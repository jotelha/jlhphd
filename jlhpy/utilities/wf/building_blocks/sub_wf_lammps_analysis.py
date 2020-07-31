# -*- coding: utf-8 -*-
"""Generic LAMMPS trajectory analyisis blocks."""

import datetime

from abc import ABC, abstractmethod

from fireworks import Firework
from fireworks.user_objects.firetasks.fileio_tasks import ArchiveDirTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator

from imteksimfw.fireworks.utilities.serialize import serialize_module_obj

class LAMMPSSubstrateTrajectoryAnalysisSubWorkflowGenerator(SubWorkflowGenerator):
    """
    LAMMPS substrate trajectory partial analysis worklfow.

    analysis dynamic infiles:
        no pull stub implemented

    - data_file:       default.lammps
    - trajectory_file: default.nc

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

        step_label = self.get_step_label('analysis')

        files_in = {
            'data_file': 'default.lammps',
            'trajectory_file': 'default.nc',
        }
        files_out = {
            'box_measures_file': 'box.txt',
            'rdf_file':          'rdf.txt',
            'fcc_rdf_file':      'fcc_rdf.txt',
        }

        # first task gets rdf and box measures, second task rdf only for fcc components
        fts_analysis = [
            CmdTask(
                cmd='lmp_extract_property',
                opt=['--verbose',
                     '--property', 'rdf', 'box',
                     '--trajectory', 'default.nc',
                     'default.lammps', 'rdf.txt', 'box.txt',
                    ],
                env='python',
                stdin_key='stdin',
                stderr_file='extract_std_property.err',
                stdout_file='extract_std_property.out',
                stdlog_file='extract_std_property.log',
                store_stdout=True,
                store_stderr=True,
                store_stdlog=True,
                fizzle_bad_rc=True),
            CmdTask(
                cmd='lmp_extract_property',
                opt=['--verbose',
                     '--modifier', 'fcc',
                     '--property', 'rdf',
                     '--trajectory', 'default.nc',
                     'default.lammps', 'fcc_rdf.txt',
                    ],
                env='python',
                stdin_key='stdin',
                stderr_file='extract_std_property.err',
                stdout_file='extract_std_property.out',
                stdlog_file='extract_std_property.log',
                store_stdout=True,
                store_stderr=True,
                store_stdlog=True,
                fizzle_bad_rc=True),
             ]

        fw_analysis = Firework(fts_analysis,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
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

        fw_list.append(fw_analysis)

        return fw_list, [fw_analysis], [fw_analysis]
