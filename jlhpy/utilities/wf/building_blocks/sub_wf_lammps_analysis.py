# -*- coding: utf-8 -*-
"""Generic LAMMPS trajectory analyisis blocks."""

import datetime

from abc import ABC, abstractmethod

from fireworks import Firework
from fireworks.user_objects.firetasks.fileio_tasks import ArchiveDirTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import CmdTask

from jlhpy.utilities.wf.workflow_generator import WorkflowGenerator

from imteksimfw.utils.serialize import serialize_module_obj
class LAMMPSTrajectoryAnalysis(WorkflowGenerator):
    """
    General LAMMPS trajectory partial analysis worklfow.

    analysis dynamic infiles:
        no pull stub implemented

        - log_file:         log.lammps
        # - data_file:       default.lammps
        # - trajectory_file: default.nc

    analysis outfiles:
        - thermo_file:       thermo.out
    """

    def main(self, fws_root=[]):
        fw_list = []

        # extract thermo
        # --------------

        step_label = self.get_step_label('extract_thermo')

        files_in = {
            'log_file': 'log.lammps',
        }
        files_out = {
            'thermo_file': 'thermo.out',
        }

        fts_extract_thermo = [
            CmdTask(
                cmd="cat log.lammps | sed -n '/^Step/,/^Loop time/p' | sed '/^colvars:/d' | head -n-1 > thermo.out",
                fizzle_bad_rc=True,
                use_shell=True),
             ]

        fw_extract_thermo = self.build_fw(
            fts_extract_thermo, step_label,
            parents=fws_root,
            files_in=files_in,
            files_out=files_out,
            category=self.hpc_specs['fw_noqueue_category'])

        fw_list.append(fw_extract_thermo)

        return fw_list, [fw_extract_thermo], [fw_extract_thermo]


class LAMMPSSubstrateTrajectoryAnalysisWorkflowGenerator(WorkflowGenerator):
    """
    LAMMPS substrate trajectory partial analysis worklfow.

    analysis dynamic infiles:
        no pull stub implemented

        - data_file:       default.lammps
        - trajectory_file: default.nc

    analysis outfiles:
        - box_measures_file: box.txt
        - rdf_file:          rdf.txt
        - fcc_rdf_file:      fcc_rdf.txt

    """

    def main(self, fws_root=[]):
        fw_list = []

        # extract thermo
        # --------------

        step_label = self.get_step_label('extract_thermo')

        files_in = {
            'log_file': 'log.lammps',
        }
        files_out = {
            'thermo_file': 'thermo.out',
        }

        fts_extract_thermo = [
            CmdTask(
                cmd="cat log.lammps | sed -n '/^Step/,/^Loop time/p' | sed '/^colvars:/d' | head -n-1 > thermo.out",
                fizzle_bad_rc=True,
                use_shell=True),
             ]

        fw_extract_thermo = Firework(fts_extract_thermo,
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

        fw_list.append(fw_extract_thermo)

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

        return fw_list, [fw_analysis, fw_extract_thermo], [fw_analysis, fw_extract_thermo]
