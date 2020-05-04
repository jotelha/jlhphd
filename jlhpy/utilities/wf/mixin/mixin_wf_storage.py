# -*- coding: utf-8 -*-
"""Indenter bounding sphere sub workflow."""

import datetime

from abc import ABC, abstractmethod

from fireworks import Firework
from fireworks.features.background_task import BackgroundTask
from fireworks.user_objects.firetasks.fileio_tasks import ArchiveDirTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
# from fireworks.user_objects.firetasks.script_task import PyTask
from imteksimfw.fireworks.user_objects.firetasks.dtool_tasks import (
    CreateDatasetTask, FreezeDatasetTask, CopyDatasetTask)
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import (
    PyEnvTask, EvalPyEnvTask)

from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator


class PushMixinABC(SubWorkflowGenerator, ABC):
    """Abstract base class for storing out files.

    Implementation must provide files_out_list."""

    @property
    @abstractmethod
    def files_out_list(self) -> list:
        """list of dict with fields file_label, file_name, type_label..."""
        ...

    @abstractmethod
    def push(self, fws_root=[]):
        ...


class PushToFilePadMixin(PushMixinABC, SubWorkflowGenerator):
    """Abstract base class for storing out files in file pad."""

    def push(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('push filepad')

        files_out = {}
        files_in = {
            f['file_label']: f['file_name'] for f in self.files_out_list
        }

        fts_push = []
        for file in self.files_out_list:
            fts_push.append(
                AddFilesTask({
                    'compress': True,
                    'paths': file['file_name'],
                    'metadata': {
                        'project': self.project_id,
                        'datetime': str(datetime.datetime.now()),
                        'type':    file['type_label']}
                })
            )

        fw_push = Firework(fts_push,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_push)

        return fw_list, [fw_push], [fw_push]


class PushToDtoolRepositoryMixin(PushMixinABC, SubWorkflowGenerator):
    """Abstract base class for storing out files in dtool dataset."""

    def push(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('push dtool')

        files_out = {}
        files_in = {
            f['file_label']: f['file_name'] for f in self.files_out_list
        }

        # background connectivity task
        ft_ssh = PyEnvTask(
            func='imteksimfw.fireworks.utilities.ssh_forward.forward',
            kwargs={
                'port_file': '.port',
            },
            kwargs_inputs={
                'remote_host':  'metadata->step_specific->dtool_push->ssh_config->remote_host',
                'remote_port':  'metadata->step_specific->dtool_push->ssh_config->remote_port',
                'ssh_host':     'metadata->step_specific->dtool_push->ssh_config->ssh_host',
                'ssh_user':     'metadata->step_specific->dtool_push->ssh_config->ssh_user',
                'ssh_keyfile':  'metadata->step_specific->dtool_push->ssh_config->ssh_keyfile',
            }
        )

        bg_fts_push = BackgroundTask(ft_ssh,
            num_launches=1, run_on_finish=False, sleep_time=0)

        fts_push = [
            CreateDatasetTask(
                name='dataset',  # TODO: assign proper name
                metadata={'project': self.project_id},
                metadata_key='metadata',
                output='metadata->step_specific->dtool_push->local_proto_dataset',
                propagate=True,
            ),
            FreezeDatasetTask(
                uri={'key': 'metadata->step_specific->dtool_push->local_proto_dataset->uri'},
                output='metadata->step_specific->dtool_push->local_frozen_dataset',
                propagate=True,
            ),
            # hopefully enough time for the bg ssh tunnel to be established
            # otherwise might need:
            # ScriptTask
            #   script: >-
            #     counter=0;
            #     while [ ! -f .port ]; do
            #       sleep 1;
            #       counter=$((counter + 1));
            #       if [ $counter -ge 10 ]; then
            #         echo "Timed out waiting for port!";
            #         exit 126;
            #       fi;
            #     done
            #   stderr_file:   wait.err
            #   stdout_file:   wait.out
            #   store_stdout:  true
            #   store_stderr:  true
            #   fizzle_bad_rc: true
            #   use_shell:     true
            #
            # read port from socket:
            EvalPyEnvTask(
                func="lambda: int(open('.port','r').readlines()[0])",
                outputs=[
                    'metadata->step_specific->dtool_push->dtool_config->{}'
                    .format(self.kwargs['dtool_port_config_key']),
                ],
            ),
            CopyDatasetTask(
                source={'key': 'metadata->step_specific->dtool_push->local_frozen_dataset->uri'},
                target={'key': 'metadata->step_specific->dtool_push->dtool_target'},
                dtool_config_key={'key': 'metadata->step_specific->dtool_push->dtool_config'},
                output='metadata->step_specific->dtool_push->remote_dataset',
                propagate=True,
            )
        ]

        fw_push = Firework(fts_push,
            name=self.get_fw_label(step_label),
            spec={
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in': files_in,
                '_files_out': files_out,
                '_background_tasks': bg_fts_push,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_push)

        return fw_list, [fw_push], [fw_push]
