# -*- coding: utf-8 -*-
import abc
import datetime
import re
import unicodedata

import pymongo

from fireworks import Firework
from fireworks.features.background_task import BackgroundTask
# from fireworks.user_objects.firetasks.fileio_tasks import ArchiveDirTask
from fireworks.user_objects.firetasks.filepad_tasks import (
    GetFilesByQueryTask, AddFilesTask)
from imteksimfw.fireworks.user_objects.firetasks.dtool_tasks import (
    CreateDatasetTask, FreezeDatasetTask, CopyDatasetTask)
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import EvalPyEnvTask
from imteksimfw.fireworks.user_objects.firetasks.ssh_tasks import SSHForwardTask


class PullMixinABC(abc.ABC):
    """Abstract base class for querying in files."""

    def pull(self, fws_root=[]):
        return [], [], []


class PushMixinABC(abc.ABC):
    """Abstract base class for storing out files."""

    def push(self, fws_root=[]):
        return [], [], []


class PullFromFilePadMixin(PushMixinABC):
    """Mixin for querying in files from file pad.

    Implementation shall provide 'source_project_id' and 'source_step'
    attributes. These may be provided file-wise by according per-file keys
    within the 'files_in_list' attribute."""

    def pull(self, fws_root=[]):
        fw_list = []

        step_label = self.get_step_label('pull_filepad')

        files_in = {}
        files_out = {
            f['file_label']: f['file_name'] for f in self.files_in_list}

        # build default query
        query = {}
        if hasattr(self, 'source_step'):
            query['metadata->step'] = self.source_step

        if hasattr(self, 'source_project_id'):
            query['metadata->project'] = self.source_project_id

        fts_pull = []
        for file in self.files_in_list:
            fts_pull.append(
                GetFilesByQueryTask(
                    query={
                        **query,
                        'metadata->type': file['file_label'],
                    },
                    sort_key='metadata.datetime',
                    sort_direction=pymongo.DESCENDING,
                    limit=1,
                    new_file_names=[file['file_name']])
                )

        fw_pull = Firework(fts_pull,
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

        fw_list.append(fw_pull)

        return fw_list, [fw_pull], [fw_pull]



class PushToFilePadMixin(PushMixinABC):
    """Abstract base class for storing out files in file pad."""

    def push(self, fws_root=[]):
        fw_list, fws_root_out, fws_leaf_out = super().push(fws_root)

        step_label = self.get_step_label('push_filepad')

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
                        'type':    file['file_label']},
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
        fws_leaf_out.append(fw_push)
        fws_root_out.append(fw_push)

        return fw_list, fws_leaf_out, fws_root_out


# TODO: add per item annotation for file label
class PushToDtoolRepositoryMixin(PushMixinABC):
    """Abstract base class for storing out files in dtool dataset."""

    def push(self, fws_root=[]):
        fw_list, fws_root_out, fws_leaf_out = super().push(fws_root)

        step_label_suffix = 'push_dtool'
        step_label = self.get_step_label(step_label_suffix)

        files_out = {}
        files_in = {
            f['file_label']: f['file_name'] for f in self.files_out_list
        }

        # make step label a valid 80 char dataset name
        dataset_name = self.get_80_char_slug()

        fts_push = [
            CreateDatasetTask(
                name=dataset_name,
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
            CopyDatasetTask(
                source={'key': 'metadata->step_specific->dtool_push->local_frozen_dataset->uri'},
                target={'key': 'metadata->step_specific->dtool_push->dtool_target'},
                dtool_config_key='metadata->step_specific->dtool_push->dtool_config',
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
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_push)
        fws_leaf_out.append(fw_push)
        fws_root_out.append(fw_push)

        return fw_list, fws_leaf_out, fws_root_out


class PushDerivedDatasetToDtoolRepositoryMixin(PushMixinABC):
    """Abstract base class for storing derive out files in dtool dataset.

    If using this mixin, then make sure the workflow starts with

        'metadata->step_specific->dtool_push->remote_dataset'

    set (to 'None' if there is no previous dataset)."""

    def push(self, fws_root=[]):
        fw_list, fws_root_out, fws_leaf_out = super().push(fws_root)

        step_label_suffix = 'push_dtool'
        step_label = self.get_step_label(step_label_suffix)

        files_out = {}
        files_in = {
            f['file_label']: f['file_name'] for f in self.files_out_list
        }

        # make step label a valid 80 char dataset name
        dataset_name = self.get_80_char_slug()

        fts_push = [
            CreateDatasetTask(
                name=dataset_name,
                metadata={'project': self.project_id},
                metadata_key='metadata',
                output='metadata->step_specific->dtool_push->local_proto_dataset',
                source_dataset={'key': 'metadata->step_specific->dtool_push->remote_dataset'},  # distinguishes this class from above's PushToDtoolRepositoryMixin
                propagate=True,
            ),
            FreezeDatasetTask(
                uri={'key': 'metadata->step_specific->dtool_push->local_proto_dataset->uri'},
                output='metadata->step_specific->dtool_push->local_frozen_dataset',
                propagate=True,
            ),
            CopyDatasetTask(
                source={'key': 'metadata->step_specific->dtool_push->local_frozen_dataset->uri'},
                target={'key': 'metadata->step_specific->dtool_push->dtool_target'},
                dtool_config_key='metadata->step_specific->dtool_push->dtool_config',
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
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step_label,
                    **self.kwargs
                }
            },
            parents=fws_root)

        fw_list.append(fw_push)
        fws_leaf_out.append(fw_push)
        fws_root_out.append(fw_push)

        return fw_list, fws_leaf_out, fws_root_out


class PushToDtoolRepositoryViaSSHJumpHostMixin(PushMixinABC):
    """Abstract base class for storing out files in dtool dataset."""

    def push(self, fws_root=[]):
        fw_list, fws_root_out, fws_leaf_out = super().push(fws_root)

        step_label_suffix = 'push_dtool'
        step_label = self.get_step_label(step_label_suffix)

        files_out = {}
        files_in = {
            f['file_label']: f['file_name'] for f in self.files_out_list
        }

        # background connectivity task
        ft_ssh = SSHForwardTask(
            port_file='.port',
            remote_host={'key': 'metadata->step_specific->dtool_push->ssh_config->remote_host'},
            remote_port={'key': 'metadata->step_specific->dtool_push->ssh_config->remote_port'},
            ssh_host={'key': 'metadata->step_specific->dtool_push->ssh_config->ssh_host'},
            ssh_user={'key': 'metadata->step_specific->dtool_push->ssh_config->ssh_user'},
            ssh_keyfile={'key': 'metadata->step_specific->dtool_push->ssh_config->ssh_keyfile'},
            #stdlog_file='bg_task.log',
        )

        bg_fts_push = [BackgroundTask(ft_ssh,
            num_launches=1, run_on_finish=False, sleep_time=0)]

        dataset_name = self.get_80_char_slug()

        fts_push = [
            CreateDatasetTask(
                name=dataset_name,
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
                stderr_file='read_port.err',
                stdout_file='read_port.out',
                stdlog_file='read_port.log',
                py_hist_file='read_port.py',
                call_log_file='read_port.calls_trace',
                vars_log_file='read_port.vars_trace',
                store_stdout=True,
                store_stderr=True,
                store_stdlog=True,
                fizzle_bad_rc=True,
            ),

            CopyDatasetTask(
                source={'key': 'metadata->step_specific->dtool_push->local_frozen_dataset->uri'},
                target={'key': 'metadata->step_specific->dtool_push->dtool_target'},
                dtool_config_key='metadata->step_specific->dtool_push->dtool_config',
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
        fws_leaf_out.append(fw_push)
        fws_root_out.append(fw_push)

        return fw_list, fws_leaf_out, fws_root_out


class PushToDtoolRepositoryAndFilePadMixin(
        PushToDtoolRepositoryMixin, PushToFilePadMixin):
    pass


# class PushDerivedDatasetToDtoolRepositoryAndFilePadMixin(
#         PushDerivedDatasetToDtoolRepositoryMixin, PushToFilePadMixin):
#     pass


class PushDerivedDatasetToDtoolRepositoryAndFilePadMixin(PushDerivedDatasetToDtoolRepositoryMixin):
    """Implements a FilePadMixin dependent on DtoolMixin"""
    def push(self, fws_root=[]):
        fw_list, fws_root_out, fws_leaf_out = super().push(fws_root)

        step_label = self.get_step_label('push_filepad')

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
                        'type':    file['file_label']},
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
            # first difference to standatrd FilepadMixin: dependency on all super push Fireworks
            parents=[*fws_root, *fws_leaf_out])

        fw_list.append(fw_push)
        # second difference to standatrd FilepadMixin: FilePad push always as stub
        # fws_leaf_out.append(fw_push)
        fws_root_out.append(fw_push)

        return fw_list, fws_leaf_out, fws_root_out


class PushToDtoolRepositoryViaSSHJumpHostAndFilePadMixin(
        PushToDtoolRepositoryViaSSHJumpHostMixin, PushToFilePadMixin):
    pass


class DefaultPullMixin(PullFromFilePadMixin):
    pass

class DefaultPushMixin(PushDerivedDatasetToDtoolRepositoryAndFilePadMixin):
    pass
