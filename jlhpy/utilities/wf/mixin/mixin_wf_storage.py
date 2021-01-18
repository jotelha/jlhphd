# -*- coding: utf-8 -*-
"""Storage mixins. All mixins are derived from PullMixin and PushMixin.

Each pull (or push) mixin must implement the method

    def pull(self, fws_root=[])

or

    def push(self, fws_root=[])

and those methods should call

    fw_list, fws_root_out, fws_leaf_out = super().pull(fws_root)

or

    fw_list, fws_root_out, fws_leaf_out = super().push(fws_root)

initialy to allow for arbitrary combinations of storage mixins.
Those three lists are to be extended accordingly eventually returned with

    return fw_list, fws_leaf_out, fws_root_out
"""
import datetime
import json
import logging

import pymongo

from fireworks import Firework
from fireworks.features.background_task import BackgroundTask
from fireworks.user_objects.firetasks.filepad_tasks import AddFilesTask
from imteksimfw.fireworks.user_objects.firetasks.dataflow_tasks import SearchDictTask
from imteksimfw.fireworks.user_objects.firetasks.dtool_tasks import (
    CreateDatasetTask, FreezeDatasetTask, CopyDatasetTask, FetchItemTask)
from imteksimfw.fireworks.user_objects.firetasks.dtool_lookup_tasks import (
    QueryDtoolTask, ReadmeDtoolTask, ManifestDtoolTask, DirectReadmeTask, DirectManifestTask)
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import EvalPyEnvTask
from imteksimfw.fireworks.user_objects.firetasks.ssh_tasks import SSHForwardTask
from imteksimfw.fireworks.user_objects.firetasks.storage_tasks import GetObjectFromFilepadTask


class PullMixin():
    """Abstract base class for querying in files."""

    def pull(self, fws_root=[]):
        return [], [], []


class PushMixin():
    """Abstract base class for storing out files."""

    def push(self, fws_root=[]):
        return [], [], []


class PullFromFilePadMixin(PullMixin):
    """Mixin for querying in files from file pad.

    Implementation shall provide 'source_project_id' and 'source_step'
    attributes. These may be provided file-wise by according per-file keys
    within the 'files_in_list' attribute."""

    def pull(self, fws_root=[]):
        fw_list, fws_root_out, fws_leaf_out = super().pull(fws_root)

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

        # TODO: this must happen in a more elegant way
        metadata_fp_source_key = self.kwargs.get('metadata_fp_source_key', 'metadata')
        metadata_fw_dest_key = self.kwargs.get('metadata_fw_dest_key', 'metadata')
        metadata_fw_source_key = self.kwargs.get('metadata_fw_source_key', 'metadata')

        fts_pull = []
        for file in self.files_in_list:
            fts_pull.append(
                GetObjectFromFilepadTask(
                    query={
                        **query,
                        'metadata->type': file['file_label'],
                    },
                    sort_key='metadata.datetime',
                    sort_direction=pymongo.DESCENDING,
                    new_file_name=file['file_name'],
                    metadata_fp_source_key=metadata_fp_source_key,
                    metadata_fw_dest_key=metadata_fw_dest_key,
                    metadata_fw_source_key=metadata_fw_source_key,
                    fw_supersedes_fp=True,
                    stdlog_file='std.log',
                    loglevel=logging.DEBUG,
                    propagate=True)
                )

        if len(fts_pull) > 0:
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
            fws_leaf_out.append(fw_pull)
            fws_root_out.append(fw_pull)

        return fw_list, fws_leaf_out, fws_root_out


class PushToFilePadMixin(PushMixin):
    """Mixing for storing out files in file pad."""

    def push(self, fws_root=[]):
        fw_list, fws_leaf_out, fws_root_out = super().push(fws_root)

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

        if len(fts_push) > 0:
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

class PullFromDtoolURIMixin(PullMixin):
    """Mixin for directly copying files from dataset identified by directly accessible URI."""

    def pull(self, fws_root=[]):
        fw_list, fws_root_out, fws_leaf_out = super().pull(fws_root)

        logger = logging.getLogger(__name__)

        # fetch items
        for file in self.files_in_list:
            file_label = file['file_label']
            if (not hasattr(self, 'files_in_info')) or (file_label not in self.files_in_info):
                logger.warning("No info on how to get infile '%s'" % file['file_label'])
                continue
            file_info = self.files_in_info[file_label]


            uri = file_info['uri']

            # readme

            step_label = self.get_step_label('pull_dtool_readme')

            files_in = {}
            files_out = {}

            # TODO: this must happen in a more elegant way
            metadata_dtool_source_key = file_info.get('metadata_dtool_source_key',
                                                      self.kwargs.get('metadata_dtool_source_key', None))
            metadata_fw_dest_key = file_info.get('metadata_fw_dest_key',
                                                 self.kwargs.get('metadata_fw_dest_key', 'metadata'))
            metadata_fw_source_key = file_info.get('metadata_fw_source_key',
                                                   self.kwargs.get('metadata_fw_source_key', 'metadata'))
            fts_readme = [
                DirectReadmeTask(
                    uri=uri,
                    output=metadata_fw_dest_key,
                    metadata_dtool_source_key=metadata_dtool_source_key,
                    metadata_fw_source_key=metadata_fw_source_key,
                    fw_supersedes_dtool=True,
                    loglevel=logging.DEBUG,
                    propagate=True,
                )
            ]

            fw_readme = self.build_fw(
                fts_readme, step_label,
                parents=fws_root,
                files_in=files_in,
                files_out=files_out,
                category=self.hpc_specs['fw_noqueue_category']
            )

            fw_list.append(fw_readme)
            fws_root_out.append(fw_readme)

            # manifest

            step_label = self.get_step_label('pull_dtool_manifest')

            files_in = {}
            files_out = {}

            fts_manifest = [
                DirectManifestTask(
                    uri=uri,
                    output='run->manifest_dtool_task',
                    propagate=False,
                )
            ]

            fw_manifest = self.build_fw(
                fts_manifest, step_label,
                parents=[fw_readme],
                files_in=files_in,
                files_out=files_out,
                category=self.hpc_specs['fw_noqueue_category']
            )

            fw_list.append(fw_manifest)

            # search item_id by filename

            files_in = {}
            files_out = {}

            step_label = self.get_step_label('pull_dtool_search_item')

            file_name = file_info.get('file_name', file['file_name'])

            fts_search_item = [
                SearchDictTask(
                    input_key='run->manifest_dtool_task->items',
                    search={'relpath': file_name},
                    marker={'relpath': True},
                    output_key='run->search_dict_task',
                    limit=1,
                    expand=True,
                    loglevel=logging.DEBUG,
                    propagate=False,
                )
            ]

            fw_search_item = self.build_fw(
                fts_search_item, step_label,
                parents=[fw_manifest],
                files_in=files_in,
                files_out=files_out,
                category=self.hpc_specs['fw_noqueue_category']
            )

            fw_list.append(fw_search_item)

            # fetch item by item_id

            step_label = self.get_step_label('pull_dtool_fetch_item')

            files_in = {}
            files_out = {file['file_label']: file['file_name']}

            fts_fetch_item = [
                FetchItemTask(
                    item_id={'key': 'run->search_dict_task'},
                    source=uri,
                    filename=file['file_name']
                )
            ]

            fw_fetch_item = self.build_fw(
                fts_fetch_item, step_label,
                parents=[fw_search_item],
                files_in=files_in,
                files_out=files_out,
                category=self.hpc_specs['fw_noqueue_category']
            )

            fws_leaf_out.append(fw_fetch_item)
            fw_list.append(fw_fetch_item)

        return fw_list, fws_leaf_out, fws_root_out


class PullFromDtoolRepositoryMixin(PullMixin):
    """Mixin for querying files from dtool lookup server and copying subsequently."""

    def pull(self, fws_root=[]):
        fw_list, fws_root_out, fws_leaf_out = super().pull(fws_root)

        logger = logging.getLogger(__name__)
        # if len(self.files_in_list) == 0:
        #    return fw_list, fws_root_out, fws_leaf_out

        # fetch items
        for file in self.files_in_list:
            file_label = file['file_label']
            if (not hasattr(self, 'files_in_info')) or (file_label not in self.files_in_info):
                logger.warning("No info on how to get infile '%s'" % file['file_label'])
                continue
            file_info = self.files_in_info[file_label]


            # query
            step_label = self.get_step_label('pull_dtool_query')

            files_in = {}
            files_out = {}

            query = file_info['query']

            fts_query = [
                QueryDtoolTask(
                    query=json.dumps(query),
                    sort_key='frozen_at',
                    sort_direction=pymongo.DESCENDING,
                    limit=1,
                    expand=True,
                    output='run->query_dtool_task',
                    loglevel=logging.DEBUG,
                    propagate=True),
            ]

            fw_query = self.build_fw(
                fts_query, step_label,
                parents=fws_root,
                files_in=files_in,
                files_out=files_out,
                category=self.hpc_specs['fw_noqueue_category']
            )

            fw_list.append(fw_query)
            fws_root_out.append(fw_query)

            # readme

            step_label = self.get_step_label('pull_dtool_readme')

            files_in = {}
            files_out = {}

            # TODO: this must happen in a more elegant way
            metadata_dtool_source_key = file_info.get('metadata_dtool_source_key',
                                                      self.kwargs.get('metadata_dtool_source_key', None))
            metadata_fw_dest_key = file_info.get('metadata_fw_dest_key',
                                                 self.kwargs.get('metadata_fw_dest_key', 'metadata'))
            metadata_fw_source_key = file_info.get('metadata_fw_source_key',
                                                   self.kwargs.get('metadata_fw_source_key', 'metadata'))
            fts_readme = [
                ReadmeDtoolTask(
                    uri={'key': 'run->query_dtool_task->uri'},
                    output=metadata_fw_dest_key,
                    metadata_dtool_source_key=metadata_dtool_source_key,
                    metadata_fw_source_key=metadata_fw_source_key,
                    fw_supersedes_dtool=True,
                    loglevel=logging.DEBUG,
                    propagate=True,
                )
            ]

            fw_readme = self.build_fw(
                fts_readme, step_label,
                parents=[fw_query],
                files_in=files_in,
                files_out=files_out,
                category=self.hpc_specs['fw_noqueue_category']
            )

            fw_list.append(fw_readme)

            # manifest

            step_label = self.get_step_label('pull_dtool_manifest')

            files_in = {}
            files_out = {}

            fts_manifest = [
                ManifestDtoolTask(
                    uri={'key': 'run->query_dtool_task->uri'},
                    output='run->manifest_dtool_task',
                    propagate=False,
                )
            ]

            fw_manifest = self.build_fw(
                fts_manifest, step_label,
                parents=[fw_readme],
                files_in=files_in,
                files_out=files_out,
                category=self.hpc_specs['fw_noqueue_category']
            )

            fw_list.append(fw_manifest)

            # search item_id by filename

            files_in = {}
            files_out = {}

            step_label = self.get_step_label('pull_dtool_search_item')

            file_name = file_info.get('file_name', file['file_name'])

            fts_search_item = [
                SearchDictTask(
                    input_key='run->manifest_dtool_task->items',
                    search={'relpath': file_name},
                    marker={'relpath': True},
                    output_key='run->search_dict_task',
                    limit=1,
                    expand=True,
                    loglevel=logging.DEBUG,
                    propagate=False,
                )
            ]

            fw_search_item = self.build_fw(
                fts_search_item, step_label,
                parents=[fw_manifest],
                files_in=files_in,
                files_out=files_out,
                category=self.hpc_specs['fw_noqueue_category']
            )

            fw_list.append(fw_search_item)

            # fetch item by item_id

            step_label = self.get_step_label('pull_dtool_fetch_item')

            files_in = {}
            files_out = {file['file_label']: file['file_name']}

            fts_fetch_item = [
                FetchItemTask(
                    item_id={'key': 'run->search_dict_task'},
                    source={'key': 'run->query_dtool_task->uri'},
                    filename=file['file_name']
                )
            ]

            fw_fetch_item = self.build_fw(
                fts_fetch_item, step_label,
                parents=[fw_search_item],
                files_in=files_in,
                files_out=files_out,
                category=self.hpc_specs['fw_noqueue_category']
            )

            fws_leaf_out.append(fw_fetch_item)
            fw_list.append(fw_fetch_item)

        return fw_list, fws_leaf_out, fws_root_out


# TODO: add per item annotation for file label
class PushToDtoolRepositoryMixin(PushMixin):
    """Mixing for storing out files in dtool dataset."""

    def push(self, fws_root=[]):
        fw_list, fws_leaf_out, fws_root_out = super().push(fws_root)

        step_label_suffix = 'push_dtool'
        step_label = self.get_step_label(step_label_suffix)

        files_out = {}
        files_in = {
            f['file_label']: f['file_name'] for f in self.files_out_list
        }

        # make step label a valid 80 char dataset name
        dataset_name = self.get_80_char_slug()

        if len(files_in) > 0:
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


class PushDerivedDatasetToDtoolRepositoryMixin(PushMixin):
    """Mixing for storing derive out files in dtool dataset.

    If using this mixin, then make sure the workflow starts with

        'metadata->step_specific->dtool_push->remote_dataset'

    set (to 'None' if there is no previous dataset)."""

    def push(self, fws_root=[]):
        fw_list, fws_leaf_out, fws_root_out = super().push(fws_root)

        step_label_suffix = 'push_dtool'
        step_label = self.get_step_label(step_label_suffix)

        files_out = {}
        files_in = {
            f['file_label']: f['file_name'] for f in self.files_out_list
        }

        # make step label a valid 80 char dataset name
        dataset_name = self.get_80_char_slug()

        if len(files_in) > 0:
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


class PushToDtoolRepositoryViaSSHJumpHostMixin(PushMixin):
    """Mixing for storing out files in dtool dataset."""

    def push(self, fws_root=[]):
        fw_list, fws_leaf_out, fws_root_out = super().push(fws_root)

        step_label_suffix = 'push_dtool'
        step_label = self.get_step_label(step_label_suffix)

        files_out = {}
        files_in = {
            f['file_label']: f['file_name'] for f in self.files_out_list
        }

        if len(files_in) > 0:
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
        fw_list, fws_leaf_out, fws_root_out = super().push(fws_root)

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

        if len(fts_push) > 0:
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
            # second difference to standard FilepadMixin: FilePad push always as stub
            # fws_leaf_out.append(fw_push)
            fws_root_out.append(fw_push)

        return fw_list, fws_leaf_out, fws_root_out


class PushToDtoolRepositoryViaSSHJumpHostAndFilePadMixin(
        PushToDtoolRepositoryViaSSHJumpHostMixin, PushToFilePadMixin):
    pass


class DefaultPullMixin(PullFromDtoolURIMixin):
    pass

class DefaultPushMixin(PushDerivedDatasetToDtoolRepositoryMixin):
    pass
