# TODO:
# 2020/07/02 remove all project_id, only project

# system basics
from abc import abstractmethod

import copy
import datetime
import logging
import itertools
import os
import pickle

from collections.abc import Iterable

# fireworks
import pymongo
from fireworks import Firework, Workflow

from fireworks.utilities.dict_mods import set_nested_dict_value

from jlhpy.utilities.wf.utils import slugify

# suggested in_files, out_file labels
FILE_LABELS = {
    'data_file': 'general',
    'input_file': 'general',
    'png_file': 'image',

    'topology_file':  'gromacs.top',
    'restraint_file': 'gromacs.posre.itp'
}

from jlhpy.utilities.wf.hpc_config import HPC_SPECS
from jlhpy.utilities.wf.utils import get_nested_dict_value

DEFAULT_LIFESPAN = 2*365  # default lifespan of datasets in days

# TODO: remove mess around project and project_id
class FireWorksWorkflowGenerator:
    def __init__(
            self,
            project_id=None,
            hpc_specs=None,
            infile_prefix=None,
            integrate_push=False,
            fw_name_template=None,
            fw_name_prefix=None,
            fw_name_suffix=None,
            wf_name_prefix=None,
            parameter_key_prefix='metadata',
            **kwargs
        ):
        """All **kwargs are treated as metadata.

        While the explicitly defined arguments above will not enter
        workflow- and FireWorks-attached metadata directly, everything
        within **kwargs will.

        args
        ----
        - parameter_key_prefix: prepend to parametric keys in queries, default: 'metadata'
        - project_id: unique name
        - machine: name of machine to run at
        - hpc_specs: if not specified, then look up by machine name
        - fw_name_template: template for building FireWorks name
        - fw_name_template:
        - fw_name_prefix:
        - fw_name_suffix:
        - wf_name_prefix:
        - wf_name_suffix:

        kwargs
        ------
        - parameter_keys: keys to treat as parametric
        - parameter_label_key_dict: same as above, but allows for labelling
                parametric keys for display pruposes. Overrides parameter_keys.
            wf_name: Workflow name, by default concatenated 'wf_name_prefix',
                'machine' and 'project_id'.
            wf_name_prefix:
            source_project_id: when querying files, use another project id
            source_step: when querying files, use this particular step label
            infile_prefix: when inserting files into db manually, use this prefix
        """
        # TODO: standardize and sort kwargs with special meaning
        self.project_id = project_id
        self.integrate_push = integrate_push

        self.kwargs = kwargs
        self.kwargs['project_id'] = project_id

        if 'machine' in self.kwargs:
            self.machine = self.kwargs['machine']
        else:
            self.machine = 'ubuntu'  # dummy

        if hpc_specs:
            self.hpc_specs = hpc_specs
        else:
            self.hpc_specs = HPC_SPECS[self.machine]

        self.wf_name_prefix = ':'.join(reversed([o.__name__ for i, o in enumerate(self.__class__.mro()) if getattr(o, 'opaque', False) or i == 0]))
        if wf_name_prefix:
            self.wf_name_prefix = ':'.join((wf_name_prefix, self.wf_name_prefix))

        if 'wf_name' in self.kwargs:
            self.wf_name = self.kwargs['wf_name']
        else:
            self.wf_name = '{prefix:}, {machine:}, {id:}'.format(
                prefix=self.wf_name_prefix,
                machine=self.machine,
                id=self.project_id)

        # define label and key such as:
        # "parameter_label_key_dict": {
        #     "n": "system->surfactant->nmolecules"
        # },
        if 'parameter_label_key_dict' in self.kwargs:
            self.parameter_label_key_dict = self.kwargs['parameter_label_key_dict']
            self.parameter_keys = list(self.parameter_label_key_dict.values())
        # or only keys such as:
        # "parameter_keys": [ "system->surfactant->nmolecules" ]
        elif 'parameter_keys' in self.kwargs:
            self.parameter_keys = self.kwargs['parameter_keys']
            if isinstance(self.parameter_keys, str):
                self.parameter_keys = [self.parameter_keys]
            self.parameter_label_key_dict = {k: k for k in self.parameter_keys}
        else:
            self.parameter_label_key_dict = {}
            self.parameter_keys = []

        assert isinstance(self.parameter_label_key_dict, dict)
        assert isinstance(self.parameter_keys, list)

        self.parameter_dict = {
            '->'.join((parameter_key_prefix, k)): get_nested_dict_value(
                self.kwargs, k) for k in self.parameter_keys}

        if fw_name_prefix:
            self.fw_name_prefix = fw_name_prefix
        elif len(self.parameter_dict) > 0:
            self.fw_name_prefix = ', '.join(([
                '{}={}'.format(
                    label,
                    self.parameter_dict['->'.join((parameter_key_prefix, key))]
                ) for label, key in self.parameter_label_key_dict.items()]
            ))
        else:
            self.fw_name_prefix = ''

        if infile_prefix:
            self.infile_prefix = infile_prefix

        # TODO needs extension to multiple sources
        if 'source_project_id' in self.kwargs:
            self.source_project_id = self.kwargs['source_project_id']
        else:
            self.source_project_id = self.project_id

        if 'source_step' in self.kwargs:
            self.source_step = self.kwargs['source_step']

        creation_date = datetime.datetime.now()
        if 'creation_date' not in self.kwargs:
            self.kwargs['creation_date'] = str(creation_date)

        if 'expiration_date' not in self.kwargs:
            self.kwargs['expiration_date'] = str(
                creation_date + datetime.timedelta(days=DEFAULT_LIFESPAN))


class WorkflowGenerator(FireWorksWorkflowGenerator):
    """A sub-workflow generator should implement three methods:
    pull, main and push. Each method returns three lists of FireWorks,
    - fws_list: all (readily interconnected) FireWorks of a sub-workflow
    - fws_leaf: all leaves of a sub-workflow
    - fws_root: all roots of a sub-workflow

    fws_list must always give rise to an interconnected sub-workflow.
    The sub-workflow returned by
    - pull: queries necessary input data
    - push: stores results
    - main: performs actual computations

    A sub-workflow's interface is defined via
    - the combined inputs expected by all FirewWorks withins its fws_root,
    - the combined outputs produced by all FirewWorks withins its fws_leaf,
    - arbitrary, documented fw_spec

    pull sub-wf is a terminating stub and does not expect any inputs.
    push sub-wf is a terminating stup and does not yield any outputs.
    main sub-wf expects intputs and produces outputs.

    Inputs and outputs can be files or specs. They are to be documented
    in the following manner (use as template):

    ### sample template ###

    dynamic infiles:
        only queried in pull stub, otherwise expected through data flow

    - data_file: default.gro
    - input_file: default.mdp

    static infiles:
        always queried within main trunk

    - template_file: sys.top.template,
        queried by {'metadata->name': file_config.GMX_PULL_TOP_TEMPLATE}
    - parameter_file: pull.mdp.template,
        queried by {'metadata->name': file_config.GMX_PULL_MDP_TEMPLATE}

    fw_spec inputs:
        key-value pairs referred to within sub-workflow
    - metadata->system->surfactant->nmolecules
    - metadata->system->surfactant->name

    outfiles:
        use regex replacement /'([^']*)':(\\s*)'([^']*)',/- \1:\2\3/
        to format from files_out dict

    - topology_file:  default.top
        tagged as {metadata->type: top_pull}
    - index_file:     out.ndx
        tagged as {metadata->type: ndx_pull}
    - input_file:     out.mdp
        tagged as {metadata->type: mdp_pull}

    fw_spec outputs:
        key - value pairs modified within sub-workflow

    - metadata->system->surfactant->length

    ### end of sample template ###

    WorkflowGenerators should list dynamic infiles and outfiles in its
    attributes 'files_out_list' and 'files_in_list' of the format

    [ {'file_label' : 'label1', 'file_name': 'name1'}, {...}, {...} ]

    to enable mixins and composite classes to easyly operate on those.
    If not provided explicitly, they are generated by looking up infiles
    and outfiles of the main trunk's roots and leaves respectively.

    The three-part workflow arising from connected pull, main and push
    sub-workflows should ideally be runnable independently.

                        + - - - - -+
                        ' main     '
                        '          '
                        ' +------+ '
                        ' | pull | '
                        ' +------+ '
                        '   |      '
                        '   |      '
                        '   v      '
     fws_root inputs    ' +------+ '
    ------------------> ' |      | '
                        ' | main | '
     fws_leaf outputs   ' |      | '
    <------------------ ' |      | '
                        ' +------+ '
                        '   |      '
                        '   |      '
                        '   v      '
                        ' +------+ '
                        ' | push | '
                        ' +------+ '
                        '          '
                        + - - - - -+

    If 'integrate_push' is set, then the layout is modified to

                        + - - - - -+
                        ' main     '
                        '          '
                        ' +------+ '
                        ' | pull | '
                        ' +------+ '
                        '   |      '
                        '   |      '
                        '   v      '
     fws_root inputs    ' +------+ '
    ------------------> ' |      | '
                        ' | main | '
     fws_leaf outputs   ' |      | '
    <------------------ ' |  &   | '
                        ' |      | '
                        ' | push | '
                        ' +------+ '
                        '          '
                        + - - - - -+

    """
    def get_step_label(self, suffix):
        return ':'.join((self.wf_name_prefix, suffix))

    # TODO: only suffix instead of step_label
    def get_fw_label(self, step_label):
        # return self.fw_name_template.format(
        #     fw_name_prefix=step_label, fw_name_suffix=self.fw_name_suffix)
        if len(self.fw_name_prefix) > 0:
            return ', '.join((self.fw_name_prefix, step_label))
        else:
            return step_label

    def get_80_char_slug(self, suffix=''):
        # timestamp - parameters - sub-workflow hierarchy - step

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')

        label = ' '.join((
            timestamp, self.fw_name_prefix,
            self.wf_name_prefix, suffix))
        slug = slugify(label)

        if len(slug) > 80:  # ellipsis
            label = ' '.join((
                timestamp,
                self.fw_name_prefix,
                self.wf_name_prefix.split(':')[0],
                suffix))
            slug = slugify(label)

        if len(slug) > 80:  # ellipsis
            slug = slug[:39].rstrip('-') + '--' + slug[-39:].lstrip('-')

        return slug

    def get_wf_label(self):
        return self.wf_name

    def build_fw(self, fts, step_label,
                 parents=[], category=None, files_in={}, files_out={},
                 queueadapter=None, fw_spec=None):
        if category is None:
            category = self.hpc_specs['fw_noqueue_category']
        if queueadapter is None:
            queueadapter = {}
        else:
            queueadapter = {
                '_queueadapter': {
                    **queueadapter
                }
            }
        if fw_spec is None:
            fw_spec = {}

        return Firework(fts,
                        name=self.get_fw_label(step_label),
                        spec={
                            '_category': category,
                            '_files_in':  files_in,
                            '_files_out': files_out,
                            **queueadapter,
                            **fw_spec,
                            'metadata': {
                                'project':  self.project_id,
                                'datetime': str(datetime.datetime.now()),
                                'step':     step_label,
                                **self.kwargs
                            }
                        },
                        parents=parents)

    def pull(self, fws_root=[]):
        """Generate FireWorks for querying input files."""
        return [], [], []

    def main(self, fws_root=[]):
        """Generate sub-workflow main part."""
        return [], [], []

    def push(self, fws_root=[]):
        """Generate FireWorks for storing output files."""
        return [], [], []

    def get_as_independent(self, fws_root=[]):
        """Return a self-sufficient FireWorks list with pull and push stub."""

        fws_pull, fws_pull_leaf, _ = self.pull()
        fws_main, fws_main_leaf, fws_main_root = self.main(
            [*fws_pull_leaf, *fws_root])
        fws_push, fws_push_leaf, _ = self.push(fws_main_leaf)

        fws_list = [*fws_pull, *fws_main, *fws_push]
        fws_root = [*fws_main_root]
        fws_leaf = [*fws_main_leaf]

        if self.integrate_push:
            fws_leaf.extend(fws_push_leaf)

        return fws_list, fws_leaf, fws_root

    def get_as_root(self, fws_root=[]):
        """Return as root FireWorks list with pull stub, but no push stub."""

        # main processing branch
        fws_pull, fws_pull_leaf, _ = self.pull()
        fws_main, fws_main_leaf, fws_main_root = self.main(
            [*fws_pull_leaf, *fws_root])

        fws_list = [
            *fws_pull, *fws_main,
        ]

        return fws_list, fws_main_leaf, fws_main_root

    def get_as_leaf(self, fws_root=[]):
        """Return as leaf FireWorks list without pull stub, but with push stub."""

        fws_main, fws_main_leaf, fws_main_root = self.main(fws_root)
        fws_push, fws_push_leaf, _ = self.push(fws_main_leaf)

        fws_list = [*fws_main, *fws_push]
        fws_root = [*fws_main_root]
        fws_leaf = [*fws_main_leaf]

        if self.integrate_push:
            fws_leaf.extend(fws_push_leaf)

        return fws_list, fws_leaf, fws_root

    def get_as_embedded(self, fws_root=[]):
        """Return as embeded FireWorks list without pull and push stub."""

        fws_main, fws_main_leaf, fws_main_root = self.main(fws_root)

        fws_list = [*fws_main]
        fws_root = [*fws_main_root]
        fws_leaf = [*fws_main_leaf]

        return fws_list, fws_leaf, fws_root

    def build_wf(self):
        """Return self-sufficient pull->main->push workflow """
        fw_list, _, _ = self.get_as_independent()
        return Workflow(
            fw_list, name=self.get_wf_label(), metadata=self.kwargs)

    def inspect_inputs(self):
        """Return fw : _files_in dict of main sub-wf expected inputs."""
        _, _, fws_root = self.main()
        return {fw.name: fw.spec['_files_in'] for fw in fws_root}

    def inspect_outputs(self):
        """Return fw : _files_out dict of main sub-wf produced outputs."""
        _, fws_leaf, _ = self.main()
        return {fw.name: fw.spec['_files_out'] for fw in fws_leaf}

    @property
    def files_in_list(self):
        return [
            {
                'file_label': label,
                'file_name': name,
            }
            for file in self.inspect_inputs().values()
            for label, name in file.items()
        ]

    @property
    def files_out_list(self):
        return [
            {
                'file_label': label,
                'file_name': name,
            }
            for file in self.inspect_outputs().values()
            for label, name in file.items()
        ]


class ProcessAnalyzeAndVisualize(WorkflowGenerator):
    """Merges three sub-workflows 'main', 'vis' and 'analysis' as shown below.


                        + - - - - -+                      + - - - - - -+
                        ' main     '                      ' analysis   '
                        '          '                      '            '
                        ' +------+ '                      '            '
                        ' | pull | '                      '            '
                        ' +------+ '                      '            '
                        '   |      '                      '            '
                        '   |      '                      '            '
                        '   v      '                      '            '
     fws_root inputs    ' +------+ '                      ' +--------+ '
    ------------------> ' |      | ' -------------------> ' |  main  | '
                        ' |      | '                      ' +--------+ '
                        ' |      | '     +- - - - - +     '            '
                        ' |      | '     ' vis      '     '            '
                        ' | main | '     '          '     + - - - - - -+
                        ' |      | '     '          '         |
                        ' |      | '     '          '         |
                        ' |      | '     '          '         |
                        ' |      | '     '          '         |
                        ' |      | ' -+  '          '         |
                        ' +------+ '  |  '          '         |
                        '          '  |  '          '         |
                        + - - - - -+  |  '          '         |
                            |         |  ' +------+ '  opt    |
                            |         +> ' | main | ' <-------+
                            |            ' +------+ '         |
                            |            '          '         |
                            |            +- - - - - +         |
                            |               |                 |
                            +---------------+-----------------+
                                            | fws_leaf outputs
                                            v


    This allows a three-parts sub-worklfow of standardized sub-branches, i.e.
    - main: main processing part
    - analysis: post-processing tasks performed on results of main body
    - vis: visualization of refults from main body and (optionally) analyisis
        branch
    - according pull and push stubs
    """

    def __init__(self, *args, main_sub_wf=None, analysis_sub_wf=None, vis_sub_wf=None,
                 vis_depends_on_analysis=False, **kwargs):
        """Takes list of instantiated sub-workflows."""
        super().__init__(*args, **kwargs)
        kwargs["wf_name_prefix"] = self.wf_name_prefix
        self.sub_wf_components = {
            'main': main_sub_wf(*args, **kwargs) if main_sub_wf is not None else None,
            'analysis': analysis_sub_wf(*args, **kwargs) if analysis_sub_wf is not None else None,
            'vis': vis_sub_wf(*args, **kwargs) if vis_sub_wf is not None else None,
        }
        self._vis_depends_on_analysis = vis_depends_on_analysis


    def push_infiles(self, fp):
        """fp: FilePad"""
        for sub_wf in self.sub_wf_components.values():
            if sub_wf is not None and hasattr(sub_wf, 'push_infiles'):
                sub_wf.push_infiles(fp)

    # chain sub-workflows
    def pull(self, fws_root=[]):
        fw_list_out, fws_leaf_out, fws_root_out = super().pull(fws_root)
        fw_list_sub, fws_leaf_sub, fws_root_sub = self.sub_wf_components['main'].pull(fws_root)
        fw_list_out.extend(fw_list_sub)
        fws_leaf_out.extend(fws_leaf_sub)
        fws_root_out.extend(fws_root_sub)
        return fw_list_out, fws_leaf_out, fws_root_out

    def main(self, fws_root=[]):
        # fws_first_sub_wf_root = None
        # fws_prev_sub_wf_leaf = fws_root
        fw_list = []
        fws_leaf = []

        fws_main, fws_main_leaf, fws_main_root = \
            self.sub_wf_components['main'].get_as_embedded(fws_root)
        fw_list.extend(fws_main)
        fws_leaf.extend(fws_main_leaf)

        fw_vis_deps = [*fws_main_leaf]
        if self.sub_wf_components['analysis'] is not None:
            fws_analysis, fws_analysis_leaf, fws_analysis_root = \
                self.sub_wf_components['analysis'].get_as_embedded(fws_main_leaf)
            fw_list.extend(fws_analysis)
            fws_leaf.extend(fws_analysis_leaf)

            if self._vis_depends_on_analysis:
                fw_vis_deps.extend(fws_analysis_leaf)

        if self.sub_wf_components['vis'] is not None:
            fws_vis, fws_vis_leaf, fws_vis_root = \
                self.sub_wf_components['vis'].get_as_embedded(fw_vis_deps)
            fw_list.extend(fws_vis)
            fws_leaf.extend(fws_vis_leaf)

        return fw_list, fws_leaf, fws_main_root


class ChainWorkflowGenerator(WorkflowGenerator):
    """Chains a set of sub-workflows."""
    def __init__(self, *args, sub_wf_components=[], **kwargs):
        """Takes list of instantiated sub-workflows."""
        super().__init__(*args, **kwargs)
        kwargs["wf_name_prefix"] = self.wf_name_prefix
        self.sub_wf_components = [sub_wf(*args, **kwargs) for sub_wf in sub_wf_components]

    def push_infiles(self, fp):
        """fp: FilePad"""
        for sub_wf in self.sub_wf_components:
            if hasattr(sub_wf, 'push_infiles'):
                sub_wf.push_infiles(fp)

    # chain sub-workflows
    # def pull(self, fws_root=[]):
    #     """Returns the pull stub of first sub-workflow in chain."""
    #     fw_list_out, fws_leaf_out, fws_root_out = super().pull(fws_root)
    #     fw_list_sub, fws_leaf_sub, fws_root_sub = self.sub_wf_components[0].pull(fws_root)
    #     fw_list_out.extend(fw_list_sub)
    #     fws_leaf_out.extend(fws_leaf_sub)
    #     fws_root_out.extend(fws_root_sub)
    #     return fw_list_out, fws_leaf_out, fws_root_out

    def main(self, fws_root=[]):
        fws_first_sub_wf_root = None
        fws_prev_sub_wf_leaf = fws_root
        fw_list = []
        for i, sub_wf in enumerate(self.sub_wf_components):
            if i+1 < len(self.sub_wf_components):
                fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                    sub_wf.get_as_leaf(fws_prev_sub_wf_leaf)
            else:
                fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                    sub_wf.get_as_embedded(fws_prev_sub_wf_leaf)

            fws_prev_sub_wf_leaf = fws_last_sub_wf_leaf
            if not fws_first_sub_wf_root:
                fws_first_sub_wf_root = fws_last_sub_wf_root
            fw_list.extend(fws_last_sub_wf)

        return fw_list, fws_last_sub_wf_leaf, fws_first_sub_wf_root

    # def push(self, fws_root=[]):
    #     """Returns the push stub of last sub-workflow in chain."""
    #     fw_list_out, fws_leaf_out, fws_root_out = super().push(fws_root)
    #     fw_list_sub, fws_leaf_sub, fws_root_sub = self.sub_wf_components[-1].push(fws_root)
    #     fw_list_out.extend(fw_list_sub)
    #     fws_leaf_out.extend(fws_leaf_sub)
    #     fws_root_out.extend(fws_root_sub)
    #     return fw_list_out, fws_leaf_out, fws_root_out

    def get_as_independent(self, fws_root=[]):
        """Return a self-sufficient FireWorks list with pull and push stub."""
        fws_first_sub_wf_root = None
        fws_prev_sub_wf_leaf = fws_root
        fw_list = []
        for i, sub_wf in enumerate(self.sub_wf_components):
            if i == 0:
                fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                    sub_wf.get_as_independent(fws_prev_sub_wf_leaf)
            else:
                fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                    sub_wf.get_as_leaf(fws_prev_sub_wf_leaf)

            fws_prev_sub_wf_leaf = fws_last_sub_wf_leaf
            if not fws_first_sub_wf_root:
                fws_first_sub_wf_root = fws_last_sub_wf_root
            fw_list.extend(fws_last_sub_wf)

        return fw_list, fws_last_sub_wf_leaf, fws_first_sub_wf_root

    def get_as_root(self, fws_root=[]):
        """Return as root FireWorks list with pull stub, but no push stub."""
        fws_first_sub_wf_root = None
        fws_prev_sub_wf_leaf = fws_root
        fw_list = []
        for i, sub_wf in enumerate(self.sub_wf_components):
            if i == 0:
                fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                    sub_wf.get_as_independent(fws_prev_sub_wf_leaf)
            elif i+1 < len(self.sub_wf_components):
                fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                    sub_wf.get_as_leaf(fws_prev_sub_wf_leaf)
            else:
                fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                    sub_wf.get_as_embedded(fws_prev_sub_wf_leaf)

            fws_prev_sub_wf_leaf = fws_last_sub_wf_leaf
            if not fws_first_sub_wf_root:
                fws_first_sub_wf_root = fws_last_sub_wf_root
            fw_list.extend(fws_last_sub_wf)

        return fw_list, fws_last_sub_wf_leaf, fws_first_sub_wf_root


    def get_as_leaf(self, fws_root=[]):
        """Return as leaf FireWorks list without pull stub, but with push stub."""

        fws_first_sub_wf_root = None
        fws_prev_sub_wf_leaf = fws_root
        fw_list = []
        for i, sub_wf in enumerate(self.sub_wf_components):
            fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                sub_wf.get_as_leaf(fws_prev_sub_wf_leaf)

            fws_prev_sub_wf_leaf = fws_last_sub_wf_leaf
            if not fws_first_sub_wf_root:
                fws_first_sub_wf_root = fws_last_sub_wf_root
            fw_list.extend(fws_last_sub_wf)

        return fw_list, fws_last_sub_wf_leaf, fws_first_sub_wf_root

    def get_as_embedded(self, fws_root=[]):
        """Return as embeded FireWorks list without pull and push stub."""
        fws_first_sub_wf_root = None
        fws_prev_sub_wf_leaf = fws_root
        fw_list = []
        for i, sub_wf in enumerate(self.sub_wf_components):
            if i+1 < len(self.sub_wf_components):
                fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                    sub_wf.get_as_leaf(fws_prev_sub_wf_leaf)
            else:
                fws_last_sub_wf, fws_last_sub_wf_leaf, fws_last_sub_wf_root = \
                    sub_wf.get_as_embedded(fws_prev_sub_wf_leaf)

            fws_prev_sub_wf_leaf = fws_last_sub_wf_leaf
            if not fws_first_sub_wf_root:
                fws_first_sub_wf_root = fws_last_sub_wf_root
            fw_list.extend(fws_last_sub_wf)

        return fw_list, fws_last_sub_wf_leaf, fws_first_sub_wf_root


class BranchingWorkflowGeneratorBlueprint(WorkflowGenerator):
    """Common base for all branching workflows."""

    @property
    def sub_wf_components(self):
        return self._sub_wf_components

    @sub_wf_components.setter
    def sub_wf_components(self, sub_wf_components):
        self._sub_wf_components = sub_wf_components

    @sub_wf_components.deleter
    def sub_wf_components(self):
        del self._sub_wf_components

    def push_infiles(self, fp):
        """fp: FilePad"""
        for sub_wf in self.sub_wf_components:
            if hasattr(sub_wf, 'push_infiles'):
                sub_wf.push_infiles(fp)

    def main(self, fws_root=[]):
        fws_list, fws_sub_wf_leaf, fws_sub_wf_root = [], [], []
        for sub_wf in self.sub_wf_components:
            cur_fws_list, cur_fws_sub_wf_leaf, cur_fws_sub_wf_root = sub_wf.main(fws_root)
            fws_list.extend(cur_fws_list)
            fws_sub_wf_leaf.extend(cur_fws_sub_wf_leaf)
            fws_sub_wf_root.extend(cur_fws_sub_wf_root)
        return fws_list, fws_sub_wf_leaf, fws_sub_wf_root

    def get_as_independent(self, fws_root=[]):
        """Return a self-sufficient FireWorks list with pull and push stub."""

        fws_list_out, fws_leaf_out, fws_root_out = [], [], []

        for sub_wf in self.sub_wf_components:
            cur_fw_list, cur_fws_leaf, cur_fws_root = sub_wf.get_as_independent(fws_root)

            fws_list_out.extend(cur_fw_list)
            fws_leaf_out.extend(cur_fws_leaf)
            fws_root_out.extend(cur_fws_root)

        return fws_list_out, fws_leaf_out, fws_root_out

    def get_as_root(self, fws_root=[]):
        """Return as root FireWorks list with pull stub, but no push stub."""

        fws_list_out, fws_leaf_out, fws_root_out = [], [], []

        for sub_wf in self.sub_wf_components:
            cur_fw_list, cur_fws_leaf, cur_fws_root = sub_wf.get_as_root(fws_root)

            fws_list_out.extend(cur_fw_list)
            fws_leaf_out.extend(cur_fws_leaf)
            fws_root_out.extend(cur_fws_root)

        return fws_list_out, fws_leaf_out, fws_root_out

    def get_as_leaf(self, fws_root=[]):
        """Return as leaf FireWorks list without pull stub, but with push stub."""

        fws_list_out, fws_leaf_out, fws_root_out = [], [], []

        for sub_wf in self.sub_wf_components:
            cur_fw_list, cur_fws_leaf, cur_fws_root = sub_wf.get_as_leaf(fws_root)

            fws_list_out.extend(cur_fw_list)
            fws_leaf_out.extend(cur_fws_leaf)
            fws_root_out.extend(cur_fws_root)

        return fws_list_out, fws_leaf_out, fws_root_out

    def get_as_embedded(self, fws_root=[]):
        """Return as embeded FireWorks list without pull and push stub."""
        fws_list_out, fws_leaf_out, fws_root_out = [], [], []

        for sub_wf in self.sub_wf_components:
            cur_fw_list, cur_fws_leaf, cur_fws_root = sub_wf.get_as_embedded(fws_root)

            fws_list_out.extend(cur_fw_list)
            fws_leaf_out.extend(cur_fws_leaf)
            fws_root_out.extend(cur_fws_root)

        return fws_list_out, fws_leaf_out, fws_root_out


class BranchingWorkflowGenerator(BranchingWorkflowGeneratorBlueprint):
    """Assemble a set of sub-workflows in parallel."""

    def __init__(self, *args, sub_wf_components=[], **kwargs):
        """Takes list of sub-workflow classes."""
        super().__init__(*args, **kwargs)
        kwargs["wf_name_prefix"] = self.wf_name_prefix
        self.sub_wf_components = [sub_wf(*args, **kwargs) for sub_wf in sub_wf_components]


class ParametricBranchingWorkflowGenerator(BranchingWorkflowGeneratorBlueprint):
    """Parametric branching of workflow.

    args
    ----
    - sub_wf: WorkflowGenerator
    - parameter_values: [ { k: [v] } ]
        list of dict of parameter label: list of parameter values, i.e.
        [{'nmolecules': [100]}]

    examples
    --------

        parameter_values = [
            {'nmolecules': [100, 200, 300], 'temperature': [10]},
            {'nmolecules': [400], 'temperature': [20, 30]}]

    will result in the following tuples

        [(100, 10), (200, 10), (300, 10), (400, 20), (400, 30)]

    or labeled tuples

        [{'nmolecules': 100, 'temperature': 10},
         {'nmolecules': 200, 'temperature': 10},
         {'nmolecules': 300, 'temperature': 10},
         {'nmolecules': 400, 'temperature': 20},
         {'nmolecules': 400, 'temperature': 30}]

    of ('nmolecules', temperature') parameter values
    """

    def __init__(self, *args, sub_wf=None, **kwargs):
        labeled_parameter_sets = []
        super().__init__(*args, **kwargs)

        # build atomic parameter sets from parameter_values
        for parameter_package in self.kwargs['parameter_values']:
            expanded_parameter_package = list(
                itertools.product(
                    *[p if isinstance(p, Iterable) else [p] for p in parameter_package.values()]
                ))
            labeled_parameter_set = [{
                    k: v for k, v in zip(
                        parameter_package.keys(), parameter_set)
                } for parameter_set in expanded_parameter_package]
            labeled_parameter_sets.extend(labeled_parameter_set)

        # build one sub-workflow for each parameter set
        sub_wf_components = []
        for parameter_set in labeled_parameter_sets:
            cur_kwargs = copy.deepcopy(kwargs)
            for k, v in parameter_set.items():
                parameter_key = self.parameter_label_key_dict[k]
                cur_kwargs = set_nested_dict_value(cur_kwargs, parameter_key, v)
            sub_wf_components.append(sub_wf(*args, **cur_kwargs))

        self.sub_wf_components = sub_wf_components



# TODO: cast below into workflow
#     def __init__( self,


#         mongodb_host = 'localhost',
#         mongodb_port = 27017,
#         mongodb_name = 'fireworks-testuser',
#         mongodb_user = 'fireworks',
#         mongodb_pwd  = 'fireworks',
#         lmp_cmd = ' '.join((
#             'module use /gpfs/homea/hka18/hka184/modules/modulefiles;',
#             'module load jlh/lammps/16Mar18-intel;',
#             'srun lmp -in {inputFile:s}' )),
#         queue           = 'JUWELS',
#         template_prefix = '/gpfs/homea/hka18/hka184/jobs/lmplab/sds/201808/N_SDS_on_AU_111_template',
#         output_prefix   = '/gpfs/homea/hka18/hka184/jobs/lmplab/sds/201808/sys',
#         sim_df = None ):
#         '''Establishes connection to Fireworks MongoDB.'''

#         self.lpad = LaunchPad(
#             host=mongodb_host,
#             port=mongodb_port,
#             name=mongodb_name,
#             username=mongodb_user,
#             password=mongodb_pwd)

#         self.fp   = FilePad(
#             host=mongodb_host,
#             port=mongodb_port,
#             database=mongodb_name,
#             username=mongodb_user,
#             password=mongodb_pwd)

#         self.template_prefix    = template_prefix
#         self.output_prefix      = output_prefix

#         self.lmp_cmd            = lmp_cmd
#         self._template_lmp_cmd  = lmp_cmd
#         self._geninfo = None

#         if queue == 'JUWELS':
#             self.use_juwels_queue()
#             self.use_juwels_templates()
#         elif queue == 'NEMO':
#             self.use_nemo_queue()
#             self.use_nemo_templates()
#         elif queue == 'BWCLOUD':
#             self.use_bwcloud_queue()
#             self.use_bwcloud_templates()
#         else:
#             raise ValueError("Queue '{:s}' does not exist!".format(queue))

#         self._set_machine_independent_templates()

#         if sim_df is not None:
#             self._sim_df = sim_df

#     def bryan_test(self):
#         print("New test function added on 29th Jan 2019")

#     def use_bwcloud_templates(self):
#         # TODO: find way to wrap environment elegantly elsewhere
#         self._template_sb_replicate_cmd = ' '.join((
#             'module load GROMACS GROMACS-Top MDTools;',
#             'replicate.sh {multiples[0]:d} {multiples[1]:d}',
#             '{multiples[2]:d} {plane:03d}'))

#         self._template_packmol_cmd = ' '.join((
#             'module load MDTools;',
#             'packmol < {infile:s}' ))

#         self._template_packmol2gmx_cmd  = ' '.join((
#             'module load MDTools;',
#             'pdb_packmol2gmx.sh {infile:s}' ))

#         self._template_gmx2pdb_cmd = ' '.join((
#             'module load GROMACS GROMACS-Top VMD;',
#             'bash {infile:s}' ))

#         # psfgen
#         self._template_psfgen_cmd  = ' '.join((
#             'module load VMD;',
#             'vmd -e {infile:s}' ))

#         # charmm2lammps.pl
#         self._template_ch2lmp_cmd  = ' '.join((
#             'module load MDTools;',
#             'charmm2lammps.pl all36_lipid_extended_stripped {system_name:s}_psfgen',
#             '-border=0 -lx={box[0]:.3f} -ly={box[1]:.3f} -lz={box[2]:.3f} ' ))

#         # lammps
#         self._template_lmp_cmd = lmp_cmd = ' '.join((
#             'module load LAMMPS;',
#             'mpirun ${{MPIRUN_OPTIONS}} lmp -in {inputFile:s}' ))

#         # VMD
#         self._template_vmd_cmd = ' '.join((
#             'module load VMD;',
#             'vmd -eofexit -e {infile:s}' ))

#         # image conversion
#         self._template_convert_cmd = 'convert "{infile:s}" "{outfile:s}"'

#         self._template_pizzapy_merge_cmd = ' '.join((
#             'module load MDTools/jlh-25Jan19-python-2.7;', # pizza.py contained her
#             'pizza.py -f merge.py "{datafile:s}" "{reffile:s}" "{outfile:s}"' ))

#         # netcdf2data.py
#         self._template_netcdf2data_cmd = ' '.join((
#             'module load MDTools Ovito;',
#             'netcdf2data.py {datafile:s} {trajfile:s}'))

#     def use_nemo_templates(self):
#         # TODO: find way to wrap environment elegantly elsewhere
#         self._template_sb_replicate_cmd = '{template_prefix:s}' + os.sep \
#             + "replicate.sh {multiples[0]:d} {multiples[1]:d} " \
#             + "{multiples[2]:d} {plane:03d}"

#         self._template_packmol_cmd = ' '.join((
#             'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
#             'module use ${{HOME}}/modulefiles;',
#             'module load mdtools;',
#             'packmol < {infile:s}' ))

#         self._template_packmol2gmx_cmd  = '{template_prefix:s}' \
#             + os.sep + 'pdb_packmol2gmx.sh {infile:s}' # make path-indep

#         self._template_gmx2pdb_cmd = ' '.join((
#             'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
#             'module use ${{HOME}}/modulefiles;',
#             'module load gromacs/2018.1-gnu-5.2;',
#             'bash {infile:s}' ))

#         # psfgen
#         self._template_psfgen_cmd  = ' '.join((
#             'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
#             'module use ${{HOME}}/modulefiles;',
#             'module load vmd/1.9.3-text;',
#             'vmd -e {infile:s}' ))

#         # charmm2lammps.pl
#         self._template_ch2lmp_cmd  = ' '.join((
#             'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
#             'module use ${{HOME}}/modulefiles;',
#             'module load mdtools;',
#             'charmm2lammps.pl all36_lipid_extended_stripped {system_name:s}_psfgen',
#             '-border=0 -lx={box[0]:.3f} -ly={box[1]:.3f} -lz={box[2]:.3f} ' ))

#         # lammps
#         self._template_lmp_cmd = lmp_cmd = ' '.join((
#             'module purge;',
#             'module load lammps/16Mar18-gnu-7.3-openmpi-3.1-colvars-09Feb19;',
#             'mpirun ${{MPIRUN_OPTIONS}} lmp -in {inputFile:s}' ))

#         # VMD
#         self._template_vmd_cmd = ' '.join((
#             'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
#             'module use ${{HOME}}/modulefiles;',
#             'module load vmd/1.9.3-text;',
#             'vmd -eofexit -e {infile:s}' ))

#         # image conversion
#         self._template_convert_cmd = 'convert "{infile:s}" "{outfile:s}"'

#         #self._template_pizzapy_merge_cmd = None # not set up on NEMO yet
#         self._template_pizzapy_merge_cmd = ' '.join((
#             'module use ${{HOME}}/modulefiles;',
#             'type deactivate && deactivate;', # deactivate virtual env if loaded
#             'module --force purge;',
#             'module load devel/python/2.7.14 mpi/openmpi/2.1-gnu-4.8;',
#             'source ${{HOME}}/venv/jlh-nemo-python-2.7/bin/activate;',
#             'module load jlh/mdtools/26Jun18-jlh-python-2.7;', # pizza.py contained her
#             'pizza.py -f merge.py "{datafile:s}" "{reffile:s}" "{outfile:s}"' ))

#         # netcdf2data.py
#         self._template_netcdf2data_cmd = ' '.join((
#             'module use /work/ws/nemo/fr_lp1029-IMTEK_SIMULATION-0/modulefiles;',
#             'module use ${{HOME}}/modulefiles;',
#             'module load ovito/3.0.0-dev234;',
#             'module load jlh/mdtools/26Jun18-jlh;',
#             'netcdf2data.py {datafile:s} {trajfile:s}'))

#     def use_juwels_templates(self):
#         # LAMMPS
#         self._template_lmp_cmd = lmp_cmd = ' '.join((
#             'module use /gpfs/homea/hka18/hka184/modules/modulefiles;',
#             'module load jlh/lammps/16Mar18-intel;',
#             'srun lmp -in {inputFile:s}' ))

#         # VMD
#         self._template_vmd_cmd = ' '.join((
#             'module use ${{HOME}}/modules/modulefiles;',
#             'module load jlh/vmd/1.9.3-text;',
#             'vmd -eofexit -e {infile:s}' ))

#         # image conversion
#         self._template_convert_cmd = ' '.join((
#             'module --force purge;',
#             'module load GCCcore/.5.5.0 ImageMagick/6.9.9-19;',
#             'convert "{infile:s}" "{outfile:s}"' ))

#         # pizza-py merge datafiles
#         self._template_pizzapy_merge_cmd = ' '.join((
#             'module use ${{HOME}}/modules/modulefiles;',
#             'type deactivate && deactivate;', # deactivate virtual env if loaded
#             'module --force purge;',
#             'module load GCC/5.5.0 Python/2.7.14;',
#             'source ${{HOME}}/venv/jlh-juwels-python-2.7/bin/activate;',
#             'module load jlh/mdtools/26Jun18-jlh-python-2.7;', # pizza.py contained her
#             'pizza.py -f merge.py "{datafile:s}" "{reffile:s}" "{outfile:s}"' ))

#     def _set_machine_independent_templates(self):
#         self._template_geninfo = \
#             "Generated on {datetime:s} by {user:s}@{machine:s}"

#         self._template_substrate_name = \
#             "AU_{plane:03d}_{multiples[0]:d}x{multiples[1]:d}x{multiples[2]:d}"

#         self._sb_replicate_suffix = '_sb_replicate'
#         self._packmol_suffix      = '_packmol'
#         self._packmol2gmx_suffix  = '_packmol2gmx'
#         self._gmx_solvate_suffix  = '_gmx_solvate'
#         self._gmx2pdb_suffix      = '_gmx2pdb'
#         self._psfgen_suffix       = '_psfgen'
#         self._ch2lmp_suffix       = '_ch2lmp'

#     # 96 tasks per node is standard setting on JUWELS
#     def use_bwcloud_queue( self,
#         packmol_queueadapter = {},
#         minimization_queueadapter = {},
#         equilibration_queueadapter = {},
#         production_queueadapter   = {},
#         std_worker          = 'bwcloud_std',
#         queue_worker        = 'bwcloud_std',
#         queue_worker_serial = 'bwcloud_std'):

#         self.std_worker         = std_worker
#         self.queue_worker       = queue_worker
#         self.queue_worker_serial= queue_worker_serial
#         self.packmol_queueadapter      = packmol_queueadapter
#         self.minimization_queueadapter = minimization_queueadapter
#         self.equilibration_queueadapter = equilibration_queueadapter
#         self.production_queueadapter   = production_queueadapter

#     # NEMO std settings 20 ppn
#     def use_nemo_queue( self,
#         packmol_queueadapter = {
#             'walltime' : '24:00:00',
#             'nodes':       1,
#             'ppn':         1,
#         },
#         minimization_queueadapter = {
#             'walltime' : '04:00:00',
#             'nodes':       1
#         },
#         equilibration_queueadapter = {
#             'walltime' : '24:00:00',
#             'nodes':       2,
#         },
#         production_queueadapter = {
#             'walltime' : '96:00:00',
#             'nodes':       4,
#         },
#         std_worker          = 'nemo_noqueue',
#         queue_worker        = 'nemo_queue',
#         queue_worker_serial = 'nemo_queue_serial'):

#         self.std_worker         = std_worker
#         self.queue_worker       = queue_worker
#         self.queue_worker_serial= queue_worker_serial
#         self.packmol_queueadapter      = packmol_queueadapter
#         self.minimization_queueadapter = minimization_queueadapter
#         self.equilibration_queueadapter = equilibration_queueadapter
#         self.production_queueadapter   = production_queueadapter

#     # 96 tasks per node is standard setting on JUWELS
#     def use_juwels_queue( self,
#         # nocont walltime 6h
#         packmol_queueadapter = {
#             'walltime' : '06:00:00',
#             'ntasks':      1,
#         },
#         minimization_queueadapter = {
#             'walltime' : '06:00:00',
#             'ntasks':    96
#         },
#         equilibration_queueadapter = {
#             'walltime' : '06:00:00',
#             'ntasks':     96
#         },
#         production_queueadapter   = {
#             'walltime' : '06:00:00',
#             'ntasks'   : 192
#         },
#         std_worker          = 'juwels_std',
#         queue_worker        = 'juwels_queue',
#         queue_worker_serial = 'juwels_queue_serial'):

#         self.std_worker         = std_worker
#         self.queue_worker       = queue_worker
#         self.queue_worker_serial= queue_worker_serial
#         self.packmol_queueadapter      = packmol_queueadapter
#         self.minimization_queueadapter = minimization_queueadapter
#         self.equilibration_queueadapter = equilibration_queueadapter
#         self.production_queueadapter   = production_queueadapter

#     def geninfo(self, reset=False, static_str=None):
#         if static_str is not None:
#             self._geninfo = static_str
#         elif reset or (self._geninfo is None):
#             user = os.getlogin()
#             machine = os.uname().nodename
#             self._geninfo = self._template_geninfo.format(
#                 datetime=datetime.datetime.now().ctime(),
#                 user=user,
#                 machine=machine)
#         return self._geninfo

# # Queries

#     def identifyIncompleteWorkflows(self,
#         precedent_step_name='packmol', subsequent_step_name='ch2lmp'):
#         """Expects name of two (not necessarily directly) subsequent steps,
#         and returns a dict of dict, identifying Firework IDs of
#         (healthy, i.e. non-fizzled) preceding steps that do not have
#         healthy subsequent steps.

#         Returns:
#         --------
#           {str: {int: [int] } dict of dict of list, i.e.
#               { system_name: { workflow_id: [firework_id] }
#         """

#         # get ALL healthy "preceding step" fireworks:
#         preceding_step_dict = self.query_step(step=precedent_step_name,
#             state_list = ['COMPLETED','RUNNING','RESERVED','READY','WAITING'])

#         # system_name, wf_id : list(fw_id) dict
#         preceding_step_wf_dict_dict = {}
#         # system_name : set(wf_id) dict
#         preceding_step_wf_set_dict = {}
#         # go through ALL found fireworks and "group by" workflow
#         # workflows are indentified by their "zero-eth" root firework
#         for system_name, fw_id_lst in preceding_step_dict.items():
#             root_fw_id_lst = []
#             preceding_step_wf_dict_dict[system_name] = {}
#             for fw_id in fw_id_lst:
#                 # identify workflow uniquely by first root fw id
#                 wf_id = self.lpad.get_wf_by_fw_id( fw_id ).root_fw_ids[0]
#                 root_fw_id_lst.append( wf_id )
#                 if wf_id in preceding_step_wf_dict_dict[system_name]:
#                     preceding_step_wf_dict_dict[system_name][wf_id].append(fw_id)
#                 else:
#                     preceding_step_wf_dict_dict[system_name][wf_id] = [fw_id]
#             preceding_step_wf_set_dict[system_name] = set( root_fw_id_lst )

#         # get ALL healthy "subsequent step" fireworks:
#         subsequent_step_dict = self.query_step(step=subsequent_step_name,
#             state_list = ['COMPLETED','RUNNING','RESERVED','READY','WAITING'])

#         # system_name, wf_id : list(fw_id) dict
#         subsequent_step_wf_dict_dict = {}
#         # system_name : set(wf_id) dict
#         subsequent_step_wf_set_dict = {}
#         # again group by workgflow:
#         for system_name, fw_id_lst in subsequent_step_dict.items():
#             root_fw_id_lst = []
#             subsequent_step_wf_dict_dict[system_name] = {}
#             for fw_id in fw_id_lst:
#                 # identify workflow uniquely by first root fw id
#                 wf_id = self.lpad.get_wf_by_fw_id( fw_id ).root_fw_ids[0]
#                 root_fw_id_lst.append( wf_id )
#                 if wf_id in subsequent_step_wf_dict_dict[system_name]:
#                     subsequent_step_wf_dict_dict[system_name][wf_id].append(fw_id)
#                 else:
#                     subsequent_step_wf_dict_dict[system_name][wf_id] = [fw_id]

#             # this set dict just holds sets of all workflows (identified by
#             # root fw_id) containing at least one healthy subsequent step
#             # for the system identified by "key"
#             subsequent_step_wf_set_dict[system_name] = set( root_fw_id_lst )

#         # first filter all workflows that have at least one healthy firework
#         # of the "precedent" step, but none of the "subsequent" step:

#         # compare keys (the case of "subsequent steps" without the according
#         # "preceding step" is assumed not to occur)
#         non_existent_subsequent = preceding_step_wf_set_dict.keys() - subsequent_step_wf_set_dict.keys()

#         # treatment of systems non-existent in consecutive step
#         wf_discrepancy_dict = {}
#         if non_existent_subsequent: #non-empty:
#             logging.warn(
#                 "{:d} systems not present at all in subsequent step.".format(
#                     len(non_existent_subsequent)))
#             wf_discrepancy_dict = {
#                 system_name: preceding_step_wf_set_dict[system_name] \
#                 for system_name in non_existent_subsequent }

#         # next look at systems, where at least one workflow exists holding both
#         # healthy preceding and subsequent step
#         system_overlap = preceding_step_wf_set_dict.keys() & subsequent_step_wf_set_dict.keys()
#         logging.info(
#             "{:d} systems both present in preceding and subsequent step.".format(
#                 len(system_overlap)))
#         # within those sets, identify the workflows not holding at least one
#         # healthy subsequent step
#         for system_name in system_overlap:
#             diff = preceding_step_wf_set_dict[system_name] - subsequent_step_wf_set_dict[system_name]
#             if diff:
#                 logging.info( ' '.join((
#                     "System {:s} has {:d} systems".format(system_name, len(diff)),
#                     "present in preceding step, but not in subsequent step")) )
#                 wf_discrepancy_dict[system_name] = diff

#         # get fireworks ids within those workflows
#         fw_discrepancy_dict = {}
#         for system_name, wf_id_set in wf_discrepancy_dict.items():
#             fw_discrepancy_dict[system_name] = {}

#             for wf_id in wf_id_set:
#                 fw_discrepancy_dict[system_name][wf_id] = preceding_step_wf_dict_dict[system_name][wf_id]

#         # TODO
#         # ATTENTION: so far the "precedent" - "subsequent" naming is misleading
#         # it is very well possible that a workflow contains forked branches
#         # with both healthy preceding and subsequent steps in one branch, but
#         # only healthy preceding and no (or unhealthy) subsequent step
#         # in the other. This workflow (and the latter preceding firework) will
#         # not be identified.
#         return fw_discrepancy_dict

#     def identifyIncompleteWorkflowsInScope(self,
#         precedent_step_name, subsequent_step_name, system_name_scope):
#         fw_dict = self.identifyIncompleteWorkflows(
#             precedent_step_name, subsequent_step_name)
#         # key is system_name, values are wf_id: [ fw_id ] dicts
#         in_scope_dict = { key: fw_dict[key] for key in
#                           fw_dict.keys() & system_name_scope }
#         # flatten second dimension of in_scope_dict (remove wf ids)
#         in_scope_dict = { system_name: [
#             fw_id for fw_id_lst in fw_id_lst_dict.values() for fw_id in fw_id_lst ]
#             for system_name, fw_id_lst_dict in in_scope_dict.items() }
#         return in_scope_dict

#     def query_step( self,
#                     step = 'prepare_system_files',
#                     state_list = ['COMPLETED', 'RUNNING', 'RESERVED', 'READY',
#                         'WAITING'],
#                     group_key = 'system_name',
#                     step_key  = 'step'):
#         '''queries spec.step_key, groups by system_name, returns dict'''

#         fw_id_list = self.lpad.get_fw_ids(
#             query = {
#                 'spec.{}'.format(step_key):  step,
#                 'state':                     { '$in': state_list },
#                 'spec.{}'.format(group_key): { '$exists': True }
#             },
#             sort = [('updated_on', pymongo.ASCENDING)] )

#         logging.info("fw_id_list: {}".format(fw_id_list))

#         system_name_list = [
#             self.lpad.get_fw_dict_by_id(fw_id)["spec"][group_key] \
#                 for fw_id in fw_id_list ]

#         system_name_set = set(system_name_list)
#         system_names = list(system_name_set)

#         fw_id_dict = {}
#         for system_name in system_names:
#             fw_id_dict[system_name] = self.lpad.get_fw_ids(
#             query = {
#                 'spec.{}'.format(step_key):  step,
#                 'spec.{}'.format(group_key): system_name,
#                 'state':                     { '$in': state_list }
#             },
#             sort = [('updated_on', pymongo.ASCENDING)] )

#         return fw_id_dict

#     def _update_query(self, query, key, obj):
#         """ query: dict of obj, key: str, obj: int, str, dict or list """
#         if obj is not None:
#             # add allowed types here:
#             if (type(obj) is int) or (type(obj) is str) or (type(obj) is dict):
#                 query.update( {key: obj} )
#             elif (type(obj) is list):
#                 query.update( {key: { '$in': obj }} )
#             else:
#                 raise ValueError( ' '.join((
#                     "Value of {} must be int, str, dict or list,".format(key),
#                     "but is {}".format(type(obj)) )) )

#     def query_systems( self, system_names,
#         step = None, state = None, restart=None ):
#         """Returns a dictionary str : list of int  with each system's
#         Firework ids sorted ascending by update time (latest last)"""

#         fw_id_list = []
#         for system_name in system_names:
#             # earlier inconsistent keys
#             query = { '$or': [
#                 {'spec.system_name':   system_name },
#                 {'spec.system':        system_name } ]
#             }
#             self._update_query(query, 'state',        state )
#             self._update_query(query, 'spec.step',    step )
#             self._update_query(query, 'spec.restart', restart )

#             logging.info( '{:s} query: {}'.format( system_name, query ) )

#             fw_id_list.append(
#                 self.lpad.get_fw_ids(
#                     query = query,
#                     sort = [('updated_on', pymongo.ASCENDING)] ) )

#             fw_id_dict = dict( zip( system_names, fw_id_list ) )
#         return fw_id_dict

#     def get_set_of_prepared_morphologies( self ):
#         # eayrlier, used key 'system' instead of 'system_name' earlier
#         fw_id_dict = self.query_step( 'ch2lmp', state_list=['COMPLETED'],
#             group_key='system_name' )
#         return set( fw_id_dict.keys() )

#     def get_set_of_healthy_morphologies( self ):
#         fw_id_dict = self.query_step( 'ch2lmp', state_list=[
#             'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
#         return set( fw_id_dict.keys() )

#     def get_set_of_prepared_minimizations( self ):
#         fw_id_dict = self.query_step( 'prepare_system_files',
#             state_list=['COMPLETED'] )
#         return set( fw_id_dict.keys() )

#     def get_set_of_healthy_minimization_preparations( self ):
#         fw_id_dict = self.query_step( 'prepare_system_files', state_list=[
#             'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
#         return set( fw_id_dict.keys() )

#     def get_set_of_minimized_systems( self ):
#         fw_id_dict = self.query_step( 'minimization',
#             state_list = ['COMPLETED'] )
#         return set( fw_id_dict.keys() )

#     def get_set_of_healthy_minimizations( self ):
#         fw_id_dict = self.query_step( 'minimization', state_list=[
#             'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
#         return set( fw_id_dict.keys() )

#     def get_set_of_nvt_equilibrated_systems( self  ):
#         fw_id_dict = self.query_step( 'equilibration_nvt',
#             state_list = ['COMPLETED'] )
#         return set( fw_id_dict.keys() )

#     def get_set_of_healthy_nvt_equilibrations( self ):
#         fw_id_dict = self.query_step( 'equilibration_nvt', state_list=[
#             'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
#         return set( fw_id_dict.keys() )

#     def get_set_of_npt_equilibrated_systems( self ):
#         fw_id_dict = self.query_step( 'equilibration_npt',
#             state_list = ['COMPLETED'] )
#         return set( fw_id_dict.keys() )

#     def get_set_of_healthy_npt_equilibrations( self ):
#         fw_id_dict = self.query_step( 'equilibration_npt', state_list=[
#             'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ])
#         return set( fw_id_dict.keys() )

#     def get_set_of_evolved_systems( self ):
#         production_10ns_systems_dict = self.query_step(
#             '10ns_production_mixed', state_list = [ 'COMPLETED' ] )
#         return set( production_10ns_systems_dict.keys() )

#     def get_set_of_fizzled_production_runs( self ):
#         fizzled_production_10ns_systems_dict = self.query_step(
#             '10ns_production_mixed',
#             state_list=[ 'FIZZLED' ])
#         return set( fizzled_production_10ns_systems_dict.keys() )

#     def get_set_of_healthy_production_runs( self ):
#         healthy_production_10ns_systems_dict = self.query_step(
#             '10ns_production_mixed', state_list = [ 'COMPLETED', 'RESERVED',
#             'RUNNING', 'READY', 'WAITING' ])
#         return set( healthy_production_10ns_systems_dict.keys() )

#     # set operations
#     def get_to_prepare( self ):
#         systems_to_prep =  self.get_set_of_prepared_morphologies() \
#             - self.get_set_of_healthy_minimization_preparations()
#         return list(systems_to_prep)

#     def get_to_minimize( self ):
#         systems_to_prep = self.get_set_of_healthy_minimization_preparations() \
#             - self.get_set_of_healthy_minimizations()
#         return list(systems_to_prep)

#     def get_to_nvt_equlibrate( self ):
#         systems_to_prep = self.get_set_of_healthy_minimizations() \
#             - self.get_set_of_healthy_nvt_equilibrations()
#         return list(systems_to_prep)

#     def get_to_npt_equlibrate( self ):
#         systems_to_prep = self.get_set_of_healthy_nvt_equilibrations() \
#             - self.get_set_of_healthy_npt_equilibrations()
#         return list(systems_to_prep)

#     def get_to_run_production( self ):
#         systems_to_prep = self.get_set_of_healthy_npt_equilibrations() \
#             - self.get_set_of_healthy_production_runs()
#         return list(systems_to_prep)

#     # TODO: rework, does not work anymore with current framework
#     # with automatic requeueing after expired walltime
#     def get_discontinued_production_runs( self ):
#         # get all fizzled production Fireworks
#         fizzled_production_10ns_systems_dict = self.query_step(
#             step = '10ns_production_mixed',
#             state_list = [ 'FIZZLED' ] )

#         # query all production workflows not archived
#         fizzled_production_wf_dict = {}
#         for key, fw_ids in fizzled_production_10ns_systems_dict.items():
#             fizzled_production_wf_dict[key] = self.lpad.get_wf_ids(
#                 query = {
#                     'name' : key + '_prepare_system_files',
#                     'state': { '$ne': 'ARCHIVED' } } )

#         # get all leaves of the wfs queried above
#         fizzled_production_wf_leaf_fw_dict = {}
#         for key, wf_id in fizzled_production_wf_dict.items():
#             if len(wf_id) > 0:
#                 wf = self.lpad.get_wf_by_fw_id(wf_id[-1])
#                 fizzled_production_wf_leaf_fw_dict[key] = wf.leaf_fw_ids

#         # intersection of fizzled fw ids and wf leaf ids
#         # (check which fizzled production fw are leaves)
#         fizzled_production_10ns_leaves_dict = {}
#         for key, fw_ids in fizzled_production_10ns_systems_dict.items():
#             if key in fizzled_production_wf_leaf_fw_dict:
#                 intersection = \
#                     set(fizzled_production_wf_leaf_fw_dict[key]) & set(fw_ids)
#                 if intersection != set():
#                     fizzled_production_10ns_leaves_dict[key] = intersection
#         return fizzled_production_10ns_leaves_dict

#     # latest last in list
#     def get_healthy( self, system_names,step='minimization' ):
#         fw_id_dict = { system_name: self.lpad.get_fw_ids(
#             query = {
#                 'spec.step':          step,
#                 'state':              {
#                     '$in': [
#                         'COMPLETED', 'RUNNING', 'RESERVED', 'READY', 'WAITING' ]
#                     },
#                 'spec.system_name':   system_name
#             },
#             sort = [('updated_on', pymongo.ASCENDING)] ) \
#             for system_name in system_names }
#         return fw_id

# # Launches
#     def append_wf_by_key( self, new_wf_dict, prev_fw_id_dict,add_if_missing=False):
#         """Expects a dictionary of new work flows and a dictionary of
#         previously existent fireworks on launchpad. Appends by key."""
#         for key, wf in new_wf_dict.items():
#             if prev_fw_id_dict[key] is None or prev_fw_id_dict[key] == []:
#                 print("{} empty in prev_fw_id_dict".format(key))
#                 if add_if_missing:
#                     self.lpad.add_wf(wf)
#                     print("Added.")
#                 else:
#                     print("Skipped.")
#                     continue

#             print("Append {} to last fw_id in list {}".format(
#                 key, prev_fw_id_dict[key]))

#             if type(wf) is Firework:
#                 wf = Workflow([wf])

#             self.lpad.append_wf(wf,[ prev_fw_id_dict[key][-1]])

#     def append_fw_to_step( self, fw_dict, step='minimization' ):
#         fw_id_dict = self.get_healthy(
#             system_names = list( fw_dict.keys() ), step = step )
#         for system_name, fw in fw_dict.items():
#             print( ' '.join(( "{}: ".format(system_name),
#                 "Append a new Firework to previous Firework",
#                 "of last ID in list {}".format(fw_id_dict[system_name]) )) )
#             self.lpad.append_wf( Workflow([fw]) ,[fw_id_dict[system_name][-1]] )

#     def add_fw( self, fw_dict ):
#         fw_id_dict = { system_name: self.lpad.add_wf( Workflow( fw ) ) \
#                 for system_name, fw in fw_dict.items() }
#         return fw_id_dict

#     def add_wf( self, wf_dict ):
#         fw_id_dict = { system_name: self.lpad.add_wf( wf ) \
#                 for system_name, wf in wf_dict.items() }
#         return fw_id_dict

# # Administration

#     def defuse_children(self, fw_id, wf = None, dry_run = False):
#         """Descends to children recursively and defuses them

#         (including this parent fw_id)

#         Parameters
#         ----------
#         fw_id: int
#         wf : fireworks.Workflow
#             will be determined automatically
#         dry_run : bool
#             Only descends and logs, does not defuse if True

#         Returns
#         -------
#         list of int
#             all defused fw_id
#         """
#         if wf is None:
#             wf = self.lpad.get_wf_by_fw_id(fw_id)

#         defused_fw_id_list = []
#         if fw_id not in wf.links:
#             logging.warn("fw_id {} not in wf.links!".format(fw_id))
#         else:
#             for child_fw_id in wf.links[fw_id]:
#                 defused_fw_id_list.extend( self.defuse_children( child_fw_id ) )

#         if not dry_run:
#             self.lpad.defuse_fw(fw_id, rerun_duplicates=False)

#         defused_fw_id_list.append(fw_id)
#         logging.info("fw_id {} defused".format(fw_id))
#         return defused_fw_id_list

# # Workflows, Fireworks & Firetasks

# # Scripted preassembly of surfactant aggregates at the AU 111 surface.
# #
# # Requires
# #
# # * modified CHARMM36 Jul17 package https://github.com/jotelha/jlh_toppar_c36_jul17
# # * modified GROMACS 2018.1 top folder including charmm36.ff https://github.com/jotelha/jlh_gmx_2018.1_top
# # * modified pdb-tools forked from https://github.com/haddocking/pdb-tools
# # * GROMACS 2018.1
# # * vmd-1.9.3 with psfgen plugin, e.g. pre-compiled distribution vmd-1.9.3.bin.LINUXAMD64.text.tar.gz
# #   from http://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD
# # * LAMMPS 16Mar18
# # * charmm2lammps (comes with LAMMPS 16Mar18 sources)
# #
# # 1. Create GROMACS .hdb, .rtp for surfactants, dummy .rtp for ions
# # 2. Load system description from some stored pandas.Dataframe, e.g. from pickle.
# #    Should contain information on substrate measures, box size, etc.
# #    ADD DETAILED DESCRIPTION OF EXPECTED VALUES.
# # 3. Identify all unique substrate slabs and create according .pdb files from multiples of unit cell with
# #    `gmx genconf`.
# # 4. Create a subfolter for every system and copy (or links) some necessary files
# # 5. Use packmol to create bilayer on gold surface
# # 6. Use `pdb_packmol2gmx` to run sum simple renumbering on residues.
# #    This is mainly necessary because gmx needs each substrate atom to have its own residue number.
# # 7. Create GROMACS .gro and .top from .pdb, solvate (and ionize system).
# #    A bash script suffixed `_gmx_solvate.sh` is created for this purpose.
# # 8. Split .gro system into .pdb chunks of at most 9999 residues
# #    in order not to violate .pdb format restritcions. psfgen cannot work on .pdb > 9999 residues.
# #    The bash script suffixed `_gmx2pdb.sh` does this.
# # 9. Generate .psf with VMD's `psfgen`.
# # 10.Generate LAMMPS data from psgen-generated .psf and .pdb with `charmm2lammps.pl`
# #
# # TODO: Doe solvation and ionization in packmol as well, remove the detour via GROMACS


# # substrates (on NEMO), to be tested
#     def prepare_substrates(self, system_names=None):
#         """Processes attribute _sim_df and constructs substrate slabs

#         _sim_df should have the columns
#           sb_crystal_plane
#           sb_multiples

#         A list of system_names can be used to only process a subset."""

#         if system_names is None:
#             sim_df = self._sim_df
#         else:
#             sim_df = self._sim_df.loc[system_names]

#         unique_sb_crystal_planes = sim_df["sb_crystal_plane"].unique()

#         sb_names = []
#         sb_replicate_fw_list = []
#         for plane in unique_sb_crystal_planes:
#             unique_sb_multiples = np.unique(
#                    np.vstack( sim_df[
#                        sim_df["sb_crystal_plane"] == plane
#                    ][["sb_multiples"]].values.flatten() ) ,
#                    axis=0 )
#             for multiples in unique_sb_multiples:
#                 sb_name = self._template_substrate_name.format(
#                     plane=plane, multiples=multiples )
#                 sb_names.append(sb_name)

#                 # create all substrate files
#                 sb_get_unit_cell_ft = FileTransferTask( {
#                     'files': [ {
#                         'src':  self.template_prefix + os.sep \
#                             + 'au_cell_P1_111.gro', # allow other surfaces here
#                         'dest': 'au_cell_P1_111.gro' } ],
#                     'mode': 'copy' } )

#                 sb_replicate_ft =  ScriptTask.from_str(
#                     self._template_sb_replicate_cmd.format(
#                         template_prefix = self.template_prefix,
#                         plane           = plane,
#                         multiples       = multiples ) , {
#                     'stdout_file':  sb_name + self._sb_replicate_suffix + '.out',
#                     'stderr_file':  sb_name + self._sb_replicate_suffix + '.err',
#                     'use_shell':    True,
#                     'fizzle_bad_rc':True } )

#                 # Attention: Explicitly overwrite previously stored file in DB:
#                 sb_delete_ft = DeleteFilesTask( {
#                     'identifiers': ["{:s}.pdb".format(sb_name)]})
#                 sb_store_ft =  AddFilesTask( {
#                     'paths':       ["{:s}_tidy.pdb".format(sb_name)],
#                     'identifiers': ["{:s}.pdb".format(sb_name)]})

#                 sb_replicate_fw = Firework( [
#                         sb_get_unit_cell_ft,
#                         sb_replicate_ft,
# 			sb_delete_ft,
#                         sb_store_ft
#                     ],
#                     spec={
#                         "_category":   self.std_worker,
#                         "_dupefinder": DupeFinderExact(),
#                         "substrate":   sb_name,
#                         "step"     :   "sb_replicate",
#                         "geninfo"  :   self.geninfo()
#                     },
#                 name="{:s}_sb_replicate".format(sb_name) )

#                 sb_replicate_fw_list.append( [ sb_replicate_fw ] )

#                 #sb_replicate_wf = Workflow( [sb_replicate_fw],
#                 #    name="{:s}_sb_replicate".format(sb_name) )

#                 # sb_replicate_wf_list.append( sb_replicate_wf )

#         sb_name_fw_dict = dict( zip( sb_names, sb_replicate_fw_list ) )
#         #sb_name_wf_dict = dict(zip(sb_names,sb_replicate_wf_list))
#         return sb_name_fw_dict

# # generic packing
#     def prepare_packmol(self, system_name, pack_aggregates,
#                         nloop = None, maxit = None ):
#         """Creates preassembled aggregates on surface"""

#         #for system_name, row in sim_df.loc[system_names].iterrows():
#         row = self._sim_df.loc[system_name,:]
#         surfactant  = row["surfactant"]
#         counterion  = row["counterion"]
#         sfN         = row["sf_nmolecules"]

#         sb_name     = row["sb_name"]

#         # measures of substrate:
#         sb_measures = np.asarray(row["sb_measures"]) / C.angstrom

#         # standard settings can be overridden
#         packmol_script_writer_task_context = {
#             'header':        self.geninfo(),
#             'system_name':   system_name,
#             'sb_name':       sb_name,
#             'tolerance':     tolerance,
#             'write_restart': True
#         }

#         if nloop is not None:
#             packmol_script_writer_task_context['nloop'] = nloop

#         if maxit is not None:
#             packmol_script_writer_task_context['maxit'] = maxit

#         packmol_script_writer_task_context.update(
#             pack_aggregates(
#                 surfactant  = surfactant,
#                 counterion  = counterion,
#                 sfN         = sfN,
#                 sb_measures = sb_measures,
#             )
#         )

#         # packmol fill script template
#         packmol_fill_script_template_ft = TemplateWriterTask( {
#             'context' :      packmol_script_writer_task_context,
#             'template_file': self.template_prefix + os.sep + 'packmol.inp',
#             'output_file':   system_name + self._packmol_suffix + '.inp' } )

#         packmol_fill_script_template_fw = Firework(
#             packmol_fill_script_template_ft,
#             spec={
#                 "_category":   self.std_worker,
#                 "_dupefinder": DupeFinderExact(),
#                 '_files_out': {
#                     'packmol_inp': system_name + self._packmol_suffix + '.inp'
#                     },
#                 "system_name": system_name,
#                 "step"   :     "packmol_fill_script_template",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_packmol_fill_script_template".format(system_name) )

#     # packmol get building blocks
#         single_surfactant_pdb = '1_{:s}.pdb'.format(surfactant)
#         single_counterion_pdb = '1_{:s}.pdb'.format(counterion)

#         packmol_get_components_ft = FileTransferTask( {
#                 'files': [ {
#                     'src':  os.path.join(
#                         self.template_prefix, single_surfactant_pdb ),
#                     'dest': single_surfactant_pdb }, {
#                     'src':  os.path.join(
#                         self.template_prefix, single_counterion_pdb ),
#                     'dest': single_counterion_pdb } ],
#                 'mode': 'copy' } )

#         # load surface from data base
#         packmol_get_substrate_ft = GetFilesTask( {
#                 'identifiers': [ '{:s}.pdb'.format(sb_name) ],
#                 'new_file_names': [ '{:s}.pdb'.format(sb_name) ] } )

#     # packmol run
#         packmol_cmd = self._template_packmol_cmd.format(
#             infile= ( system_name + self._packmol_suffix + '.inp' ) )
#         packmol_ft =  ScriptTask.from_str( packmol_cmd,{
#             'stdout_file':  system_name + self._packmol_suffix + '.out',
#             'stderr_file':  system_name + self._packmol_suffix + '.err',
#             'use_shell':    True,
#             'fizzle_bad_rc':True } )

#         mkdir_ft =  ScriptTask.from_str(
#             'mkdir -p "{:s}"'.format(
#                 self.output_prefix + os.sep + system_name ),
#                 {
#                     'use_shell':    True, 'fizzle_bad_rc': False
#                 } )

#         copy_output_files_ft = FileTransferTask( {
#             'files': [
#                 system_name + self._packmol_suffix + '.inp',
#                 system_name + self._packmol_suffix + '.pdb',
#             ],
#             'dest': os.path.join(self.output_prefix,system_name),
#             'mode': 'copy' } )

#         packmol_fw = Firework(
#             [
#                 packmol_get_components_ft,
#                 packmol_get_substrate_ft,
#                 packmol_ft,
#                 mkdir_ft,
#                 copy_output_files_ft
#             ],
#             spec={
#                 "_queueadapter": self.packmol_queueadapter,
#                 "_category":   self.queue_worker_serial,
#                 "_dupefinder": DupeFinderExact(),
#                 "_files_in" : {
#                     "packmol_inp":  (system_name + self._packmol_suffix + '.inp') },
#                 "_files_out": {
#                     "packmol_pdb" : (system_name + self._packmol_suffix + '.pdb'),
#                     "packmol_pdb_FORCED" : (system_name + self._packmol_suffix + '.pdb_FORCED')
#                 },
#                 "system_name": system_name,
#                 "step"     :   "packmol",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name= system_name + self._packmol_suffix,
#             parents=[packmol_fill_script_template_fw] )

#         packmol_wf = Workflow(
#             [
#                 packmol_fill_script_template_fw,
#                 packmol_fw
#             ],
#             {
#                 packmol_fill_script_template_fw: packmol_fw
#             },
#             name = system_name + self._packmol_suffix )

#         return packmol_wf



# # convert pdb to lammps data format
#     def prepare_pdb2lmp(self,system_name):
#         """Processes _sim_df and constructs preassembled surfactant aggregates

#         _sim_df should have the columns
#           surfactant    -- the surfactant's name
#           sf_nmolecules -- the number of surfactant molecules in the system
#           box           -- measures of simulation box in SI units (m)
#           sb_name       -- substrate slab name
#           substrate     -- subsrtate material, i.e. AU
#           counterion    -- counterion, i.e. NA or BR
#           solvent       -- i.e. H2O or water
#           sf_preassembly-- i.e. hemycylinders
#           ci_preassembly-- i.e. at_polar_head
#           sb_crystal_plane i.e. 111

#         """

#         row = self._sim_df.loc[system_name,:]

#         sb_name    = row["sb_name"]

#         surfactant = row["surfactant"]
#         sfN        = row["sf_nmolecules"]

#         # make more flexible at some point
#         if surfactant == 'SDS': # surfactant is anionic
#             nanion = 0
#             ncation = sfN
#         else: # CTAB (surfactant is cationic)
#             nanion = sfN
#             ncation = 0

#         box_nanometer = np.asarray( row["box"] ) / C.nano
#         box_angstrom  = np.asarray( row["box"] ) / C.angstrom

#         consecutive_fw_list = []

#     # retrieve packmol output file, even if fizzled (probably due to walltime)

#         mkdir_ft =  ScriptTask.from_str(
#             'mkdir -p "{:s}"'.format(
#                 self.output_prefix + os.sep + system_name ),
#             {
#                 'use_shell':    True,
#                 'fizzle_bad_rc': False
#             } )

#         recover_packmol_ft = RecoverPackmolTask( {
#             'dest':      self.output_prefix + os.sep + system_name,
#             'recover':   True,
#             'glob_patterns': [
#                 '*_restart.pack',
#                 '*_packmol.pdb',
#                 '*_packmol.pdb_FORCED',
#                 '*_packmol.inp'
#             ],
#             'forward_glob_patterns': {
#                 "packmol_pdb" : ["*_packmol.pdb_FORCED", "*_packmol.pdb"]
#             }
#         } )

#         recover_packmol_fw = Firework( [ mkdir_ft, recover_packmol_ft ],
#             spec={
#                 '_category':              self.std_worker,
#                 '_dupefinder':            DupeFinderExact(),
#                 '_allow_fizzled_parents': True,
#                 '_files_in':  {
#                     'packmol_pdb' :       system_name + self._packmol_suffix + '.pdb',
#                     'packmol_pdb_FORCED': system_name + self._packmol_suffix + '.pdb_FORCED'
#                 },
#                 '_files_out': {
#                     'packmol_pdb' : system_name + self._packmol_suffix + '.pdb'
#                 },
#                 'system_name' :           system_name,
#                 'step':                   'recover_packmol',
#                 'geninfo':                self.geninfo()
#             },
#             name="{:s}_recover_packmol".format(system_name) )

#         consecutive_fw_list.append(recover_packmol_fw)

#     # pdb to gro
#         packmol2gmx_ft =  ScriptTask.from_str(
#             self._template_packmol2gmx_cmd.format(
#                 template_prefix = self.template_prefix,
#                 infile = system_name + self._packmol_suffix + '.pdb' ),
#             {
#                 'stdout_file':  system_name + self._packmol2gmx_suffix + '.out',
#                 'stderr_file':  system_name + self._packmol2gmx_suffix + '.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         packmol2gmx_fw = Firework(
#             packmol2gmx_ft,
#             spec = {
#                 "_category":   self.std_worker,
#                 #"_allow_fizzled_parents": True,
#                 "_dupefinder": DupeFinderExact(),
#                 "_files_in":   {
#                     "packmol_pdb" : system_name + self._packmol_suffix + '.pdb'
#                     },
#                 "_files_out":  { "pdb_for_gmx":  system_name + '.pdb' },
#                 "system_name" : system_name,
#                 "step"   :     "packmol2gmx",
#                 'geninfo':     self.geninfo()
#                 },
#             name=(system_name + self._packmol2gmx_suffix) ,
#             parents=[recover_packmol_fw]
#         )
#         consecutive_fw_list.append(packmol2gmx_fw)

#     # solvate in gromacs:
#         gmx_solvate_fill_script_template_ft = TemplateWriterTask( {
#             'context': {
#                 'header':     self.geninfo(),
#                 'system_name':system_name,
#                 'surfactant': surfactant,
#                 'ncation':    ncation,
#                 'nanion':     nanion,
#                 'box':        box_nanometer,
#                 'ionize':     False
#             },
#             'template_file': self.template_prefix + os.sep + 'gmx_solvate.sh',
#             'output_file':   system_name + self._gmx_solvate_suffix + '.sh'} )

#         single_surfactant_pdb = '1_{:s}.pdb'.format( surfactant )

#         gmx_solvate_get_files_ft = FileTransferTask( {
#                 'files': [ {
#                     'src':  self.template_prefix + os.sep + single_surfactant_pdb,
#                     'dest': single_surfactant_pdb } ],
#                 'mode': 'copy' } )

#         gmx_solvate_ft =  ScriptTask.from_str(
#             self._template_gmx2pdb_cmd.format(
#                 infile = system_name + self._gmx_solvate_suffix + '.sh' ),
#             {
#                 'stdout_file':  system_name + self._gmx_solvate_suffix + '.out',
#                 'stderr_file':  system_name + self._gmx_solvate_suffix + '.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             } )
#         gmx_solvate_fw = Firework(
#             [
#                 gmx_solvate_fill_script_template_ft,
#                 gmx_solvate_get_files_ft,
#                 gmx_solvate_ft
#             ],
#             spec={
#                 "_category":  self.queue_worker_serial,
#                 "_dupefinder": DupeFinderExact(),
#                 "_files_in":   { "pdb_for_gmx":  (system_name + '.pdb') },
#                 "_files_out":  {
#                     "ionized_gro" : "{:s}_solvated.gro".format(system_name)
#                     },
#                 "system_name": system_name,
#                 "step"   :     "gmx_solvate",
#                 'geninfo':     self.geninfo()
#             },
#             name="{:s}_gmx_solvate".format(system_name),
#             parents=[packmol2gmx_fw])

#         consecutive_fw_list.append(gmx_solvate_fw)

#         # the system has already been ionized in packmol
#         # thus, this step is skipped in GROMACS
#         #copy_outfile_ft = FileTransferTask( {
#         #    'files': [ {
#         #        'src':  absolute_dir + os.sep + system_name + '_solvated.gro',
#         #        'dest': absolute_dir + os.sep + system_name + '_ionized.gro' } ],
#         #    'mode': 'copy' } )
#         #copy_outfile_fw = Firework(copy_outfile_ft,
#         #    spec={
#         #        "_dupefinder": DupeFinderExact(),
#         #        "system_name" :     system_name,
#         #        "step"   :     "gmx_solvate_copy_outfile"
#         #    },
#         #    name="{:s}_copy_outfile".format(system_name))

#     # convert to pdb chunks again
#         pdb_segment_chunk_glob_pattern = '*_[0-9][0-9][0-9].pdb'

#         gmx2pdb_fill_script_template_ft = TemplateWriterTask( {
#             'context': {
#                 'header':       self.geninfo(),
#                 'system_name':  system_name,
#             },
#             'template_file': self.template_prefix + os.sep + 'gmx2pdb.sh',
#             'output_file': system_name + self._gmx2pdb_suffix + '.sh'} )

#         gmx2pdb_ft =  ScriptTask.from_str(
#             self._template_gmx2pdb_cmd.format(
#                 infile = system_name + self._gmx2pdb_suffix + '.sh' ),
#             {
#                 'stdout_file':  system_name + self._gmx2pdb_suffix + '.out',
#                 'stderr_file':  system_name + self._gmx2pdb_suffix + '.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         gmx2pdb_tar_ft = ScriptTask.from_str(
#             'tar -czf {:s} {:s}'.format(
#                 system_name + '_segments.tar.gz',
#                 pdb_segment_chunk_glob_pattern ),
#             {
#                 'stdout_file':  system_name + self._gmx2pdb_suffix + '_tar.out',
#                 'stderr_file':  system_name + self._gmx2pdb_suffix + '_tar.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         mkdir_ft =  ScriptTask.from_str(
#             'mkdir -p "{:s}"'.format(
#                 self.output_prefix + os.sep + system_name ),
#                 {
#                     'use_shell':    True, 'fizzle_bad_rc': False
#                 } )

#         copy_output_files_ft = FileTransferTask( {
#             'files': [
#                 system_name + '_ionized.gro',
#                 system_name + '_segments.tar.gz',
#             ],
#             'dest': self.output_prefix + os.sep + system_name + os.sep,
#             'mode': 'copy' } )

#         gmx2pdb_fw = Firework(
#             [
#                 gmx2pdb_fill_script_template_ft,
#                 gmx2pdb_ft,
#                 gmx2pdb_tar_ft,
#                 mkdir_ft,
#                 copy_output_files_ft,

#             ],
#             spec={
#                 "_category":   self.std_worker,
#                 "_dupefinder": DupeFinderExact(),
#                 "_files_in":   {
#                     "ionized_gro" : "{:s}_ionized.gro".format(system_name)
#                 },
#                 "_files_out":  {
#                     "segments_tar":  "{:s}_segments.tar.gz".format(
#                         system_name )
#                 },
#                 "system_name" :system_name,
#                 "step"   :     "gmx2pdb",
#                 'geninfo':     self.geninfo()
#             },
#             name="{:s}_gmx2pdb".format(system_name),
#             parents=[gmx_solvate_fw])

#         consecutive_fw_list.append(gmx2pdb_fw)

#     ### PSFGEN

#     # make psfgen input
#         psfgen_untar_tf = ScriptTask.from_str('tar -xf {:s}'.format(
#             system_name + '_segments.tar.gz'),
#             {
#                 'stdout_file':  system_name + self._psfgen_suffix + '_untar.out',
#                 'stderr_file':  system_name + self._psfgen_suffix + '_untar.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         makeIdPdbDict_ft = MakeSegIdSegPdbDictTask( {
#             'glob_pattern': pdb_segment_chunk_glob_pattern } )

#         # in order to make FWAction be able to alter context content,
#         # the following TemplateWriterTask uses global specs from the
#         # encapsulating Firework
#         psfgen_fill_script_template_ft = TemplateWriterTask( {
#             'use_global_spec' : True} )
#         #psfgen_fill_script_template_fw = Firework( psfgen_fill_script_template_ft,
#         #    spec={
#         #        "_dupefinder": DupeFinderExact(),
#         #        "system_name"     : system_name,
#         #        "step"       : "psfgen_fill_script_template",
#         #    },
#         #    name="{:s}_psfgen_fill_script_template".format(system_name))

#     # get necessary files
#         psfgen_get_files_ft = FileTransferTask( {
#                 'files': [ {
#                     'src':  self.template_prefix \
#                         + os.sep + 'par_all36_lipid_extended_stripped.prm',
#                     'dest': 'par_all36_lipid_extended_stripped.prm' }, {
#                     'src':  self.template_prefix + os.sep \
#                         + 'top_all36_lipid_extended_stripped.rtf',
#                     'dest': 'top_all36_lipid_extended_stripped.rtf' } ],
#                 'mode': 'copy' } )

#     # run psfgen
#         psfgen_ft =  ScriptTask.from_str(
#             self._template_psfgen_cmd.format( infile = (
#                 system_name + self._psfgen_suffix + '.pgn')),
#             {
#                 'stdout_file':  system_name + self._psfgen_suffix + '.out',
#                 'stderr_file':  system_name + self._psfgen_suffix + '.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         psfgen_fw = Firework(
#             [
#                 psfgen_untar_tf,
#                 makeIdPdbDict_ft,
#                 psfgen_fill_script_template_ft,
#                 psfgen_get_files_ft,
#                 psfgen_ft
#             ],
#             spec={
#                 "_category":  self.std_worker,
#                 "_dupefinder": DupeFinderExact(),
#                 "_files_in":  {
#                     "segments_tar": "{:s}_segments.tar.gz".format(
#                         system_name ) },
#                 "_files_out": {
#                     "psfgen_pdb": "{:s}_psfgen.pdb".format(system_name),
#                     "psfgen_psf":    "{:s}_psfgen.psf".format(system_name)
#                 },
#                 "system_name": system_name,
#                 "step"       : "psfgen",
#                 # for the template writer task
#                 'context': {
#                     'header':      self.geninfo(),
#                     'system_name': system_name
#                 },
#                 'template_file': self.template_prefix + os.sep + 'psfgen.pgn',
#                 'output_file':   system_name + self._psfgen_suffix + '.pgn',
#                 'geninfo':       self.geninfo()
#             },
#             name = system_name +self._psfgen_suffix,
#             parents=[gmx2pdb_fw])

#         consecutive_fw_list.append(psfgen_fw)

#     # run charmm2lammps
#         ch2lmp_get_files_ft = FileTransferTask( {
#                 'files': [ {
#                     'src':  self.template_prefix + os.sep \
#                         + 'par_all36_lipid_extended_stripped.prm',
#                     'dest': 'par_all36_lipid_extended_stripped.prm' }, {
#                     'src':  self.template_prefix + os.sep \
#                         + 'top_all36_lipid_extended_stripped.rtf',
#                     'dest': 'top_all36_lipid_extended_stripped.rtf' } ],
#                 'mode': 'copy' } )

#         ch2lmp_ft =  ScriptTask.from_str(
#                 self._template_ch2lmp_cmd.format(
#                     system_name = system_name, box = box_angstrom), {
#                 'stdout_file':  system_name + self._ch2lmp_suffix + '.out',
#                 'stderr_file':  system_name + self._ch2lmp_suffix + '.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True } )

#         # If existent, just deletes existing files
#         file_identifier = "{:s}_psfgen.data".format(system_name)

#         ch2lmp_delete_ft = DeleteFilesTask(
#             {
#                 'identifiers': [file_identifier]
#             }
#         )

#         ch2lmp_store_ft =  AddFilesTask(
#             {
#                 'paths':       ["{:s}_psfgen.data".format(system_name)],
#                 'identifiers': [ file_identifier ],
#                 'metadata': {
#                     'system_name':          system_name,
#                     'sb_name':              sb_name,
#                     'surfactant':           row["surfactant"],
#                     'substrate':            row["substrate"],
#                     'counterion':           row["counterion"],
#                     'solvent':              row["solvent"],
#                     'sf_preassembly':       row["sf_preassembly"],
#                     'ci_preassembly':       row["ci_initial_placement"],
#                     'sb_crystal_plane':     row["sb_crystal_plane"]
#                 }
#             }
#         )

#         copy_output_files_ft = FileTransferTask( {
#             'files': [
#                 "{:s}_psfgen.data".format(system_name),
#                 "{:s}_psfgen.in".format(system_name),
#                 "{:s}_psfgen_ctrl.pdb".format(system_name),
#                 "{:s}_psfgen_ctrl.psf".format(system_name)
#             ],
#             'dest': self.output_prefix + os.sep + system_name,
#             'mode': 'copy' } )

#         ch2lmp_fw = Firework(
#             [
#                 ch2lmp_get_files_ft,
#                 ch2lmp_ft,
#                 ch2lmp_delete_ft,
#                 ch2lmp_store_ft,
#                 mkdir_ft,
#                 copy_output_files_ft
#             ],
#             spec={
#                 "_category":   self.std_worker,
#                 "_dupefinder": DupeFinderExact(),
#                 "_files_in": {
#                     "psfgen_pdb":    "{:s}_psfgen.pdb".format(system_name),
#                     "psfgen_psf":    "{:s}_psfgen.psf".format(system_name) },
#                 "_files_out": {
#                     "ch2lmp_data":      "{:s}_psfgen.data".format(system_name),
#                     "ch2lmp_in":        "{:s}_psfgen.in".format(system_name),
#                     "ch2lmp_ctrl_pdb":  "{:s}_psfgen_ctrl.pdb".format(system_name),
#                     "ch2lmp_ctrl_psf":  "{:s}_psfgen_ctrl.psf".format(system_name)
#                 },
#                 "system_name": system_name,
#                 "step"       : "ch2lmp",
#                 'geninfo':     self.geninfo()
#             },
#             name= system_name + self._ch2lmp_suffix,
#             parents=[psfgen_fw])

#         consecutive_fw_list.append(ch2lmp_fw)

#         # workflow
#         parent_links = { consecutive_fw_list[i] : consecutive_fw_list[i+1] \
#                         for i in range(len(consecutive_fw_list)-1) }

#         return Workflow( consecutive_fw_list, parent_links,
#             name="{:s}_prep_wf".format(system_name) )


# # MD in LAMMPS (originally on JUWELS, to be tested)
#     def prepare_systems(  self, system_name ):
#         """Initiates workflows for all systems in system_names"""

#         # prepare_system_files_fw_list = []
#         # for system_name in system_names:
#         # A LAMMPS data file file suffixed _psfgen.data
#         # is expected to exist in the FilePad
#         data_file = '{:s}_psfgen.data'.format(system_name)

#         # standard LAMMPS input files lmp_header.input and
#         # lmp_minimization.input are expected to reside within
#         # template _prefix
#         #get_input_files_ft = FileTransferTask( {
#         #    'files': [
#         #        {
#         #            'src':  self.template_prefix + os.sep + 'lmp_header.input',
#         #            'dest': 'lmp_header.input'
#         #        }, {
#         #            'src':  self.template_prefix + os.sep + 'lmp_minimization.input',
#         #            'dest': 'lmp_minimization.input'
#         #        }
#         #    ],
#         #    'mode': 'copy' } )

#         get_data_file_ft = GetFilesTask( {
#                 'identifiers': [ data_file ],
#                 'new_file_names': [ data_file ] } )

#         # write a tiny one-line file to tell lammps the system name
#         # might be changed to -v baseName at LAMMPS command line instead
#         write_system_specific_input_ft = FileWriteTask( {
#             'files_to_write': [
#                 {
#                     'filename': 'system_specific.input',
#                     'contents':  'variable baseName string {:s}'.format(
#                         system_name )
#                 }
#             ] } )

#         # creates output directory if necessary
#         # better replace by in-python task at some point
#         mkdir_ft =  ScriptTask.from_str(
#             'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
#                 {
#                     'use_shell':    True, 'fizzle_bad_rc': False
#                 } )

#         copy_output_files_ft = FileTransferTask( {
#             'files': [
#                 data_file
#             ],
#             'dest': self.output_prefix + os.sep + system_name + os.sep,
#             'mode': 'copy' } )

#         prepare_system_files_fw = Firework(
#             [
#                 #get_input_files_ft,
#                 get_data_file_ft,
#                 write_system_specific_input_ft,
#                 mkdir_ft,
#                 copy_output_files_ft
#             ],
#             spec={
#                 "_category":   self.std_worker,
#                 "_dupefinder": DupeFinderExact(),
#                 "_files_out":  {
#                     "data_file"   :         data_file,
#                     "system_specific_file": 'system_specific.input'
#                 },
#                 "system_name": system_name,
#                 "step"     :   "prepare_system_files",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_prepare_system_files".format(system_name))

#         return prepare_system_files_fw
#         # prepare_system_files_fw_list.append((system_name,prepare_system_files_fw))

#         #prepare_system_files_fw_dict = dict(prepare_system_files_fw_list)
#         # returns a system_name : Firework dict
#         #return prepare_system_files_fw_dict

#     def prepare_minimizations( self, system_names, robust = False ):
#         """Prepares LAMMPS minimizations.

#         Paramters
#         ---------
#         system_names: list of str
#             Subset of systems to process.
#         robust: bool
#             Passes additional command line switch
#             '-v robust_minimization 1' to LAMMPS executable,
#             causing lmp_minimization.input issues a
#             "neigh_modify delay 0 every 1 check yes one 5000 page 250000"
#             statement. See LAMMPS documentation for more info.
#         """

#         lmp_cmd_suffix = ' -v robust_minimization 1' if robust else  ''

#         minimization_fw_list = []
#         for system_name in system_names:
#             get_input_files_ft = FileTransferTask( {
#                 'files': [
#                     {
#                         'src':  self.template_prefix + os.sep + 'lmp_header.input',
#                         'dest': 'lmp_header.input'
#                     }, {
#                         'src':  self.template_prefix + os.sep + 'lmp_minimization.input',
#                         'dest': 'lmp_minimization.input'
#                     }
#                 ],
#                 'mode': 'copy' } )

#             minimization_ft =  ScriptTask.from_str(
#                 self.lmp_cmd.format(
#                     inputFile = 'lmp_minimization.input' ) + lmp_cmd_suffix,
#                 {
#                     'stdout_file':  system_name + '_minimization.out',
#                     'stderr_file':  system_name + '_minimization.err',
#                     'use_shell':    True,
#                     'fizzle_bad_rc':True
#                 })

#             extract_thermo_ft =  ScriptTask.from_str(
#                 self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
#                     system_name + '_minimization.log',
#                     system_name + '_minimization_thermo.out'),
#                     {
#                         'use_shell':    True, 'fizzle_bad_rc': False
#                     } )

#             mkdir_ft =  ScriptTask.from_str(
#                 'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
#                     {
#                         'use_shell':    True, 'fizzle_bad_rc': False
#                     } )

#             copy_output_files_ft = FileTransferTask( {
#                 'files': [
#                     system_name + '_minimized.lammps',
#                     system_name + '_minimization.log',
#                     system_name + '_minimization_thermo.out'
#                 ],
#                 'dest': self.output_prefix + os.sep + system_name,
#                 'mode': 'copy' } )

#             minimization_fw = Firework(
#                 [
#                     get_input_files_ft,
#                     minimization_ft,
#                     extract_thermo_ft,
#                     mkdir_ft,
#                     copy_output_files_ft
#                 ],
#                 spec={
#                     "_queueadapter": self.minimization_queueadapter,
#                     "_category":     self.queue_worker,
#                     "_dupefinder":   DupeFinderExact(),
#                     "_files_in":  {
#                         "data_file"   :         system_name + '_psfgen.data',
#                         "system_specific_file": 'system_specific.input'
#                     },
#                     "_files_out":  {
#                         "data_file"   :         system_name + '_minimized.lammps',
#                         "system_specific_file": 'system_specific.input'
#                     },
#                     "system_name": system_name,
#                     "step"     :   "minimization",
#                     "geninfo":     self.geninfo() # serves to distinguish duplicates
#                 },
#                 name="{:s}_minimzation".format(system_name))

#             minimization_fw_list.append((system_name,minimization_fw))

#         system_name_minimization_fw_dict = dict(minimization_fw_list)
#         return system_name_minimization_fw_dict

#     def prepare_nvt_equlibrations( self, system_names ):
#         equilibration_nvt_fw_list = []
#         for system_name in system_names:
#             get_input_files_ft = FileTransferTask( {
#                 'files': [
#                     {
#                         'src':  self.template_prefix + os.sep + 'lmp_header.input',
#                         'dest': 'lmp_header.input'
#                     }, {
#                         'src':  self.template_prefix + os.sep + 'lmp_equilibration_nvt.input',
#                         'dest': 'lmp_equilibration_nvt.input'
#                     }
#                 ],
#                 'mode': 'copy' } )

#             # for now, switch mpiio off with mechanism in lammps input script
#             equilibration_nvt_ft =  ScriptTask.from_str(
#                 self.lmp_cmd.format(inputFile = 'lmp_equilibration_nvt.input'),
#                 {
#                     'stdout_file':  system_name + '_equilibration_nvt.out',
#                     'stderr_file':  system_name + '_equilibration_nvt.err',
#                     'use_shell':    True,
#                     'fizzle_bad_rc':True
#                 })

#             extract_thermo_ft =  ScriptTask.from_str(
#                 self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
#                     system_name + '_nvtEquilibration.log',
#                     system_name + '_nvtEquilibration_thermo.out'),
#                     {
#                         'use_shell':    True, 'fizzle_bad_rc': False
#                     } )

#             mkdir_ft =  ScriptTask.from_str(
#                 'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
#                     {
#                         'use_shell':    True, 'fizzle_bad_rc': False
#                     } )

#             copy_output_files_ft = FileTransferTask( {
#                 'files': [
#                     system_name + '_nvtEquilibrated.lammps',
#                     system_name + '_nvtEquilibration.log',
#                     system_name + '_nvtEquilibration_thermo.out'
#                 ],
#                 'dest': self.output_prefix + os.sep + system_name + os.sep,
#                 'mode': 'copy' } )

#             equilibration_nvt_fw = Firework(
#                 [
#                     get_input_files_ft,
#                     equilibration_nvt_ft,
#                     extract_thermo_ft,
#                     mkdir_ft,
#                     copy_output_files_ft
#                 ],
#                 spec={
#                     "_queueadapter": self.equilibration_queueadapter,
#                     "_category":   self.queue_worker,
#                     "_dupefinder": DupeFinderExact(),
#                     "_files_in":  {
#                         "data_file"   :         system_name + '_minimized.lammps',
#                         "system_specific_file": 'system_specific.input'
#                     },
#                     "_files_out":  {
#                         "data_file"   :         system_name + '_nvtEquilibrated.lammps',
#                         "system_specific_file": 'system_specific.input'
#                     },
#                     "system_name": system_name,
#                     "step"     :   "equilibration_nvt",
#                     "geninfo":     self.geninfo() # serves to distinguish duplicates
#                 },
#                 name="{:s}_equilibration_nvt".format(system_name))

#             equilibration_nvt_fw_list.append((system_name,equilibration_nvt_fw))

#         system_name_equilibration_nvt_fw_dict = dict(equilibration_nvt_fw_list)
#         return system_name_equilibration_nvt_fw_dict

#     def prepare_npt_equlibrations( self, system_names ):
#         equilibration_npt_fw_list = []
#         for system_name in system_names:
#             get_input_files_ft = FileTransferTask( {
#                 'files': [
#                     {
#                         'src':  self.template_prefix + os.sep + 'lmp_header.input',
#                         'dest': 'lmp_header.input'
#                     }, {
#                         'src':  self.template_prefix + os.sep + 'lmp_equilibration_npt.input',
#                         'dest': 'lmp_equilibration_npt.input'
#                     }
#                 ],
#                 'mode': 'copy' } )

#             # for now, switxch mpiio off with mechanism in lammps input script
#             equilibration_npt_ft =  ScriptTask.from_str(
#                 self.lmp_cmd.format(inputFile='lmp_equilibration_npt.input'),
#                 {
#                     'stdout_file':  system_name + '_equilibration_npt.out',
#                     'stderr_file':  system_name + '_equilibration_npt.err',
#                     'use_shell':    True,
#                     'fizzle_bad_rc':True
#                 })

#             extract_thermo_ft =  ScriptTask.from_str(
#                 self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
#                     system_name + '_nptEquilibration.log',
#                     system_name + '_nptEquilibration_thermo.out'),
#                     {
#                         'use_shell':    True, 'fizzle_bad_rc': False
#                     } )

#             mkdir_ft =  ScriptTask.from_str(
#                 'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
#                     {
#                         'use_shell':    True, 'fizzle_bad_rc': False
#                     } )

#             copy_output_files_ft = FileTransferTask( {
#                 'files': [
#                     system_name + '_nptEquilibrated.lammps',
#                     system_name + '_nptEquilibration.log',
#                     system_name + '_nptEquilibration_thermo.out'
#                 ],
#                 'dest': self.output_prefix + os.sep + system_name + os.sep,
#                 'mode': 'copy' } )

#             equilibration_npt_fw = Firework(
#                 [
#                     get_input_files_ft,
#                     equilibration_npt_ft,
#                     extract_thermo_ft,
#                     mkdir_ft,
#                     copy_output_files_ft
#                 ],
#                 spec={
#                     "_queueadapter": self.equilibration_queueadapter,
#                     "_category":   self.queue_worker,
#                     "_dupefinder": DupeFinderExact(),
#                     "_files_in":  {
#                         "data_file"   :         system_name + '_nvtEquilibrated.lammps',
#                         "system_specific_file": 'system_specific.input'
#                     },
#                     "_files_out":  {
#                         "data_file"   :         system_name + '_nptEquilibrated.lammps',
#                         "system_specific_file": 'system_specific.input'
#                     },
#                     "system_name": system_name,
#                     "step"     :   "equilibration_npt",
#                     "geninfo":     self.geninfo() # serves to distinguish duplicates
#                 },
#                 name="{:s}_equilibration_npt".format(system_name))

#             equilibration_npt_fw_list.append((system_name,equilibration_npt_fw))

#         system_name_equilibration_npt_fw_dict = dict(equilibration_npt_fw_list)
#         return system_name_equilibration_npt_fw_dict

#     #
#     def prepare_production( self, system_names ):
#         production_fw_list = []
#         for system_name in system_names:
#             get_input_files_ft = FileTransferTask( {
#                 'files': [
#                     {
#                         'src':  self.template_prefix + os.sep + 'lmp_header.input',
#                         'dest': 'lmp_header.input'
#                     }, {
#                         'src':  self.template_prefix + os.sep + 'lmp_production_mixed.input',
#                         'dest': 'lmp_production_mixed.input'
#                     }, {
#                         'src':  self.template_prefix + os.sep + 'lmp_10ns_production_mixed.input',
#                         'dest': 'lmp_10ns_production_mixed.input'
#                     }
#                 ],
#                 'mode': 'copy' } )


#             lmp_cmd = ' '.join((
#                 self.lmp_cmd.format(inputFile='lmp_10ns_production_mixed.input'),
#                 '-v baseName {baseName:s}'.format(baseName=system_name) ))

#             production_ft =  ScriptTask.from_str(
#                 lmp_cmd,
#                 {
#                     'stdout_file':  system_name + '_10ns_production.out',
#                     'stderr_file':  system_name + '_10ns_production.err',
#                     'use_shell':    True,
#                     'fizzle_bad_rc':True
#                 })

#             extract_thermo_ft =  ScriptTask.from_str(
#                 self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
#                     system_name + '_10ns_production_mixed.log',
#                     system_name + '_10ns_production_mixed_thermo.out'),
#                     {
#                         'use_shell':    True, 'fizzle_bad_rc': False
#                     } )

#             mkdir_ft =  ScriptTask.from_str(
#                 'mkdir -p "{:s}"'.format(self.output_prefix + os.sep + system_name),
#                     {
#                         'use_shell':    True, 'fizzle_bad_rc': False
#                     } )

#             copy_output_files_ft = FileTransferTask( {
#                 'files': [
#                     system_name + '_production_mixed.lammps',
#                     system_name + '_10ns_production_mixed.log',
#                     system_name + '_10ns_production_mixed_thermo.out'
#                 ],
#                 'dest': self.output_prefix + os.sep + system_name + os.sep,
#                 'mode': 'copy' } )

#             production_fw = Firework(
#                 [
#                     get_input_files_ft,
#                     production_ft,
#                     extract_thermo_ft,
#                     mkdir_ft,
#                     copy_output_files_ft
#                 ],
#                 spec={
#                     "_queueadapter": self.production_queueadapter,
#                     "_category":   self.queue_worker,
#                     "_dupefinder": DupeFinderExact(),
#                     "_files_in":  {
#                         "data_file"   :         system_name + '_nptEquilibrated.lammps',
#                     },
#                     "_files_out":  {
#                         "data_file"   :         system_name + '_production_mixed.lammps',
#                     },
#                     "system_name": system_name,
#                     "step"     :   "10ns_production_mixed",
#                     "geninfo":     self.geninfo() # serves to distinguish duplicates
#                 },
#                 name="{:s}_10ns_production_mixed".format(system_name))

#             production_fw_list.append((system_name,production_fw))

#         system_name_production_fw_dict = dict(production_fw_list)
#         return system_name_production_fw_dict

#     # tuned to continue runs by self.colvars_production or self.production
#     def prepare_restart( self, system_name,
#         lmp_suffix_template = ' '.join(('-v baseName {baseName:s}',
#             '-v has_indenter 1 -v pbc2d 0 -v mpiio 0 -v use_colvars 1')),
#         output_subfolder = 'production' ):
#         #restart_fw_list = []
#         #for system_name in system_names:
#         #lmp_cmd = ' '.join((self._template_lmp_cmd.format(
#         #    inputFile = 'lmp_production.input' ), lmp_suffix_template.format(
#         #        baseName=system_name, dataFile='datafile.lammps' ) ))

#         recover_lammps_ft = RecoverLammpsTask( {
#             'restart_file_glob_pattern': ['*.restart[0-9]'],
#             'fizzle_on_no_restart_file': True, # only if restart neccessary
#             'dest': self.output_prefix + os.sep + system_name + \
#                 os.sep + output_subfolder,
#             'other_glob_patterns': [
#                 '*.nc', # netcdf trajectory
#                 '*.log', # LAMMPS logfile
#                 '*.colvars.traj', # colvars outputs
#                 '*.colvars.state',
#                 '*.ti.count',
#                 '*.ti.grad',
#                 '*.ti.pmf' ],
#             'file_transfer_mode': 'copy',
#             'continue': True,
#             'recover':  True,
#             'lmp_queueadapter': self.production_queueadapter,
#             'lmp_cmd': self._template_lmp_cmd.format(
#                 inputFile='lmp_production.input'),
#             'lmp_opt': lmp_suffix_template
#         } )

#         recover_lammps_fw = Firework( recover_lammps_ft,
#             spec = {
#                 '_category':                   self.std_worker,
#                 '_allow_fizzled_parents':      True,
#                 'system_name':                 system_name,
#                 'step':                        'recover_lammps',
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#                 },
#             name    = '{:s}_recover_lammps'.format(system_name) )

#         #restart_fw_list.append((system_name,recover_lammps_fw))

#         #system_name_restart_fw_dict = dict(restart_fw_list)
#         #return system_name_restart_fw_dict
#         return recover_lammps_fw

#     def extract_frame( self, system_name,
#         topology_file_name = None, trajectory_file_name = None,
#         frame = 0, netcdf2data_cmd_suffix = '', additional_spec = {} ):

#         # if no topology file specified, pull from previous step
#         if topology_file_name is None:
#             topology_file_name = 'datafile.lammps'
#             additional_spec["_files_in"] = {"data_file": 'datafile.lammps'}

#         netcdf2data_cmd_suffix += '--frames {frame:d}'

#         # assumes output pattern 'frame_{:d}.lammps'
#         netcdf2data_cmd = ' '.join((
#             self._template_netcdf2data_cmd.format(
#                 datafile = topology_file_name, trajfile = trajectory_file_name),
#             netcdf2data_cmd_suffix.format( frame=int(frame) ) ))

#         netcdf2data_ft =  ScriptTask.from_str(
#             netcdf2data_cmd,
#             {
#                 'stdout_file':  'netcdf2data.out',
#                 'stderr_file':  'netcdf2data.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         # TODO: correct fixed file name
#         fw_spec = {
#             "_category":   self.std_worker,
#             "_dupefinder": DupeFinderExact(),
#             "_files_out": {
#                 "data_file": "frame_{:08d}.lammps".format(frame)
#             },
#             "system_name": system_name,
#             "step"     :   "netcdf2frame",
#             "geninfo":     self.geninfo() # serves to distinguish duplicates
#         }

#         if len(additional_spec) > 0:
#             fw_spec.update( additional_spec )

#         netcdf2data_fw = Firework(
#             [ netcdf2data_ft ], spec = fw_spec,
#             name= system_name + '_netcdf2frame_{:d}'.format(frame) )

#         return netcdf2data_fw


#     def store_data_file(
#         self, system_name, file_identifier = None, file_name = None,
#         additional_spec = {} ):

#         if file_identifier is None:
#             file_identifier = "{:s}_10ns.lammps".format(system_name)
#         if file_name is None:
#             file_name = file_identifier

#         # if existent, just deletes existing file
#         delete_ft = DeleteFilesTask(
#             {
#                 'identifiers': [file_identifier]
#             }
#         )

#         store_ft =  AddFilesTask(
#             {
#                 'paths':       [ file_name ],
#                 'identifiers': [ file_identifier ],
#                 'metadata': {
#                     'system_name':          system_name,
#                     # 'sb_name':              row["sb_name"],
#                     # 'surfactant':           row["surfactant"],
#                     # 'substrate':            row["substrate"],
#                     # 'counterion':           row["counterion"],
#                     # 'solvent':              row["solvent"],
#                     # 'sf_preassembly':       row["sf_preassembly"],
#                     # 'ci_preassembly':       row["ci_initial_placement"],
#                     # 'sb_crystal_plane':     row["sb_crystal_plane"],
#                 }
#             }
#         )

#         # TODO: correct fixed file name
#         fw_spec = {
#             "_category":     self.std_worker,
#             "_dupefinder":   DupeFinderExact(),
#             "_files_in": {
#                 "data_file": file_name
#             },
#             "system_name": system_name,
#             "step"     :   "store_data_file",
#             "geninfo":     self.geninfo() # serves to distinguish duplicates
#         }

#         if len(additional_spec) > 0:
#             fw_spec.update( additional_spec )


#         fw = Firework(
#             [
#                 delete_ft,
#                 store_ft
#             ],
#             spec = fw_spec,
#             name="{:s}_store_data_file".format(system_name) )
#         return fw

#     def pull_datafile_from_db(
#         self, system_name, data_file_identifier = None, data_file_identifier_suffix = '_initial.lammps', fworker = None ):

#         if data_file_identifier is None:
#             data_file_identifier = system_name + data_file_identifier_suffix

#         if fworker is None: fworker = self.std_worker

#         # read from data base
#         get_data_files_ft = GetFilesTask( {
#                 'identifiers': [ data_file_identifier ],
#                 'new_file_names': [ 'datafile.lammps' ] } )

#         fw = Firework(
#             [ get_data_files_ft ],
#             spec={
#                 "_category":     fworker,
#                 #"_dupefinder":   DupeFinderExact(),
#                 "_files_out":  {
#                     "data_file": 'datafile.lammps'
#                 },
#                 "system_name": system_name,
#                 "step"     :   "pull_datafile_from_db",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_pull_datafile_from_db".format(system_name))
#         return fw

#     def initiate_indenter_workflow(
#         self, system_name, interface_file, indenter_file):

#         # read from data base
#         get_data_files_ft = GetFilesTask( {
#                 'identifiers': [ indenter_file, interface_file ],
#                 'new_file_names': [ indenter_file, interface_file ] } )

#         fw = Firework(
#             [ get_data_files_ft ],
#             spec={
#                 "_category":     self.std_worker,
#                 "_dupefinder":   DupeFinderExact(),
#                 "_files_out":  {
#                     "interface_file": interface_file,
#                     "indenter_file":  indenter_file
#                 },
#                 "system_name": system_name,
#                 "step"     :   "initiate_indenter_workflow",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_initiate_indenter_workflow".format(system_name))

#         return fw

#     def indenter_insertion(
#         self, system_name, fworker = None, desired_distance = 100.0 ):
#         """Insert indenter.

#         Paramters
#         ---------
#         system_name: str
#         """
#         if fworker is None: fworker = self.std_worker

#         row = self._sim_df.loc[system_name,:]
#         surfactant  = row["surfactant"]

#         interface_file = 'interface.lammps'
#         indenter_file =  'indenter.pdb'

#         get_input_files_ft = GetFilesTask( {
#              'identifiers': [ 'jlh_vmd.tcl', 'indenter_insertion.tcl' ],
#              'new_file_names': [ 'jlh_vmd.tcl', 'indenter_insertion.template' ] } )

#         #get_input_files_ft = FileTransferTask( {
#         #    'files': [
#         #        {
#         #            'src':  self.template_prefix + os.sep + 'jlh_vmd.tcl',
#         #            'dest': 'jlh_vmd.tcl'
#         #        }
#         #    ],
#         #    'mode': 'copy' } )

#         indenter_insertion_fill_script_template_ft = TemplateWriterTask( {
#             'context' :      {
#                 'header':           self.geninfo(),
#                 'surfactant':       surfactant,
#                 'desired_distance': desired_distance,
#                 'interface_file':   interface_file,
#                 'indenter_file':    indenter_file,
#                 'output_prefix':    system_name
#             },
#             'template_file': 'indenter_insertion.template',
#             'output_file':   'indenter_insertion.tcl' } )

#         indenter_insertion_ft =  ScriptTask.from_str(
#             self._template_vmd_cmd.format(
#                 infile = 'indenter_insertion.tcl' ),
#             {
#                 'stdout_file':  'indenter_insertion.out',
#                 'stderr_file':  'indenter_insertion.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         convert_ft =  ScriptTask.from_str(
#             self._template_convert_cmd.format(
#                 infile  = system_name + '.tga',
#                 outfile = system_name + '.png'),
#                 {
#                     'use_shell':     True,
#                     'fizzle_bad_rc': False
#                 } )

#         #dest = self.output_prefix + os.sep + system_name + os.sep + 'indenter_immersed'
#         #mkdir_ft =  ScriptTask.from_str(
#         #    'mkdir -p "{:s}"'.format(dest),
#         #        {
#         #            'use_shell':     True,
#         #            'fizzle_bad_rc': False
#         #        } )

#         #copy_output_files_ft = FileTransferTask( {
#         #    'files': [
#         #        system_name + '.lammps',
#         #        system_name + '.psf',
#         #        system_name + '.pdb',
#         #        system_name + '.png',
#         #        'indenter_insertion.out', # for checking vmd tcl log file
#         #    ],
#         #    'dest': dest,
#         #    'mode': 'copy' } )

#         fw = Firework(
#             [
#                 get_input_files_ft,
#                 indenter_insertion_fill_script_template_ft,
#                 indenter_insertion_ft,
#                 convert_ft
#                 #mkdir_ft,
#                 #copy_output_files_ft
#             ],
#             spec={
#                 #"_queueadapter": self.minimization_queueadapter,
#                 "_category":     fworker,
#                 #"_dupefinder":   DupeFinderExact(),
#                 "_files_in":  {
#                     "interface_file" :     interface_file,
#                     "indenter_file"  :     indenter_file
#                 },
#                 "_files_out":  {
#                     "pdb_file": system_name + '.pdb',
#                     "psf_file": system_name + '.psf',
#                     "data_file": system_name + '.lammps'
#                 },
#                 "system_name": system_name,
#                 "step"     :   "indenter_insertion",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_indenter_insertion".format(system_name))
#         return fw

#     def add_parameters_to_datafile_and_store(
#         self, system_name, reffile_identifier = None, outfile_identifier = None,
#         indenter_suffix = None, reffile_suffix = '_psfgen.data',
#         outfile_suffix = '_initial.lammps',
#         output_subfolder = 'datafiles_merged' ):

#         previous_system_name = system_name
#         if indenter_suffix is not None:
#             if system_name.endswith(indenter_suffix) and len(indenter_suffix) > 0:
#                 previous_system_name = system_name[:-len(indenter_suffix)]

#         if reffile_identifier is None:
#             reffile_identifier = previous_system_name + reffile_suffix

#         if outfile_identifier is None:
#             outfile_identifier = system_name + outfile_suffix

#         # pull reference file from data base
#         get_data_files_ft = GetFilesTask( {
#                 'identifiers':    [ reffile_identifier ],
#                 'new_file_names': [ 'reffile.lammps' ] } )

#         # MergeLammpsDataFiles contains functionality,
#         # merge.py is pizza.py-callable wrapper
#         # get_input_files_ft = FileTransferTask( {
#         #    'files': [
#         #        {
#         #            'src':  self.template_prefix + os.sep + 'MergeLammpsDataFiles.py',
#         #            'dest': 'MergeLammpsDataFiles.py'
#         #        },
#         #        {
#         #            'src':  self.template_prefix + os.sep + 'merge.py',
#         #            'dest': 'merge.py'
#         #        }
#         #    ],
#         #    'mode': 'copy' } )
#         get_input_files_ft = GetFilesTask( {
#                 'identifiers':    [ 'MergeLammpsDataFiles.py', 'merge.py' ],
#                 'new_file_names': [ 'MergeLammpsDataFiles.py', 'merge.py' ] } )

#         pizzapy_merge_ft =  ScriptTask.from_str(
#             self._template_pizzapy_merge_cmd.format(
#                 datafile = 'datafile.lammps',
#                 reffile  = 'reffile.lammps',
#                 outfile  = outfile_identifier ),
#             {
#                 'stdout_file':  'pizzapy_merge.out',
#                 'stderr_file':  'pizzapy_merge.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         # if existent, just deletes existing file
#         delete_ft = DeleteFilesTask(
#             {
#                 'identifiers': [ outfile_identifier ]
#             }
#         )

#         store_ft =  AddFilesTask(
#             {
#                 'paths':       [ outfile_identifier ],
#                 'identifiers': [ outfile_identifier ],
#                 'metadata': {
#                     'system_name':          system_name
#                 }
#             }
#         )

#         # dest = self.output_prefix + os.sep + system_name + os.sep + output_subfolder
#         # mkdir_ft =  ScriptTask.from_str(
#         #     'mkdir -p "{:s}"'.format(dest),
#         #        {
#         #            'use_shell':     True,
#         #            'fizzle_bad_rc': False
#         #        } )

#         # copy_output_files_ft = FileTransferTask( {
#         #   'files': [
#         #        outfile_identifier,
#         #        'pizzapy_merge.out', # for checking vmd tcl log file
#         #    ],
#         #    'dest': dest,
#         #    'mode': 'copy' } )

#         fw = Firework(
#             [
#                 get_data_files_ft,
#                 get_input_files_ft,
#                 pizzapy_merge_ft,
#                 delete_ft,
#                 store_ft,
#                 # mkdir_ft,
#                 # copy_output_files_ft
#             ],
#             spec={
#                 "_category":     self.std_worker,
#                 # "_dupefinder":   DupeFinderExact(),
#                 "_files_in": {
#                     "data_file": 'datafile.lammps'
#                 },
#                 "_files_out": {
#                     "data_file": outfile_identifier
#                 },
#                 "system_name": system_name,
#                 "step"     :   "pizzapy_merge",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_pizzapy_merge".format(system_name))
#         return fw

#     # LAMMPS

#     def minimize(self, system_name,
#         lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s}'):
#         """
#         Sample for lmp_suffix: for a call like
#             srun lmp -in lmp_minimization.input \
#                 -v has_indenter 1 -v robust_minimization 0 -v pbc2d 1 \
#                 -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
#                 -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
#         set lmp_suffix='-v has_indenter 1 -v robust_minimization 0 -v pbc2d 1 \
#             -v baseName {baseName:s} -v dataFile {dataFile:s}'
#         """

#         get_input_files_ft = GetFilesTask( {
#                 'identifiers':    [ 'lmp_header.input', 'lmp_minimization.input', 'extract_thermo.sh' ] } )
#         # get_input_files_ft = FileTransferTask( {
#         #    'files': [
#         #        {
#         #            'src':  self.template_prefix + os.sep + 'lmp_header.input',
#         #            'dest': 'lmp_header.input'
#         #        }, {
#         #            'src':  self.template_prefix + os.sep + 'lmp_minimization.input',
#         #            'dest': 'lmp_minimization.input'
#         #        }
#         #    ],
#         #    'mode': 'copy' } )

#         lmp_cmd = ' '.join((self._template_lmp_cmd.format(
#             inputFile = 'lmp_minimization.input' ), lmp_suffix_template.format(
#                 baseName=system_name, dataFile='datafile.lammps' ) ))

#         minimization_ft =  ScriptTask.from_str(
#             lmp_cmd,
#             {
#                 'stdout_file':  'lmp_minimization.out',
#                 'stderr_file':  'lmp_minimization.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         extract_thermo_ft =  ScriptTask.from_str(
#             'bash extract_thermo.sh {:s} {:s}'.format(
#                 system_name + '_minimization.log',
#                 system_name + '_minimization_thermo.out'),
#                 {
#                     'use_shell':    True, 'fizzle_bad_rc': False
#                 } )

#         # dest = self.output_prefix + os.sep + system_name + os.sep + 'minimized'
#         # mkdir_ft =  ScriptTask.from_str(
#         #    'mkdir -p "{:s}"'.format(dest),
#         #        {
#         #            'use_shell':     True,
#         #            'fizzle_bad_rc': False
#         #        } )

#         # copy_output_files_ft = FileTransferTask( {
#         #    'files': [
#         #        system_name + '_minimized.lammps',
#         #        system_name + '_minimization.log',
#         #        system_name + '_minimization_thermo.out',
#         #        'lmp_minimization.out',
#         #    ],
#         #    'dest': dest,
#         #    'mode': 'copy' } )

#         fw = Firework(
#             [
#                 get_input_files_ft,
#                 minimization_ft,
#                 extract_thermo_ft,
#                 #mkdir_ft,
#                 #copy_output_files_ft
#             ],
#             spec={
#                 "_queueadapter": self.minimization_queueadapter,
#                 "_category":     self.queue_worker,
#                 #"_dupefinder":   DupeFinderExact(),
#                 "_files_in":  {
#                     "data_file"   :  'datafile.lammps',
#                 },
#                 "_files_out":  {
#                     "data_file"   :         system_name + '_minimized.lammps',
#                 },
#                 "system_name": system_name,
#                 "step"     :   "minimization",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_minimzation".format(system_name))
#         return fw

#     def nvtEquilibrate(self, system_name,
#         lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s}'):
#         """
#         Sample for lmp_suffix: for a call like
#             srun lmp -in lmp_equilibration_nvt.input \
#                 -v has_indenter 0 -v pbc2d 0 -v reinitialize_velocities 1\
#                 -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
#                 -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
#         set lmp_suffix='-v has_indenter 0 -v reinitialize_velocities 1 \
#             -v baseName {baseName:s} -v dataFile {dataFile:s}'
#         """

#         get_input_files_ft = GetFilesTask( {
#                 'identifiers':    [ 'lmp_header.input', 'lmp_equilibration_nvt.input', 'extract_thermo.sh' ] } )
#         #get_input_files_ft = FileTransferTask( {
#         #    'files': [
#         #        {
#         #            'src':  self.template_prefix + os.sep + 'lmp_header.input',
#         #            'dest': 'lmp_header.input'
#         #        }, {
#         #            'src':  self.template_prefix + os.sep + 'lmp_equilibration_nvt.input',
#         #            'dest': 'lmp_equilibration_nvt.input'
#         #        }
#         #    ],
#         #    'mode': 'copy' } )

#         lmp_cmd = ' '.join((self._template_lmp_cmd.format(
#             inputFile = 'lmp_equilibration_nvt.input' ), lmp_suffix_template.format(
#                 baseName=system_name, dataFile='datafile.lammps' ) ) )

#         lmp_ft =  ScriptTask.from_str(
#             lmp_cmd,
#             {
#                 'stdout_file':  'lmp_nvtEquilibration.out',
#                 'stderr_file':  'lmp_nvtEquilibration.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         extract_thermo_ft =  ScriptTask.from_str(
#             'bash extract_thermo.sh {:s} {:s}'.format(
#                 system_name + '_nvtEquilibration.log',
#                 system_name + '_nvtEquilibration_thermo.out'),
#                 {
#                     'use_shell':    True, 'fizzle_bad_rc': False
#                 } )

#         #tail_ft_list = self.get_post_lammps_tasks( system_name,
#         #    step = 'equilibration_nvt',
#         #    suffix = '_nvtEquilibration',
#         #    lmp_suffix = '_nvtEquilibrated',
#         #    output_folder = 'nvtEquilibrated' )

#         fw = Firework(
#             [
#                 get_input_files_ft,
#                 lmp_ft,
# 		extract_thermo_ft
#                 #*tail_ft_list
#             ],
#             spec={
#                 "_queueadapter": self.equilibration_queueadapter,
#                 "_category":     self.queue_worker,
#                 #"_dupefinder":   DupeFinderExact(),
#                 "_files_in":  {
#                     "data_file"   :  'datafile.lammps',
#                 },
#                 "_files_out":  {
#                     "data_file"   :         system_name + '_nvtEquilibrated.lammps',
#                 },
#                 "system_name": system_name,
#                 "step"     :   "equilibration_nvt",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_equilibration_nvt".format(system_name))
#         return fw

#     def nptEquilibrate(self, system_name,
#         lmp_suffix_template='-v baseName {baseName:s} -v dataFile {dataFile:s}'):
#         """
#         Sample for lmp_suffix: for a call like
#             srun lmp -in lmp_equilibration_npt.input \
#                 -v has_indenter 1 -v pbc2d 0 -v reinitialize_velocities 1\
#                 -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
#                 -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
#         set lmp_suffix='-v has_indenter 1 -v reinitialize_velocities 1 \
#             -v baseName {baseName:s} -v dataFile {dataFile:s}'
#         """

#         get_input_files_ft = GetFilesTask( {
#                 'identifiers':    [ 'lmp_header.input', 'lmp_equilibration_npt.input', 'extract_thermo.sh' ] } )

#         lmp_cmd = ' '.join((self._template_lmp_cmd.format(
#             inputFile = 'lmp_equilibration_npt.input' ), lmp_suffix_template.format(
#                 baseName=system_name, dataFile='datafile.lammps' ) ))

#         lmp_ft =  ScriptTask.from_str(
#             lmp_cmd,
#             {
#                 'stdout_file':  'lmp_nptEquilibration.out',
#                 'stderr_file':  'lmp_nptEquilibration.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         extract_thermo_ft =  ScriptTask.from_str(
#             'bash extract_thermo.sh {:s} {:s}'.format(
#                 system_name + '_nptEquilibration.log',
#                 system_name + '_nptEquilibration_thermo.out'),
#                 {
#                     'use_shell':    True, 'fizzle_bad_rc': False
#                 } )

#         fw = Firework(
#             [
#                 get_input_files_ft,
#                 lmp_ft,
#                 extract_thermo_ft
#             ],
#             spec={
#                 "_queueadapter": self.equilibration_queueadapter,
#                 "_category":     self.queue_worker,
#                 "_files_in":  {
#                     "data_file"   :  'datafile.lammps',
#                 },
#                 "_files_out":  {
#                     "data_file"   :         system_name + '_nptEquilibrated.lammps',
#                 },
#                 "system_name": system_name,
#                 "step"     :   "equilibration_npt",
#                 "geninfo":     self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_equilibration_npt".format(system_name))
#         return fw

#     def production(self, system_name,
#         total_steps      = 5000000, # time steps, 5 mio ~ 10 ns
#         lmp_suffix_template=' '.join((
#             '-v baseName {baseName:s} -v dataFile {dataFile:s}',
#             '-v has_indenter 0 -v pbc2d 0 -v mpiio 0 -v use_colvars 0')) ):
#         """
#         Sample for lmp_suffix: for a call like
#             srun lmp -in lmp_production.input \
#                 -v has_indenter 0 -v pbc2d 0 -v mpiio 0 \
#                 -v thermo_frequency 1000 -v reinitialize_velocities 0 \
#                 -v use_colvars 0 -v productionSteps 1000 \
#                 -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
#                 -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
#         set lmp_suffix='-v has_indenter 0 -v reinitialize_velocities 0 \
#             -v baseName {baseName:s} -v dataFile {dataFile:s} ...'
#         """
#         get_input_files_ft = GetFilesTask( {
#             'identifiers':    [
#                'lmp_header.input',
#                'lmp_production.input',
#                'lmp_production_mixed.input',
#                'extract_thermo.sh' ] } )

#         lmp_cmd = ' '.join((self._template_lmp_cmd.format(
#             inputFile = 'lmp_production.input' ), lmp_suffix_template.format(
#                 baseName=system_name, dataFile='datafile.lammps' ) ))

#         lmp_ft =  ScriptTask.from_str(
#             lmp_cmd,
#             {
#                 'stdout_file':  'lmp_production.out',
#                 'stderr_file':  'lmp_production.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         extract_thermo_ft =  ScriptTask.from_str(
#             'bash extract_thermo.sh {:s} {:s}'.format(
#                 system_name + '_production.log',
#                 system_name + '_production_thermo.out'),
#                 {
#                     'use_shell':    True, 'fizzle_bad_rc': False
#                 } )

#         # tail_ft_list = self.get_post_lammps_tasks( system_name,
#         #    step = 'production', suffix = '_production', prefix = 'production_',
#         #    lmp_suffix = '_production_mixed',
#         #    output_folder = 'production_{:d}'.format(total_steps) )

#         fw = Firework(
#             [
#                 get_input_files_ft,
#                 lmp_ft,
#                 extract_thermo_ft
#                 #*tail_ft_list
#             ],
#             spec={
#                 "_queueadapter": self.production_queueadapter,
#                 "_category":     self.queue_worker,
#                 "_files_in":  {
#                     "data_file"   :  'datafile.lammps',
#                 },
#                 "_files_out":  {
#                     "data_file"   :     system_name + '_production.lammps',
#                     "trajectory_file" : system_name + '_production_mixed.nc',
#                     "thermo_file" :     system_name + '_production_thermo.out'
#                 },
#                 "system_name":    system_name,
#                 "step"     :      "production",
#                 "total_steps":    total_steps,
#                 "geninfo":        self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_produtcion_{:d}".format(system_name, total_steps) )
#         return fw

#     # TODO: parameters should appear in sim_df only
#     def colvars_production(self, system_name,
#         force_constant   = 2500.0, # kcal / mole
#         total_steps      = 500000, # time steps for indenter approach
#         initial_distance = 120.0, # nm, COM-COM
#         final_distance   = 10.0, #nm, COM-COM
#         lmp_suffix_template=' '.join((
#             '-v baseName {baseName:s} -v dataFile {dataFile:s}',
#             '-v has_indenter 1 -v pbc2d 0 -v mpiio 0 -v use_colvars 1')) ):
#         """
#         Sample for lmp_suffix: for a call like
#             srun lmp -in lmp_production.input \
#                 -v has_indenter 1 -v pbc2d 0 -v mpiio 0 \
#                 -v thermo_frequency 1000 -v reinitialize_velocities 0 \
#                 -v use_colvars 1 -v productionSteps 1000 \
#                 -v baseName 377_SDS_on_AU_111_51x30x2_monolayer \
#                 -v dataFile 377_SDS_on_AU_111_51x30x2_monolayer.lammps
#         set lmp_suffix='-v has_indenter 1 -v reinitialize_velocities 0 \
#             -v baseName {baseName:s} -v dataFile {dataFile:s} ...'
#         """

#         get_input_files_ft = FileTransferTask( {
#             'files': [
#                 {
#                     'src':  self.template_prefix + os.sep + 'lmp_header.input',
#                     'dest': 'lmp_header.input'
#                 }, {
#                     'src':  self.template_prefix + os.sep + 'lmp_production.input',
#                     'dest': 'lmp_production.input'
#                 }, {
#                     'src':  self.template_prefix + os.sep + 'lmp_production_mixed.input',
#                     'dest': 'lmp_production_mixed.input'
#                 } #,  {
#                 #    'src':  self.template_prefix + os.sep + 'colvars.inp',
#                 #    'dest': 'colvars.inp'
#                 # }
#             ],
#             'mode': 'copy' } )

#         colvars_script_writer_task_context = {
#             'header':         self.geninfo(),
#             'output_frequency':            1,
#             'restart_frequency':        1000,
#             'lower_boundary':            0.0,
#             'upper_boundary':          160.0,
#             'force_constant':              force_constant,
#             'initial_com_com_distance':  initial_distance,
#             'final_com_com_distance':      final_distance,
#             'total_steps':                    total_steps #  1ns
#         }

#         colvars_fill_script_template_ft = TemplateWriterTask( {
#             'context' :      colvars_script_writer_task_context,
#             'template_file': self.template_prefix + os.sep + 'colvars.inp',
#             'output_file':   'colvars.inp' } )

#         lmp_cmd = ' '.join((self._template_lmp_cmd.format(
#             inputFile = 'lmp_production.input' ), lmp_suffix_template.format(
#                 baseName=system_name, dataFile='datafile.lammps' ) ))

#         lmp_ft =  ScriptTask.from_str(
#             lmp_cmd,
#             {
#                 'stdout_file':  'lmp_production.out',
#                 'stderr_file':  'lmp_production.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':True
#             })

#         tail_ft_list = self.get_post_lammps_tasks( system_name,
#             step = 'production', suffix = '_production', prefix = 'production_',
#             lmp_suffix = '_production_mixed',
#             output_folder = 'production_{:d}'.format(total_steps) )

#         # insert removal of colvars output
#         # extract_thermo_ft =  ScriptTask.from_str(
#         #     self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
#         #         system_name + '_production.log',
#         #         system_name + '_production_thermo.out'),
#         #         {
#         #             'use_shell':    True, 'fizzle_bad_rc': False
#         #         } )
#         #
#         # colvars_file_endings = [
#         #     '*.colvars.traj',
#         #     '*.colvars.state',
#         #     '*.ti.count',
#         #     '*.ti.grad',
#         #     '*.ti.pmf'
#         # ]
#         #
#         # tar_ft = ScriptTask.from_str(
#         #     'tar -czf {:s} {:s}'.format(
#         #         'colvars.tar.gz',
#         #         ' '.join(colvars_file_endings) ),
#         #     {
#         #         'stdout_file':  'colvars_tar.out',
#         #         'stderr_file':  'colvars_tar.err',
#         #         'use_shell':    True,
#         #         'fizzle_bad_rc':False
#         #     })
#         #
#         # dest = self.output_prefix + os.sep + system_name + os.sep + 'production_{:d}'.format(total_steps)
#         # mkdir_ft =  ScriptTask.from_str(
#         #     'mkdir -p "{:s}"'.format(dest),
#         #         {
#         #             'use_shell':     True,
#         #             'fizzle_bad_rc': False
#         #         } )
#         #
#         # copy_output_files_ft = FileTransferTask( {
#         #     'files': [
#         #         system_name + '_production_mixed.lammps',
#         #         system_name + '_production_mixed.nc',
#         #         system_name + '_production.log',
#         #         system_name + '_production_thermo.out',
#         #         'lmp_production.out',
#         #         'colvars.tar.gz'
#         #     ],
#         #     'dest': dest,
#         #     'mode': 'copy' } )

#         fw = Firework(
#             [
#                 get_input_files_ft,
#                 colvars_fill_script_template_ft,
#                 lmp_ft,
#                 *tail_ft_list
#                 # extract_thermo_ft,
#                 # mkdir_ft,
#                 # tar_ft,
#                 # copy_output_files_ft
#             ],
#             spec={
#                 "_queueadapter": self.production_queueadapter,
#                 "_category":     self.queue_worker,
#                 "_dupefinder":   DupeFinderExact(),
#                 "_files_in":  {
#                     "data_file"   :  'datafile.lammps',
#                 },
#                 "_files_out":  {
#                     "data_file"   :  system_name + '_production.lammps',
#                 },
#                 "system_name":    system_name,
#                 "step"     :      "production",
#                 "force_constant": force_constant,
#                 "total_steps":    total_steps,
#                 "geninfo":        self.geninfo() # serves to distinguish duplicates
#             },
#             name="{:s}_produtcion_{:d}".format(system_name, total_steps) )
#         return fw

#     def get_post_lammps_tasks(self, system_name,
#         step='production', suffix=None, prefix=None, lmp_suffix = None,
#         output_folder = None):
#         if suffix is None:
#             suffix = '_' + step
#         if prefix is None:
#             prefix = step + '_'
#         if lmp_suffix is None:
#             lmp_suffix = suffix
#         if output_folder is None:
#             output_folder = step

#         # insert removal of colvars output
#         extract_thermo_ft =  ScriptTask.from_str(
#             self.template_prefix + os.sep + 'extract_thermo.sh {:s} {:s}'.format(
#                 system_name + suffix + '.log',
#                 system_name + suffix + '_thermo.out'),
#                 {
#                     'use_shell':    True, 'fizzle_bad_rc': False
#                 } )

#         colvars_file_endings = [
#             '*.colvars.traj',
#             '*.colvars.state',
#             '*.ti.count',
#             '*.ti.grad',
#             '*.ti.pmf'
#         ]

#         tar_ft = ScriptTask.from_str(
#             'tar -czf {:s} {:s}'.format(
#                 'colvars.tar.gz',
#                 ' '.join(colvars_file_endings) ),
#             {
#                 'stdout_file':  'colvars_tar.out',
#                 'stderr_file':  'colvars_tar.err',
#                 'use_shell':    True,
#                 'fizzle_bad_rc':False
#             })

#         dest = self.output_prefix + os.sep + system_name + os.sep + output_folder
#         mkdir_ft =  ScriptTask.from_str(
#             'mkdir -p "{:s}"'.format(dest),
#                 {
#                     'use_shell':     True,
#                     'fizzle_bad_rc': False
#                 } )

#         copy_output_files_ft = FileTransferTask( {
#             'files': [
#                 system_name + lmp_suffix + '.lammps',
#                 system_name + lmp_suffix + '.nc',
#                 system_name + suffix + '.log',
#                 system_name + suffix + '_thermo.out',
#                 'lmp' + suffix + '.out',
#                 'colvars.tar.gz'
#             ],
#             'dest':          dest,
#             'mode':          'copy',
#             'ignore_errors': True } )

#         return [ extract_thermo_ft, tar_ft, mkdir_ft, copy_output_files_ft ]

#     def post_lammps_fw(self, system_name, parent_fw_id, launch_dir = None,
#         step='production', suffix=None, prefix=None, lmp_suffix = None,
#         output_folder=None):
#         if suffix is None:
#             suffix = '_' + step
#         if prefix is None:
#             prefix = step + '_'
#         if lmp_suffix is None:
#             lmp_suffix = suffix
#         if output_folder is None:
#             output_folder = step

#         fw_spec = {
#             "_allow_fizzled_parents": True,
#             "_category":     self.std_worker,
#             "_dupefinder":   DupeFinderExact(),
#             "_files_out":  {
#                 "data_file"   :  system_name + suffix + '.lammps',
#             },
#             "system_name":    system_name,
#             "step"     :      'post_lammps',
#             "geninfo":        self.geninfo() # serves to distinguish duplicates
#         }
#         if launch_dir is not None:
#             fw_spec["_launch_dir"] = launch_dir

#         fw = Firework(
#             self.get_post_lammps_tasks( system_name,
#             step, suffix, prefix, lmp_suffix, output_folder ),
#             spec=fw_spec,
#             name="{:s}_post_lammps".format(system_name))
#         return fw

#     def batch_prepare_fw( self, system_names, fw_creator ):
#         fw_list = []
#         for system_name in system_names:
#             fw = fw_creator( system_name )
#             fw_list.append((system_name,fw))

#         fw_dict = dict(fw_list)
#         return fw_dict
