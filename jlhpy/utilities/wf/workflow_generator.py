# TODO:
# 2020/07/02 remove all project_id, only project
# 2020/12/06 merge labels on infiles lists

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

        # this adresses above issue
        if 'files_in_info' in self.kwargs:
            self.files_in_info = self.kwargs['files_in_info']

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

    def get_n_char_slug(self, suffix='', length=76):
        # timestamp - parameters - sub-workflow hierarchy - step

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')

        label = ' '.join((
            timestamp, self.fw_name_prefix,
            self.wf_name_prefix, suffix))
        slug = slugify(label)

        if len(slug) > length:  # ellipsis
            label = ' '.join((
                timestamp,
                self.fw_name_prefix,
                self.wf_name_prefix.split(':')[0],
                suffix))
            slug = slugify(label)

        if len(slug) > length:  # ellipsis
            slug = slug[:length//2-1].rstrip('-') + '--' + slug[-length//2+1:].lstrip('-')

        return slug

    def get_wf_label(self):
        return self.wf_name

    def build_fw(self, fts, step_label,
                 parents=[], category=None, files_in={}, files_out={},
                 queueadapter=None, fw_spec=None, **kwargs):
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
        if kwargs is not None:
            fw_spec.update(kwargs)

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


class EncapsulatingWorkflowGenerator(WorkflowGenerator):
    """Common base for all branching workflows."""

    @property
    def sub_wf(self):
        return self._sub_wf

    @sub_wf.setter
    def sub_wf(self, sub_wf):
        self._sub_wf = sub_wf

    @sub_wf.deleter
    def sub_wf(self):
        del self._sub_wf

    def __init__(self, *args, sub_wf=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_wf = sub_wf(*args, **kwargs)

    def push_infiles(self, fp):
        """fp: FilePad"""
        if hasattr(self.sub_wf, 'push_infiles'):
            self.sub_wf.push_infiles(fp)

    def main(self, fws_root=[]):
        return self.sub_wf.main(fws_root)

    def get_as_independent(self, fws_root=[]):
        """Return a self-sufficient FireWorks list with pull and push stub."""
        return self.sub_wf.get_as_independent(fws_root)

    def get_as_root(self, fws_root=[]):
        """Return as root FireWorks list with pull stub, but no push stub."""
        return self.sub_wf.get_as_root(fws_root)

    def get_as_leaf(self, fws_root=[]):
        """Return as leaf FireWorks list without pull stub, but with push stub."""
        return self.sub_wf.get_as_leaf(fws_root)

    def get_as_embedded(self, fws_root=[]):
        """Return as embeded FireWorks list without pull and push stub."""
        return self.sub_wf.get_as_embedded(fws_root)


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
                    *[p if (isinstance(p, Iterable) and not isinstance(p, str)) 
                      else [p] for p in parameter_package.values()]
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