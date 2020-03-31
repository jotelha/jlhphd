#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:53:39 2020

@author: jotelha
"""

import datetime
import glob
import os
import pymongo

from fireworks import Firework, Workflow
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import PickledPyEnvTask

from jlhpy.utilities.geometry.bounding_sphere import get_bounding_sphere_from_file

from jlhpy.utilities.wf.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator


class GetBoundingSphereSubWorkflowGenerator(SubWorkflowGenerator):

    def push_infiles(self, fp):
        step = self.get_step_label('push_infiles')

        infiles = sorted(glob.glob(os.path.join(
            self.infile_prefix, '*.pdb')))

        files = {os.path.basename(f): f for f in infiles}

        # metadata common to all these files
        metadata = {
            'project': self.project_id,
            'type': 'initial_file_pdb',
            'step': step,
            **self.kwargs
        }

        fp_files = []

        # insert these input files into data base
        for name, file_path in files.items():
            identifier = '/'.join((self.project_id, name))  # identifier is like a path on a file system
            metadata["name"] = name
            fp_files.append(
                fp.add_file(
                    file_path,
                    identifier=identifier,
                    metadata=metadata))

        return fp_files

    def pull(self, fws_root=[]):
        step = self.get_step_label('pull')

        fw_list = []

        files_in = {}
        files_out = {
            'data_file':       'default.pdb',
        }

        fts = [
            GetFilesByQueryTask(
                query = {
                    'metadata->project':    self.project_id, # earlier
                    'metadata->type':       'initial_file_pdb',
                    #'metadata->nmolecules': self.kwargs["nmolecules"]
                    ## TODO: generalize query
                },
                sort_key = 'metadata.datetime',
                sort_direction = pymongo.DESCENDING,
                limit = 1,
                new_file_names = ['default.pdb'] ) ]

        fw = Firework(fts,
            name = ', '.join((
                step, self.fw_name_template.format(**self.kwargs))),
            spec = {
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in': files_in,
                '_files_out': files_out,
                'metadata': {
                    'project': self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':    step,
                     **self.kwargs
                }
            },
            parents = fws_root )

        fw_list.append(fw)

        return fw_list, fw

    def main(self, fws_root):
        step = self.get_step_label('main')
        fw_list = []

        files_in = {
            'data_file':      'default.pdb',
        }
        files_out = {}

        func_str = serialize_module_obj(get_bounding_sphere_from_file)

        fts = [ PickledPyEnvTask(
            func = func_str,
            args = [ 'default.pdb' ],
            outputs = [
                'metadata->system->indenter->bounding_sphere->center',
                'metadata->system->indenter->bounding_sphere->radius',
            ],
            stderr_file  = 'std.err',
            stdout_file  = 'std.out',
            store_stdout = True,
            store_stderr = True,
            ) ]

        fw = Firework( fts,
            name = ', '.join((
                step, self.fw_name_template.format(**self.kwargs))),
            spec = {
                '_category': self.hpc_specs['fw_noqueue_category'],
                '_files_in':  files_in,
                '_files_out': files_out,
                'metadata': {
                    'project':  self.project_id,
                    'datetime': str(datetime.datetime.now()),
                    'step':     step,
                     **self.kwargs
                }
            },
            parents = fws_root )

        fw_list.append(fw)

        return fw_list, fw

    # def get_center(self, infile):
    #
    #     S = atoms.get_positions()
    #     C, R_sq = miniball.get_bounding_ball(S)
    #     R = np.sqrt(R_sq)
    #     del S
    #
    #     #xmin = atoms.get_positions().min(axis=0)
    #     #xmax = atoms.get_positions().max(axis=0)
    #
    #     self.C = C * UNITS.angstrom
    #     self.R = R * UNITS.angstrom
    #     self.A = 4*np.pi*R**2 * UNITS.angstrom**2
    #
    #     return C, R
    #     # n_per_nm_sq = np.array([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # molecules per square nm
    #     # N = np.round(A_nm*n_per_nm_sq).astype(int)
    #
    #     # return A
    #     #A_nm = A_Ang / 10**2
