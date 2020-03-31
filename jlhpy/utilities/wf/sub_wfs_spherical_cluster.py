#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:53:39 2020

@author: jotelha
"""

import datetime
# import dill
import pymongo

from fireworks import Firework, Workflow
from fireworks.user_objects.firetasks.filepad_tasks import GetFilesByQueryTask
from imteksimfw.fireworks.user_objects.firetasks.cmd_tasks import PyEnvTask

from jlhpy.utilities.geometry.bounding_sphere import get_bounding_sphere_from_file

from jlhpy.utilities.wf.serialize import serialize_module_obj
from jlhpy.utilities.wf.workflow_generator import SubWorkflowGenerator, HPC_SPECS

class GetBoundingSphereSubWorkflowGenerator(SubWorkflowGenerator):

    def pull(self, fws_root=[]):
        # global project_id, source_project_id, HPC_SPECS, machine

        fw_list = []

        files_in = {}
        files_out = {
            'data_file':       'default.pdb',
        }

        fts_fetch = [
            GetFilesByQueryTask(
                query = {
                    'metadata->project':    self.project_id, # earlier
                    'metadata->type':       'initial_file_pdb',
                    'metadata->nmolecules': self.kwargs["nmolecules"]
                },
                sort_key = 'metadata.datetime',
                sort_direction = pymongo.DESCENDING,
                limit = 1,
                new_file_names = ['default.pdb'] ) ]

    def main(self, fws_root):
        step = '_'.join((type(self).__name__, 'main'))
        fw_list = []

        files_in = {
            'input_file':      'default.pdb',
        }
        files_out = {}

        func_str = serialize_module_obj(get_bounding_sphere_from_file)

        fts = [ PyEnvTask(
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

        fw = Firework(fts,
            name = ', '.join((
                step, self.fw_name_template.format(**self.kwargs))),
            spec = {
                '_category': self.hpc_specs[self.machine]['fw_noqueue_category'],
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
        return fw_list

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
