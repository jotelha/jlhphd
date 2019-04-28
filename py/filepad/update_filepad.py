#!/usr/bin/env python
# run this script from this folder to update all relevant files in the filepad
from fireworks.utilities.filepad import FilePad
import pandas as pd
import os

fp = FilePad(
    host='localhost',
    port=27018,
    database='fireworks-jhoermann',
    username='fireworks',
    password='fireworks')

prefix = os.getcwd()

#template_prefix = os.path.split(prefix)[0]
template_prefix = "/mnt/dat/work/testuser/adsorption/N_surfactant_on_substrate_template"

print("Template prefix: {:s}".format(template_prefix))

# TCL:

fp.delete_file(identifier='jlh_vmd.tcl')
fp.add_file(
    os.path.join(template_prefix,'vmd','jlh_vmd.tcl'),
    identifier='jlh_vmd.tcl',
    metadata={
        'type':     'script',
        'language': 'tcl',
        'usecase':  'indenter insertion in vmd'})

fp.delete_file(identifier='indenter_insertion.tcl')
fp.add_file(
    os.path.join(template_prefix,'vmd','indenter_insertion.tcl'),
    identifier='indenter_insertion.tcl',
    metadata={
        'type':     'template',
        'language': 'tcl',
        'usecase':  'indenter insertion in vmd'})

# LAMMPS:
fp.delete_file(identifier='lmp_equilibration_npt.input')
fp.add_file(
    os.path.join(template_prefix,'lmp_input','lmp_equilibration_npt.input'),
    identifier='lmp_equilibration_npt.input',
    metadata={
        'type':     'input',
        'language': 'LAMMPS',
        'usecase':  'npt equilibration'})

fp.delete_file(identifier='lmp_equilibration_nvt.input')
fp.add_file(
    os.path.join(template_prefix,'lmp_input','lmp_equilibration_nvt.input'),
    identifier='lmp_equilibration_nvt.input',
    metadata={
        'type':     'input',
        'language': 'LAMMPS',
        'usecase':  'nvt equilibration'})

fp.delete_file(identifier='lmp_header.input')
fp.add_file(
    os.path.join(template_prefix,'lmp_input','lmp_header.input'),
    identifier='lmp_header.input',
    metadata={
        'type':     'input',
        'language': 'LAMMPS',
        'usecase':  'surfactant adsorption'})

fp.delete_file(identifier='lmp_minimal_header.input')
fp.add_file(
    os.path.join(template_prefix,'lmp_input','lmp_minimal_header.input'),
    identifier='lmp_minimal_header.input',
    metadata={
        'type':     'input',
        'language': 'LAMMPS',
        'usecase':  'surfactant adsorption'})

fp.delete_file(identifier='lmp_minimization.input')
fp.add_file(
    os.path.join(template_prefix,'lmp_input','lmp_minimization.input'),
    identifier='lmp_minimization.input',
    metadata={
        'type':     'input',
        'language': 'LAMMPS',
        'usecase':  'minimization'})

fp.delete_file(identifier='lmp_production.input')
fp.add_file(
    os.path.join(template_prefix,'lmp_input','lmp_production.input'),
    identifier='lmp_production.input',
    metadata={
        'type':     'input',
        'language': 'LAMMPS',
        'usecase':  'production'})
