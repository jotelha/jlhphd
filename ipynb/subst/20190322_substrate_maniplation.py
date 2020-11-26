#!/usr/bin/env python
# coding: utf-8

# # Substrate replacement
# Replace existing substrate by new substrate of different width

# In[1]:


import os, sys


# In[2]:


import numpy as np


# In[3]:


import ase


# In[4]:


from ase.visualize import view
import nglview as nv


# In[5]:


import ovito


# In[6]:


os.getcwd()


# In[7]:


py_prefix = os.path.split( os.getcwd() )[0]     + os.path.sep + 'adsorption'     + os.path.sep + 'N_surfactant_on_substrate_template'     + os.path.sep + 'py'


# In[8]:


py_prefix


# In[9]:


sys.path.append(py_prefix)


# In[10]:


import postprocessing


# In[11]:


postprocessing.sds_t2n_array


# In[12]:


# Read full lammps datafile
f = ase.io.read('datafile_full.lammps',format='lammps-data')


# In[14]:


f.set_cell(np.eye(3)*(np.max(f.get_positions(),axis=0) - np.min(f.get_positions(),axis=0)))


# In[15]:


f.get_cell()


# In[17]:


f.get_cell_lengths_and_angles()[:3]


# In[ ]:


f_reference = f.copy()


# In[ ]:


# do not alter atomic numbers:
f_reference.set_atomic_numbers(
   postprocessing.sds_t2n_array[f.get_atomic_numbers() ] )


# In[ ]:


import parmed


# In[ ]:


parmed.load_file()


# In[ ]:


f_reference.get_chemical_formula()


# In[ ]:


# remove water for display purposes
g = f_reference[ 
    (np.array(f_reference.get_chemical_symbols()) != 'O') & (
        np.array(f_reference.get_chemical_symbols()) != 'H')]
f_reference_dry = g.copy()


# In[ ]:


import logging


# In[ ]:


os.getenv("PATH").split( '|' )


# In[ ]:


os.pathsep


# In[ ]:


logging.root.level


# In[ ]:


logging.WARNING


# In[ ]:


logging.getLevelName( logging.root.level )


# In[ ]:


logger = logging.getLogger('exchange_substrate.exchangeSubstrate')


# In[ ]:


logger.level


# In[ ]:


f_reference_dry


# In[ ]:


staticView = nv.show_ase(f_reference_dry)

staticView.remove_ball_and_stick()
staticView.add_spacefill()

staticView.download_image(filename='test.png')


# In[ ]:


# assume only substrate is Au and extract
substrate = f[ (np.array(f_reference.get_chemical_symbols()) == 'Au') ]


# In[ ]:


len(substrate)


# In[ ]:


staticView = nv.show_ase(substrate)
staticView.remove_ball_and_stick()
staticView.add_spacefill()
staticView


# In[ ]:


# simulation box as defined in data file
print(f.get_cell())


# In[ ]:


# mininimum and maximum coordinates defined by substrate
substrate_box = np.array([np.min(substrate.get_positions(),axis=0),np.max(substrate.get_positions(),axis=0)])


# In[ ]:


substrate_box


# In[ ]:


substrate_measures = substrate_box[1] - substrate_box[0]


# In[ ]:


substrate_measures


# In[ ]:


# cellX = substrate.get_cell()[0,0]
# cellY = substrate.get_cell()[1,1]
cellZ = substrate.get_cell()[2,2]


# In[ ]:


# shift substrate back into unit box
# make minimum x and y coordinates align with 0,0
shiftX = - np.min(substrate.get_positions(),axis=0)[0] 
shiftY = - np.min(substrate.get_positions(),axis=0)[1] 
# shift whole system in such a way that coordinateas of substrate
# wrapped at periodic boundary to the "upper" half of the box
# lie outside of the box now
shiftZ = cellZ - np.min(
    substrate.get_positions()[
        substrate.get_positions()[:,2] > cellZ/2],axis=0)[2] 

# wrap these coordinates back to lower half
# new_positions = ase.geometry.wrap_positions(
#    f.get_positions() + [shiftX,shiftY,shiftZ], f.get_cell(), pbc=(0,0,1),
#    eps=1e-3)

new_positions_pbc3 = ase.geometry.wrap_positions(
    f.get_positions() + [shiftX,shiftY,shiftZ], f.get_cell(), pbc=(1,1,1),
    eps=1e-3)


# In[ ]:


np.min(new_positions_pbc3,axis=0)


# In[ ]:


# np.max(new_positions,axis=0)


# In[ ]:


np.max(new_positions_pbc3,axis=0)


# In[ ]:


# f_pbc1 = f.copy()


# In[ ]:


# f_pbc1.set_positions(new_positions)


# In[ ]:


f_pbc3 = f.copy()


# In[ ]:


f_pbc3.set_positions(new_positions_pbc3)


# In[ ]:


# wrapped_substrate = f_pbc1[ 
#    (np.array(f_pbc1.get_chemical_symbols()) == 'Au') ]


# In[ ]:


# extract "wrapped" substrate from now neatly aligned system
wrapped_substrate_pbc3 = f_pbc3[ 
    (np.array(f_reference.get_chemical_symbols()) == 'Au') ]


# In[ ]:


# g = f_pbc1[ 
#    (np.array(f_pbc1.get_chemical_symbols()) != 'O') & (
#        np.array(f_pbc1.get_chemical_symbols()) != 'H')]
# f_pbc1_dry = g.copy()


# In[ ]:


g = f_pbc3[ 
    (np.array(f_reference.get_chemical_symbols()) != 'O') & (
        np.array(f_reference.get_chemical_symbols()) != 'H')]
f_pbc3_dry = g.copy()


# In[ ]:


staticView = nv.show_ase(f_pbc3_dry)
staticView.remove_ball_and_stick()
staticView.add_spacefill()
staticView


# In[ ]:


from ase.lattice.cubic import FaceCenteredCubic


# In[ ]:


# create a "perfect" crystaline reference substrate
# of same measures as currently used substrate
reference_substrate = FaceCenteredCubic(
    'Au', directions=[[1,-1,0],[1,1,-2],[1,1,1]], size=(51,30,2), pbc=(1,1,0))


# In[ ]:


staticView = nv.show_ase(reference_substrate)
staticView.remove_ball_and_stick()
staticView.add_spacefill()
staticView


# In[ ]:


len(reference_substrate)


# In[ ]:


len(wrapped_substrate_pbc3)


# In[ ]:


reference_substrate.get_center_of_mass()


# In[ ]:


wrapped_substrate_pbc3.get_center_of_mass()


# In[ ]:


# compare the two substrates' center of mass
# they still might be slightly disaligned
# especially in z-direction, if lattice exhibits defects


# In[ ]:


reference_substrate_shift = wrapped_substrate_pbc3.get_center_of_mass() - reference_substrate.get_center_of_mass()


# In[ ]:


aligned_reference_substrate = reference_substrate.copy()


# In[ ]:


aligned_reference_substrate.positions = reference_substrate.positions + reference_substrate_shift


# In[ ]:


# compare again after shift


# In[ ]:


aligned_reference_substrate.get_center_of_mass()


# In[ ]:


wrapped_substrate_pbc3.get_center_of_mass()


# In[ ]:


# create new substrate of different thickness


# In[ ]:


new_substrate_16 = FaceCenteredCubic(
    'Au', directions=[[1,-1,0],[1,1,-2],[1,1,1]], size=(51,30,16), pbc=(1,1,0))a


# In[ ]:


new_substrate = new_substrate_16.copy()


# In[ ]:


# reference substrate and new substrate should by x-y-aligned
# but still have to be "upper surface" - aligned


# In[ ]:


new_substrate.positions = new_substrate.positions + reference_substrate_shift


# In[ ]:


np.max(aligned_reference_substrate.positions,axis=0)


# In[ ]:


np.max(new_substrate.positions,axis=0)


# In[ ]:


# ATTENTION: alignment based on maximum coordinates is 
# prone to errors int the case of defetcs (protrusions)
shiftZ = (
    np.max(aligned_reference_substrate.positions,axis=0) \
    - np.max(new_substrate.positions,axis=0))[2]


# In[ ]:


shiftZ


# In[ ]:


aligned_new_substrate = new_substrate.copy()


# In[ ]:


aligned_new_substrate.positions = new_substrate.positions + [0,0,shiftZ]


# In[ ]:


np.max(aligned_reference_substrate.positions,axis=0)


# In[ ]:


np.max(aligned_new_substrate.positions,axis=0)


# In[ ]:


# remove old substrate from original system
nonSubstrate = f_pbc3[ (np.array(f_reference.get_chemical_symbols()) != 'Au') ]


# In[ ]:


# find lammps data file rtype for Au
substrate_type = np.where(
    postprocessing.sds_t2n_array == ase.data.atomic_numbers['Au'])[0][0]


# In[ ]:


substrate_type


# In[ ]:


substrate.arrays['numbers']


# In[ ]:


[substrate_type]*9


# In[ ]:


aligned_new_substrate.set_atomic_numbers(
    [substrate_type]*len(aligned_new_substrate))


# In[ ]:


aligned_new_substrate


# In[ ]:


postprocessing.sds_t2e_array[10]


# In[ ]:


np.where(f_reference.arrays['type'] == 11)


# In[ ]:


# combine new substrtae and old system without old substrate
f_new = nonSubstrate + aligned_new_substrate


# In[ ]:


g = f_new[ 
    (np.array(f_reference.get_chemical_symbols()) != 'O') & (
        np.array(f_reference.get_chemical_symbols()) != 'H')]
f_new_dry = g.copy()


# In[ ]:


staticView = nv.show_ase(f_new_dry)
staticView.remove_ball_and_stick()
staticView.add_spacefill()
staticView


# In[ ]:


# ase cannot write LAMMPS data files, thus use ovito


# In[ ]:


from ovito.io import import_file, export_file


# In[ ]:


from ovito.data import DataCollection


# In[ ]:


from ovito.pipeline import StaticSource, Pipeline


# In[128]:


f_new[:1].arrays


# In[129]:


f_ovito = f_new.copy()


# In[130]:


# ovito does not process any string type attributes
del f_ovito.arrays['angles']
del f_ovito.arrays['bonds']
del f_ovito.arrays['dihedrals']


# In[80]:


# based on previous data file
pipeline = import_file('datafile_full.lammps',atom_style='full')


# In[88]:


from ovito.modifiers import SelectTypeModifier, DeleteSelectedModifier


# In[83]:


pipeline.modifiers.append( SelectTypeModifier(
    operate_on = "particles",
    property="Particle Type",
    types = {11} ))


# In[90]:


pipeline.modifiers.append( DeleteSelectedModifier())


# In[91]:


dc = pipeline.compute()


# In[92]:


dc.number_of_particles


# In[94]:


dc.


# In[ ]:


# based on ase-processed system


# In[131]:


data = DataCollection.create_from_ase_atoms(f_ovito)


# In[132]:


type(data)


# In[133]:


pipeline = Pipeline(source = StaticSource(data = data))


# In[134]:


export_file(
    pipeline, "processed.lammps", "lammps_data", atom_style = 'full')


# In[ ]:





# In[217]:


# Read full lammps datafile
f_processed = ase.io.read('processed.lammps',format='lammps-data')
#f_processed.set_atomic_numbers(
#    postprocessing.sds_t2n_array[f_processed.get_atomic_numbers() ] )


# In[223]:


f_new


# In[219]:


f_processed


# In[224]:


# remove stuff for display purposes
g = f_processed[ (np.array(f_processed.get_chemical_symbols()) == 'Be') ]
f_disp = g.copy()


# In[225]:


f_disp


# In[226]:


staticView = nv.show_ase(f_disp)

staticView.remove_ball_and_stick()
staticView.add_spacefill()

staticView


# In[187]:


postprocessing.sds_t2n_array


# In[ ]:




