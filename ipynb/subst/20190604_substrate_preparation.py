#!/usr/bin/env python
# coding: utf-8

# # Prepare 25nm x 25nm x 25 nm substrate block

# In[55]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[56]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[57]:


import os, sys

import numpy as np
#import ase

# import ovito


# In[58]:


# file formats, input - output
import ase.io
from ase.io import read, write
#import parmed as pmd

# visualization
from ase.visualize import view
from ase.visualize.plot import plot_atoms

import nglview as nv
import matplotlib.pyplot as plt
import ipywidgets # just for jupyter notebooks


# In[7]:


from IPython.display import Image


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# matplotlib settings

# expecially for presentation, larger font settings for plotting are recommendable
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 26

plt.rc("font", size=MEDIUM_SIZE)          # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams["figure.figsize"] = (8,5) # the standard figure size

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 14
plt.rcParams["lines.markeredgewidth"]=1


# In[10]:


from ase.lattice.cubic import FaceCenteredCubic
from ase.build import fcc111


# In[11]:


substrate_ref_ortho = fcc111('Au',size=(87,100,106),a=4.075,periodic=True,orthogonal=True)


# In[18]:


substrate_ref_ortho = fcc111('Au',size=(2,2,2),a=4.075,periodic=True,orthogonal=True)


# In[54]:


substrate_ref_ortho


# In[60]:


unit_cell = fcc111('Au',size=(1,2,1),a=4.075,periodic=True,orthogonal=True)


# In[49]:


size = np.round(np.array((250.0,250.0,250.0))/unit_cell.cell.lengths()*np.array([1.,2.,1.])).astype(int)


# In[52]:


size


# In[51]:


substrate_ref_ortho = fcc111('Au',size=size,a=4.075,periodic=True,orthogonal=True)


# In[53]:


np.array((250.0,250.0,250.0))/


# In[25]:


np.array((250.0,250.0,250.0))/unit_cell.cea


# In[15]:


250.0/np.array((87,100,106))


# In[12]:


substrate_ref_ortho.cell


# In[29]:


staticView = nv.show_ase(unit_cell)
staticView.remove_ball_and_stick()
staticView.add_spacefill()
staticView


# In[12]:


prefix = '/mnt/dat/work/testuser/adsorption/N_surfactant_on_substrate_template/dat/subst/coord/AU_111_250Ang_cube'


# In[13]:


write(os.path.join(prefix,'AU_111_250Ang_cube.xyz'),substrate_ref_ortho)


# In[14]:


write(os.path.join(prefix,'AU_111_250Ang_cube.pdb'),substrate_ref_ortho)


# In[15]:


np.max(substrate_ref_ortho.positions,axis=0)


# In[ ]:




