#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[6]:


import os, sys

import numpy as np
#import ase

import ovito


# In[7]:


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


# In[8]:


from IPython.display import Image


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


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


# In[11]:


from ase.lattice.cubic import FaceCenteredCubic
from ase.build import fcc111


# In[31]:


substrate_ref_ortho = fcc111('Au',size=(12,14,6),a=4.075,periodic=True,orthogonal=True)


# In[32]:


substrate_ref_ortho.cell


# In[33]:


staticView = nv.show_ase(substrate_ref_ortho)
staticView.remove_ball_and_stick()
staticView.add_spacefill()
staticView


# In[35]:


write('AU_111_12x7x2.xyz',substrate_ref_ortho)


# In[36]:


write('AU_111_12x7x2.pdb',substrate_ref_ortho)


# In[43]:


np.max(substrate_ref_ortho.positions,axis=0)


# In[47]:


np.array([5.76, 4.739, 15.08], dtype=float)/2


# In[48]:


np.pi


# In[ ]:




