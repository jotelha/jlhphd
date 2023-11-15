#!/usr/bin/env python
# coding: utf-8

# # EAM substrate lattice constants
# probed for substrate block of crystal plane orientation [[1,-1,0],[1,1,-2],[1,1,1]] cell constant mutliples (51,30,8)  measures in xyz directions. Initial configuration created with
# ```python
# FaceCenteredCubic('Au', directions=[[1,-1,0],[1,1,-2],[1,1,1]], size=(51,30,8), pbc=(1,1,0) )
# ```

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', 'Application.log_level="WARN"')


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import logging
from postprocessing import analyze_rdf, logger


# In[3]:


# matplotlib settings

# expecially for presentation, larger font settings for plotting are recommendable
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 26

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titlex

plt.rcParams["figure.figsize"] = (8,5) # the standard figure size

plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 14
plt.rcParams["lines.markeredgewidth"]=1 


# In[3]:


def find_peak(rdf_file, peak_range = (4.0,4.2), plot = True, verbose = True, zero = 1e-8, bin_scale= 5 ):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tabulate import tabulate
    # format is ovito 3.0.0-dev234 coordination analysis text output
    df = pd.read_csv(rdf_file,
        delim_whitespace=True,header=None,skiprows=3,index_col=0,
        names=['bin','distance','weight'])
    
    mean_bin_width = np.sum((df.distance - np.roll(df.distance,1))[1:]) / len(df[1:])
        
    selection = (df.distance > peak_range[0]) & (df.distance < peak_range[1])
    nonzero   = df.weight > zero
    #selected_bins = np.nonzero(selection)[0]
    non_zero_selected_bins = np.nonzero( selection & nonzero )[0]
    
    peak_position = ( np.sum(df.distance[selection]*df.weight[selection]) / np.sum(df.weight[selection]) )

    if plot:
        plt.bar(df.distance[selection],df.weight[selection], width = bin_scale*mean_bin_width)
        
    if verbose:
        msg = [ 
            ["Mean bin width:", "{:g}".format(mean_bin_width) ],
            ["Non-zero bins within selection:", *non_zero_selected_bins],
            ["According distances:", *df.distance[ non_zero_selected_bins ]],
            ["According weights:", *df.weight[ non_zero_selected_bins ]],
            ["Peak position:", "{:g}".format(peak_position)] ]
        print(tabulate(msg, tablefmt='plain')) # for fixed number format: floatfmt=".8f"
    
    return peak_position


# ## Initial configuration, approximate box measures

# In[15]:


logger.setLevel(logging.INFO)


# In[16]:


peak_positions = {}


# In[17]:


peak_positions["Initial configuration"] = analyze_rdf(
    '010_initial_config.txt', interval=(3.9,4.2))


# ## Minimized bulk, fixed box

# For the following 3d-periodic minimizations, placed in a box approximately aligned to the lattice with LAMMPS input snippet
# 
# ```LAMMPS
# # [...]
# # "tighten" box around system and add one lattice constant in all directions:
# change_box all boundary s s s
# 
# # zero-align system
# displace_atoms all move $((-xlo)) $((-ylo)) $((-zlo)) units box
# 
# # switch on periodic bc again
# change_box all boundary p p p
# 
# # add one lattice constant in each direction
# 
# # default lattice constant for Au
# variable substrate_lattice_constant index 4.07
# # fcc lattice with (111) normal in z-direction
# lattice fcc ${substrate_lattice_constant} orient x 1 -1 0 orient y 1 1 -2 orient z 1 1 1
# # If the spacing option is not specified, the lattice spacings are computed by
# # LAMMPS in the following way. A unit cell of the lattice is mapped into the
# # simulation box (scaled and rotated), so that it now has (perhaps) a modified
# # size and orientation. The lattice spacing in X is defined as the difference
# # between the min/max extent of the x coordinates of the 8 corner points of the
# # modified unit cell (4 in 2d). Similarly, the Y and Z lattice spacings are
# # defined as the difference in the min/max of the y and z coordinates.
# 
# # Computed with the help of the following web tool
# # http://neutron.ornl.gov/user_data/hb3a/exp16/tools/Crystal%20Plane%20Spacings%20and%20Interplanar%20Angles.htm
# # the inter-planar spacings of [1,-1,0], [1,1,-2] and [1,1,1] are respectively
# #   d = [2.878, 1.662, 2.35]
# # Interestingly, the automatic computation of lattice spacings xlat,ylat,zlat
# # in LAMMPS results in 
# #   l = [5.75585, 6.64628, 7.04945], 
# # which is the equivalent of
# #   a*{Sqrt[2], Sqrt[8/3], Sqrt[3]} 
# # with lattice constant a = 4.07 for Au.
# # This correspongs to 
# #   n = [2, 4, 3] = l / d (element-wise)
# # crystal planes withinin each dimension within the LAMMPS-computed 
# # lattice spacings.
# change_box all x final $(xlo) $(xhi+(xlat/2.0)) y final $(ylo) $(yhi+(ylat/4.0)) z final $(zlo) $(zhi+(zlat/3.0)) units box
# ```
# 

# In[18]:


peak_positions["Minimized bulk, fixed box"] = analyze_rdf('020_bulk_minimized_box_fixed.txt', interval=(3.9,4.2))


# ## Minimized bulk, relaxed box

# In[19]:


peak_positions["Minimized bulk, relaxed box"] = analyze_rdf('030_bulk_minimized_box_relaxed.txt', interval=(3.9,4.2))


# ## Minimized slab, fixed box

# adedd vacuum slab in z-direction

# In[20]:


peak_positions["Minimized slab, fixed box"] = analyze_rdf('040_slab_minimized_box_fixed.txt', interval=(3.9,4.2))


# ## Minimized slab, relaxed box
# box relaxed in xy directions

# In[10]:


peak_positions["Minimized slab, relaxed box"] = find_peak('050_slab_minimized_box_relaxed.txt', peak_range=(3.9,4.2))


# ## 20 ps NVT bulk, temperature ramped 0 to 298K
# final configuration

# In[11]:


peak_positions["NVT 20ps bulk, final"] = find_peak('065_bulk_nvt_20_ps_0_to_298_K_ramped_final.txt', peak_range=(3.5,4.5))


# ## 20 ps NVT slab, temperature ramped 0 to 298K
# final configuration

# In[12]:


peak_positions["NVT 20ps slab, final"] = find_peak('060_slab_nvt_20_ps_0_to_298_K_ramped_final.txt', peak_range=(3.5,4.5))


# ## 200 ps NPT bulk, 298 K
# ### relaxed at zero pressure, final configuration

# In[13]:


peak_positions["NPT 200ps bulk, final"] = find_peak('075_bulk_npt_200_ps_298_K_0_atm_final.txt', peak_range=(3.5,4.5))


# ### relaxed at zero pressure, averaged over all 1000 frames

# In[3]:


rdf_ref = np.loadtxt('075_bulk_npt_200_ps_298_K_0_atm_final.txt')


# In[32]:


df = pd.read_csv('075_bulk_npt_200_ps_298_K_0_atm_final.txt',
       delim_whitespace=True,header=None,skiprows=3,index_col=0,
       names=['bin','distance','weight'])


# In[4]:


plt.plot(rdf_ref[:,1],rdf_ref[:,2])


# In[35]:


rdf_tot = np.loadtxt('075_bulk_npt_200_ps_298_K_0_atm_total_rdf.txt')


# In[36]:


df = pd.read_csv('075_bulk_npt_200_ps_298_K_0_atm_total_rdf.txt',
       delim_whitespace=True,header=None,skiprows=0,
       names=['distance','weight'])


# In[37]:


df.index.name = 'bin'


# In[40]:


type(())


# In[41]:


df.shape


# In[45]:


df.distance = np.linspace(0,8,df.shape[0]+1)[:-1]


# In[47]:


df.tail()


# In[6]:


rdf_tot_3_col = np.append( 
    rdf_ref[:,0:2] ,
    np.reshape( rdf_tot[:,1], (len(rdf_tot[:,1]),1)), axis=1)


# In[18]:


plt.plot(rdf_tot_3_col[:,1], rdf_tot_3_col[:,2] )


# In[19]:


np.savetxt('075_bulk_npt_200_ps_298_K_0_atm_total_rdf_3_col.txt',
    rdf_tot_3_col,
    header='# 1: Bin number\n# 2: r\n# 3: g(r)')


# In[20]:


peak_positions["NPT 200ps bulk, average"] = find_peak('075_bulk_npt_200_ps_298_K_0_atm_total_rdf_3_col.txt', peak_range=(3.5,4.5))


# ## 200 ps NPT slab, 298 K
# ### relaxed in xy direction (zero pressure), final configuration

# In[21]:


peak_positions["NPT 200ps slab, final"] = find_peak('070_slab_npt_200_ps_298_K_0_atm_final.txt', peak_range=(3.5,4.5))


# ### relaxed in xy direction (zero pressure), averaged over all 1000 frames

# In[1]:


rdf_ref = np.loadtxt('070_slab_npt_200_ps_298_K_0_atm_final.txt')


# In[23]:


plt.plot(rdf_ref[:,1],rdf_ref[:,2])


# In[24]:


rdf_tot = np.loadtxt('070_slab_npt_200_ps_298_K_0_atm_total_rdf.txt')


# In[25]:


rdf_tot_3_col = np.append( 
    rdf_ref[:,0:2] ,
    np.reshape( rdf_tot[:,1], (len(rdf_tot[:,1]),1)), axis=1)


# In[26]:


plt.plot(rdf_tot_3_col[:,1], rdf_tot_3_col[:,2] )


# In[27]:


np.savetxt('070_slab_npt_200_ps_298_K_0_atm_total_rdf_3_col.txt',
    rdf_tot_3_col,
    header='# 1: Bin number\n# 2: r\n# 3: g(r)')


# In[28]:


peak_positions["NPT 200ps slab, average"] = find_peak('070_slab_npt_200_ps_298_K_0_atm_total_rdf_3_col.txt', peak_range=(3.5,4.5))


# # Summary

# In[29]:


header = 'mean lattice constant (Angstrom)'


# In[30]:


print(
    tabulate( zip(peak_positions.keys(), peak_positions.values()),
            headers  = ('',header),
            tablefmt = 'fancy_grid'))


# ## by box measures:

# In[81]:


cell_measures_bulk = np.loadtxt('075_bulk_npt_200_ps_298_K_0_atm_cell_measures.txt')


# In[82]:


cell_measures_slab = np.loadtxt('070_slab_npt_200_ps_298_K_0_atm_cell_measures.txt')


# In[83]:


cell_measures_slab[:,2] -= 60 # correction for 60 Ang vacuum


# In[84]:


t = np.linspace(0,200.0,len(cell_measures_bulk))


# ## NPT 200 ps, bulk: xox measures

# In[85]:


plt.subplot(131)
plt.plot(t, cell_measures_bulk[:,0], label='x' )
plt.title('x')
plt.xlabel("t (ps)")
plt.ylabel("d ($\AA$)")
plt.subplot(132)
plt.plot(t, cell_measures_bulk[:,1], label='y' )
plt.title('y')
plt.xlabel("t (ps)")
plt.ylabel("d ($\AA$)")
plt.subplot(133)
plt.plot(t, cell_measures_bulk[:,2], label='z' )
plt.title('z')
#plt.legend()
plt.xlabel("t (ps)")
plt.ylabel("d ($\AA$)")


# ## NPT 200 ps, slab: box measures

# In[86]:


plt.plot(t, cell_measures_slab[:,0], label='x' )
plt.plot(t, cell_measures_slab[:,1], label='y' )
plt.legend()
plt.xlabel("t (ps)")
plt.ylabel("d ($\AA$)")


# In[59]:


# cut off initial relaxation:


# In[60]:


mean_measures_bulk = np.mean(cell_measures_bulk[100:,:],axis=0)


# In[61]:


mean_measures_bulk


# In[62]:


mean_measures_slab = np.mean(cell_measures_slab[100:,:],axis=0)


# In[63]:


mean_measures_slab


# In[64]:


# crystal orientation orient = [[1,-1,0], [1,1,-2], [1,1,1]]


# In[65]:


# relation between plane_spacings in this oreintation and lattice constant:
plane_spacing_to_lattice_constant = np.array([np.sqrt(2), np.sqrt(6), np.sqrt(3)] )


# In[66]:


plane_spacing_to_lattice_constant


# In[67]:


# 4.0702 is the bulk lattice constant minimized at 0 K
approximate_crystal_plane_spacing = 4.0702 / plane_spacing_to_lattice_constant


# In[68]:


approximate_crystal_plane_spacing


# In[69]:


# expected number of crystal planes:


# In[70]:


crystal_plane_count = np.round(mean_measures_bulk / approximate_crystal_plane_spacing)


# In[71]:


crystal_plane_count


# In[72]:


exact_crystal_plane_spacing_bulk = mean_measures_bulk / crystal_plane_count


# In[73]:


exact_crystal_plane_spacing_slab = mean_measures_slab / crystal_plane_count


# In[74]:


# deviations in % from 4.07 lattice spacing , bulk:
print(100.0*( exact_crystal_plane_spacing_bulk - approximate_crystal_plane_spacing) / approximate_crystal_plane_spacing)


# In[75]:


# deviations in % from 4.07 lattice spacing, slab:
print(100.0* (exact_crystal_plane_spacing_slab - approximate_crystal_plane_spacing) / approximate_crystal_plane_spacing )


# In[76]:


exact_crystal_plane_spacing_bulk*plane_spacing_to_lattice_constant


# In[77]:


exact_crystal_plane_spacing_slab*plane_spacing_to_lattice_constant


# In[79]:


print(
    tabulate( [
        [
            'NPT 200 ps, bulk', 
            *(exact_crystal_plane_spacing_bulk*plane_spacing_to_lattice_constant)
        ], [
            'NPT 200 ps, slab', 
            *(exact_crystal_plane_spacing_slab*plane_spacing_to_lattice_constant)
        ] ],
        headers  = ('mean anisotropic lattice constants','x','y','z'),
        tablefmt = 'fancy_grid'))

