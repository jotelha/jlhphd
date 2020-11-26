#!/usr/bin/env python
# coding: utf-8

# # Substrate 150 Ang cube

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[3]:


get_ipython().run_line_magic('config', 'Application.log_level="DEBUG"')
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# In[4]:


import os.path


# In[5]:


from tabulate import tabulate
from pprint import pprint


# In[6]:


from fireworks.utilities.wfb import WorkflowBuilder


# ## NEMO

# In[7]:


prefix= '/mnt/dat/work/testuser/adsorption/N_surfactant_on_substrate_template/fw/specific/substrate/AU/111/150Ang_cube'


# In[8]:


wfb = WorkflowBuilder(os.path.join(prefix,'system_nemo.yaml'))
wfb.template_dir = os.path.join(prefix, 'templates')
wfb.build_dir = os.path.join(prefix, 'build_nemo')
wfb.initialize_template_engine()
try:
    undefined_variables_output = wfb.show_undefined_variables()
except Exception as e:
    print(e)
    error = e
    raise
    
### Conversion to tree with degenerate vertices
wfb.descend()
wfb.build_degenerate_graph()
#wfb.plot()
show_attributes_output = wfb.show_attributes()
## Build Workflow
try:
    wfb.fill_templates()
except Exception as e:
    print(e)
    error = e
    raise

try:
    wf = wfb.compile_workflow()
except Exception as e:
    print(e)
    error = e
    raise


# In[9]:


wfb.plot()


# In[ ]:




