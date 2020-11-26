#!/usr/bin/env python
# coding: utf-8

# ## Build Workflow from Fireworks yaml files

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', 'Application.log_level="DEBUG"')


# In[1824]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# In[3]:


from tabulate import tabulate
from pprint import pprint


# In[4]:


from fireworks.utilities.wfb import WorkflowBuilder


# # Build workflow from templates

# In[2074]:


sorted( set([1,5,2,9]))


# In[2075]:


wfb = WorkflowBuilder('system.yaml')
wfb.initialize_template_engine()
print(wfb.show_undefined_variables())
### Conversion to tree with degenerate vertices
wfb.descend()
wfb.build_degenerate_graph()
wfb.plot()
print(wfb.show_attributes())
## Build Workflow
wfb.fill_templates()
wf = wfb.compile_workflow()


# In[2056]:


wfb = WorkflowBuilder('system.yaml')


# In[2057]:


wfb.initialize_template_engine()


# In[2058]:


print(wfb.show_undefined_variables())


# In[2059]:


wfb.plot()


# ### Conversion to tree with degenerate vertices

# In[2060]:


wfb.descend()


# In[2061]:


wfb.plot()


# In[2062]:


wfb.initialize_template_engine()


# In[2063]:


h = wfb.build_degenerate_graph()


# In[2064]:


wfb.plot(wfb.h)


# In[2065]:


print(wfb.show_attributes())


# ## Build Workflow

# In[2066]:


wfb.g.vs[0]["transient"]


# In[2069]:


wfb.fill_templates()


# In[2070]:


wfb.compile_workflow()


# In[1945]:


pp = PrettyPrinter(indent=2)


# In[1946]:


fws_yaml = sorted(glob("fw_*.yaml"))


# In[1947]:


print(tabulate( [ [ row ] for row in fws_yaml ] ,headers=["Files"],tablefmt='simple'))


# In[1983]:


print(wfb.show_attributes(exclude=["name"]))


# In[40]:


pp.pprint(metadata)


# In[41]:


assert "metadata" in metadata


# In[42]:


assert "name" in metadata


# In[43]:


fws_id = { fw: (-10*(i+1)) for i, fw in enumerate(fws_set) }


# In[1948]:


pp.pprint(fws_id)


# In[20]:


fws_dict = { fw: Firework.from_file(fw, f_format='yaml') for fw in fws_set }


# In[21]:


for name, fw in fws_dict.items():
    fw.fw_id = fws_id[name]


# In[22]:


links = { 
    fws_dict[parent]: [ 
        fws_dict[child] for child in children ] for parent, children in dependencies.items() }


# In[23]:


fws_list = list(fws_dict.values())


# In[24]:


wf = Workflow( fws_list, links, name=metadata["name"], metadata=metadata["metadata"]  )


# In[25]:


wf.to_file("wf.yaml",f_format='yaml')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




