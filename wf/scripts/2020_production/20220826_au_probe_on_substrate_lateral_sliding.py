# In[20]:
import os, os.path
import datetime
# FireWorks functionality
from fireworks.utilities.dagflow import DAGFlow, plot_wf
from fireworks import Firework, LaunchPad, Workflow
from fireworks.utilities.filepad import FilePad


timestamp = datetime.datetime.now()
yyyymmdd = timestamp.strftime('%Y%m%d')
yyyy_mm_dd = timestamp.strftime('%Y-%m-%d')

# plotting defaults
visual_style = {}

# generic plotting defaults
visual_style["layout"] = 'kamada_kawai'
visual_style["bbox"] = (1600, 1200)
visual_style["margin"] = [400, 100, 400, 200]

visual_style["vertex_label_angle"] = -3.14/4.0
visual_style["vertex_size"] = 8
visual_style["vertex_shape"] = 'rectangle'
visual_style["vertex_label_size"] = 10
visual_style["vertex_label_dist"] = 4

# edge defaults
visual_style["edge_color"] = 'black'
visual_style["edge_width"] = 1
visual_style["edge_arrow_size"] = 1
visual_style["edge_arrow_width"] = 1
visual_style["edge_label_size"] = 8

# In[22]:

# prefix = '/mnt/dat/work/testuser/indenter/sandbox/20191110_packmol'
prefix = '/home/jotelha/git/jlhphd'
work_prefix = '/home/jotelha/tmp/{date:s}_fw/'.format(date=yyyymmdd)
try:
    os.makedirs(work_prefix)
except:
    pass
os.chdir(work_prefix)

# the FireWorks LaunchPad
lp = LaunchPad.auto_load() #Define the server and database
# FilePad behaves analogous to LaunchPad
fp = FilePad.auto_load()

# In[]:

# from '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach' probe on monolayer approach at 1 m / s
index_file_input_datasets = [
 {'concentration': 0.5,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '9244b291-40d0-4e9f-b985-6b2928d7b53c'},
 {'concentration': 0.75,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'c120adb1-4456-4be8-bcb8-fa6208f688a5'},
 {'concentration': 1.0,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'b3def621-636e-4d5c-a114-9aa3c7162272'},
 {'concentration': 1.5,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '64d57620-d6ff-4e39-bf3c-c7cdac01f656'},
 {'concentration': 1.75,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '24e8c4ac-7f16-4d97-8400-e4154bbded9a'},
 {'concentration': 2.0,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '5e844a04-fb3b-416e-8bce-316bc29421a0'},
 {'concentration': 2.25,
  'x_shift': 0.0,
  'y_shift': 25.0,
  'uuid': '1a5146ba-37d4-47d3-8fce-e22beca2c5a1'},
 {'concentration': 2.75,
  'x_shift': 0.0,
  'y_shift': 25.0,
  'uuid': 'fa39688d-ff02-4a39-85c0-ce1bf7cf73d3'},
 {'concentration': 2.25,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '7c16ea40-1d92-43c3-a807-254560078e2b'}
]
 
# from '2022-08-24-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration'
probe_on_substrate_input_datasets = [
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 45.0,
   "uuid": "ea9777b9-808c-4807-a98f-aec177ea441b"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 50.0,
   "uuid": "9736f3d1-6b0f-4284-a230-0b31540b1608"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 19.0,
   "uuid": "48ace7dc-0068-45e7-941a-f1f4224cff1a"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 15.0,
   "uuid": "b9316f97-f7ce-4ddc-a0f0-5629859e369b"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 13.0,
   "uuid": "8f372739-3135-4a2b-a681-de26c7b9dead"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 25.0,
   "uuid": "9353a9cb-73b0-41bc-91e8-53504610f537"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 11.0,
   "uuid": "5c13cb1b-cec6-48c5-8407-2b549f7ef21b"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 17.0,
   "uuid": "f53c314c-18f5-46ad-ab0e-f4f5c160c2c8"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 14.0,
   "uuid": "b7858482-65f7-4c75-9a71-58fe44f8ebc7"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 35.0,
   "uuid": "e312ef6d-cea1-405a-8f35-bcb2644fa989"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 18.0,
   "uuid": "444e0e33-ae91-4f0a-a56f-729b8925ddee"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 30.0,
   "uuid": "de3172d3-a1d4-441c-9512-0be8494c632e"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 20.0,
   "uuid": "7b3933a4-76d3-4e40-8e81-ccf2496079c2"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 40.0,
   "uuid": "4bf4d18a-5f23-4757-a536-f09cfd35bc05"
  },
  {
   "nmolecules": 156,
   "concentration": 0.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 12.0,
   "uuid": "3352f8ef-a5e8-44d7-8107-3e937a76947c"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 15.0,
   "uuid": "b8f3e550-633b-4ae3-9f2f-64d1792ea9e5"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 19.0,
   "uuid": "7e2294e8-ec6b-4ccf-8a80-84446a19946b"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 35.0,
   "uuid": "f65ae2d0-664b-4232-b731-562d08ffdd2b"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 50.0,
   "uuid": "0a9e1c85-1ee9-4b7f-aca9-eb7acff4007f"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 14.0,
   "uuid": "e0624dcc-1f13-4598-82e9-28e133c7942c"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 30.0,
   "uuid": "e6d3e698-b0d8-406e-b2fc-7d813d085903"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 20.0,
   "uuid": "024d18e9-3834-4aa4-905d-f9a7c34794d0"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 40.0,
   "uuid": "16b938ab-5a53-4611-b6d0-fa85a7bf7743"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 45.0,
   "uuid": "ceb64760-a0f6-4165-824d-ac8ccea8fb0f"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 12.0,
   "uuid": "5d3e888b-4f62-471a-b6ab-11cebc159754"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 13.0,
   "uuid": "37689e62-e32d-43f2-a487-c867bc73da92"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 17.0,
   "uuid": "e4d594f8-67cd-4ef9-a55b-fa2d6ebe39c9"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 11.0,
   "uuid": "a3536116-1dca-4407-b678-216ef7922174"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 16.0,
   "uuid": "e49b57d6-8650-43c8-a605-806f77177b5d"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 18.0,
   "uuid": "b2225970-9f1c-4be2-aab8-ec24ca031811"
  },
  {
   "nmolecules": 235,
   "concentration": 0.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 25.0,
   "uuid": "1aa5cc2b-5cea-47ec-814b-eee9be60f5ea"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 19.0,
   "uuid": "d2c4078d-8c7e-4193-a192-ae33240faec0"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 14.0,
   "uuid": "982c8b0b-88af-4253-a3e2-e53e1e6db9e1"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 25.0,
   "uuid": "4edfb7c9-724c-4ee8-a3c3-6d1152a1ac10"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 17.0,
   "uuid": "661912de-f7d6-4c3e-8eb4-36f1be2d54cf"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 50.0,
   "uuid": "36f00d75-8f39-481b-a896-f92926958cd3"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 12.0,
   "uuid": "8bda8a84-381b-47c2-997f-fa4e3cd00302"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 20.0,
   "uuid": "c47b167b-cdfb-4867-a6a9-c6f622a8e7ca"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 30.0,
   "uuid": "4b1c0fd3-0531-42db-98e5-b703aacdb461"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 13.0,
   "uuid": "2a5f91ef-0681-41db-87f0-4971ddcf1a98"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 45.0,
   "uuid": "12c64be3-3b37-4b9c-bdc9-7796e2989d40"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 11.0,
   "uuid": "19816bc7-1732-4212-9dbc-6b38f766df0f"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 18.0,
   "uuid": "6f657ab4-378b-4ea2-b4dd-540bc6ac9137"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 40.0,
   "uuid": "4d859891-0779-4589-bc69-4a90376cbd3c"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 35.0,
   "uuid": "5f46a7af-8cdb-49bc-8a0f-3790bfcef640"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 16.0,
   "uuid": "26c07abf-2434-4bf0-9212-5c4b4a441bde"
  },
  {
   "nmolecules": 313,
   "concentration": 1.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 15.0,
   "uuid": "f0e584c4-8343-4ab4-ba27-987480fd624f"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 40.0,
   "uuid": "800b0602-2fd5-46cb-9c56-b6b5152e2be2"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 35.0,
   "uuid": "384c6dfd-8da6-438c-b242-8c18911de4b5"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 20.0,
   "uuid": "fdb386c0-4c65-4530-9221-fd4f0c055e57"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 14.0,
   "uuid": "9134b7dd-a9cc-4fdc-a4a7-af386104701e"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 50.0,
   "uuid": "ca5ba926-d604-42dc-a3a4-e7fc9f286e66"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 18.0,
   "uuid": "0255b6d1-9b61-4ec5-b1b6-9be46775de95"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 17.0,
   "uuid": "f66d2eb3-6761-46ea-a071-dbced6baff0a"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 11.0,
   "uuid": "d4a54690-e40e-4a43-92bd-438a0657d8c3"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 13.0,
   "uuid": "31798623-51e7-4ada-aad9-949b6b969d9e"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 19.0,
   "uuid": "e14d95e1-ddca-451e-b375-6cc0afe86c9b"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 30.0,
   "uuid": "c4ea415e-3674-4d6d-a21c-b478eb0c3cb5"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 16.0,
   "uuid": "a91b02f8-7629-44a0-884a-8b03291db623"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 25.0,
   "uuid": "3e853e5e-b7c6-47a1-93cc-76f53c9747b8"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 15.0,
   "uuid": "47ac643d-e8f9-4c2a-a32c-f6756e7eaf66"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 45.0,
   "uuid": "0d442f03-6937-4a62-935e-d0c088f55cf6"
  },
  {
   "nmolecules": 447,
   "concentration": 1.5,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 12.0,
   "uuid": "0dbaeb06-c166-44c9-9193-325f29d3fda9"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 11.0,
   "uuid": "eac32e78-6926-405d-9f4c-4b2e82780042"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 50.0,
   "uuid": "b3baa5ef-a176-4d99-8b78-310e3ff48898"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 15.0,
   "uuid": "219a20de-5b6d-4122-b2e1-026d4eb2581e"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 30.0,
   "uuid": "4f7b1d1b-cc10-443a-9828-7fe856bc90b1"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 35.0,
   "uuid": "230d32f0-ae60-4603-99fa-5de79edbf19b"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 20.0,
   "uuid": "83f2b523-da20-4221-98b3-dc494a5895dd"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 16.0,
   "uuid": "c5ae92cb-5cd2-4ab2-beca-0cd27d9e9d22"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 45.0,
   "uuid": "801072ea-c879-4466-ad53-781561ab84a4"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 40.0,
   "uuid": "bdd0354e-9dc5-44a9-91c4-8802f469dbc2"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 17.0,
   "uuid": "3b7fb047-364e-410a-8180-6aad9667ea9d"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 18.0,
   "uuid": "dc0cf7ec-1caf-4f15-8b28-e3b7a1a2165a"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 19.0,
   "uuid": "75225ad4-a660-4fb4-9f7d-d1da6ac18456"
  },
  {
   "nmolecules": 525,
   "concentration": 1.75,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 14.0,
   "uuid": "7a45dfc8-7d20-4dc2-befa-c4159fb47ee7"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 19.0,
   "uuid": "497a16ca-6d0c-40d2-9ed6-eb54d02c0d09"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 12.0,
   "uuid": "b27042d4-0c93-4eef-9092-1582887b81b7"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 20.0,
   "uuid": "6a0e0b64-3405-4f4f-884b-5de8e1bba0f7"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 11.0,
   "uuid": "e36e1108-68b8-4e99-b1e3-3a545ef8acaa"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 17.0,
   "uuid": "3e9423b7-f24d-4b84-b71a-595a6eccac93"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 35.0,
   "uuid": "71b3580b-fc87-4fe3-b072-08e5788cd46f"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 15.0,
   "uuid": "13438ddf-3520-4017-ab05-0c3f503734bb"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 14.0,
   "uuid": "4ed34ae5-a4c1-48a1-aac2-27b59a98894a"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 16.0,
   "uuid": "ebfee7e7-e98f-4a2b-8105-e3dd08e112d1"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 13.0,
   "uuid": "61903e65-7e26-43f6-a7ec-dee8bb3a0f94"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 25.0,
   "uuid": "8e6f7bf6-4e35-4567-af4c-9488aef95a78"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 45.0,
   "uuid": "60d4d10c-ad5d-41fb-8353-3b5d0eee278b"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 18.0,
   "uuid": "5d2dd10d-9cdb-4224-a5fa-d7545faaff78"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 50.0,
   "uuid": "4f4eb184-8a50-455f-98f4-f8cc241c0bed"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 40.0,
   "uuid": "5a16a3a0-1d45-4609-bf4f-cf49df09515e"
  },
  {
   "nmolecules": 603,
   "concentration": 2.0,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 30.0,
   "uuid": "38d4f50b-83d3-460a-ab29-d2c56da94bea"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 45.0,
   "uuid": "d14a6382-0d7b-4dce-8fe1-b4df15ebf134"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 20.0,
   "uuid": "fdcdd015-82ef-4f11-8ca7-8a39f2192c6f"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 50.0,
   "uuid": "46324eca-1843-4b20-bd34-772c26ca9436"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 35.0,
   "uuid": "8db905c0-87c6-4486-a28e-14e0e2e8cd52"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 18.0,
   "uuid": "c1a92ab9-b882-420f-8bc9-d1666c4e30bc"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 25.0,
   "uuid": "0a0fe6ee-940a-4127-8a70-a44fca42d6e4"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 13.0,
   "uuid": "19c012a4-ca3e-470f-80a4-3fb330f4155a"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 11.0,
   "uuid": "cad6b9d2-f7ad-4afb-b025-96432e870414"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 15.0,
   "uuid": "46db6f74-9113-476f-9f74-c2f474ffef4a"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 40.0,
   "uuid": "9ee84978-1e3c-4d3e-a1ac-cfedc9641c5e"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 17.0,
   "uuid": "6ae04703-0fe1-47a2-8eed-d3917ae0ecfa"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 12.0,
   "uuid": "a0a4661a-6377-48cf-bbf9-8a9686c9cc0c"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 16.0,
   "uuid": "2aa7bab6-bb6d-401c-bc38-5a413d505ef0"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 19.0,
   "uuid": "85403cf0-902f-4603-95b0-ad59b799ca73"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 30.0,
   "uuid": "1901400f-4659-42b7-8010-343642b1b38b"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 14.0,
   "uuid": "b720947e-2d49-451b-9d3f-f8af4f9ea70b"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 16.0,
   "uuid": "1e6d1d82-0992-4d36-814a-f8243053e3d1"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 50.0,
   "uuid": "4d654820-139f-4b47-a689-d7a64f567b86"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 15.0,
   "uuid": "f93143fc-8ad2-475a-af40-308400e913ef"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 30.0,
   "uuid": "91bb7240-986b-4286-8574-efc445e665e8"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 17.0,
   "uuid": "c6d9a00f-5149-4844-b1cb-e8c60164fa38"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 13.0,
   "uuid": "a4686bff-1278-48a2-987b-fb3b82edd38a"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 35.0,
   "uuid": "27b5bfee-cbd0-4d35-bfd9-94cf6fc523b1"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 40.0,
   "uuid": "ac3ba48f-0d76-4afa-baf6-5bbe38adb1a5"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 11.0,
   "uuid": "e289ab01-b59d-4c33-9c56-8cdb88aaa3e4"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 45.0,
   "uuid": "bbf03527-b699-43e4-aaab-72b3d8900c59"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 12.0,
   "uuid": "c0fdee59-f59c-4175-8ef7-d0f0b19cd037"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 14.0,
   "uuid": "cca1d08f-29da-4d86-a66c-0fb484a5bbd2"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 20.0,
   "uuid": "4e740d9b-da06-4f5d-802c-68afb0ee1d70"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 18.0,
   "uuid": "387fd449-b09e-4a1b-878d-96993697ce9e"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 19.0,
   "uuid": "f8e364cd-a09b-4e07-9468-5a536f1a83d8"
  },
  {
   "nmolecules": 838,
   "concentration": 2.75,
   "x_shift": 0.0,
   "y_shift": 25.0,
   "distance": 25.0,
   "uuid": "6097e4ab-f301-4d3f-a99a-d698ebfcb0b3"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 15.0,
   "uuid": "d2f71d69-7705-499e-902f-f9bfe7997fd8"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 16.0,
   "uuid": "547453d2-f226-4cf2-9349-7e2e139f90ce"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 40.0,
   "uuid": "7dc98a42-6753-46f1-bbf8-500a75e7002b"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 17.0,
   "uuid": "c52da397-a604-44e1-ad63-64079448f8d2"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 35.0,
   "uuid": "b0af8ac9-5f2a-4d10-87a4-c4fbc93c7282"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 25.0,
   "uuid": "865705c8-2d11-4724-953c-b67673895e61"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 13.0,
   "uuid": "e9747e97-f09f-48e1-8161-a5c6df9794b0"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 30.0,
   "uuid": "eb822cf6-2178-4a45-9fd6-c0f0385fbcdc"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 50.0,
   "uuid": "1d1a92da-1deb-4fac-a4a7-c44a9f9094fa"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 19.0,
   "uuid": "66231659-16a9-424c-80d7-fd210e3d760e"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 11.0,
   "uuid": "9a7a245f-64ab-4a3f-b256-65cbf8b0c436"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 20.0,
   "uuid": "4d3e8a56-21d3-42bb-bb59-e1846352b81f"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 18.0,
   "uuid": "7ee60594-d0bb-44a3-8bdb-2e27d5ac78d9"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 45.0,
   "uuid": "493478fd-0259-4a1f-975f-1b451e0fba2b"
  },
  {
   "nmolecules": 681,
   "concentration": 2.25,
   "x_shift": 0.0,
   "y_shift": 0.0,
   "distance": 12.0,
   "uuid": "62c302fa-26f0-48d6-b1f8-28434d8a7515"
  }
]

index_file_input_datasets_index_map = { (d["concentration"], d["x_shift"], d["y_shift"]): i for i, d in enumerate(index_file_input_datasets) }

for d in probe_on_substrate_input_datasets:
    d['index_file_uuid'] = index_file_input_datasets[index_file_input_datasets_index_map[(d["concentration"],d["x_shift"], d["y_shift"])]]['uuid']
    
probe_on_substrate_input_datasets_index_map = {(d["nmolecules"], d["distance"], d["x_shift"], d["y_shift"]): i for i, d in enumerate(probe_on_substrate_input_datasets)}
# In[29]
# parameters

parameter_sets = [
    {
        'direction_of_linear_movement': d,
        'constant_indenter_velocity': -1.0e-5, # 1 m / s
        'steps': 1500000, # 3 nm sliding
        'netcdf_frequency': 1000,
        'thermo_frequency': 1000,
        'thermo_average_frequency': 1000,
        'restart_frequency': 1000,
    } for d in range(0,2)
]

parameter_dict_list = [{**d, **p} for p in parameter_sets for d in probe_on_substrate_input_datasets]

#parameter_dict_list = [{**d, **p} for p in parameter_sets for d in probe_on_substrate_input_datasets 
#                       if p['direction_of_linear_movement'] == 0 and d['distance'] < 20]
# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_lateral_sliding import ProbeOnSubstrateLateralSliding

# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# index = probe_on_substrate_input_datasets_index_map[0,0,25.0]
# In[25]:
    
project_id = '2022-08-26-sds-on-au-111-probe-on-substrate-lateral-sliding'

wf_list = []
# for c, substrate_uuid, probe_uuid in probe_on_substrate_input_datasets:
# c = 0.03
for p in parameter_dict_list:
    wfg = ProbeOnSubstrateLateralSliding(
        project_id=project_id,
        files_in_info={
            'data_file': {
                'query': {'uuid': p['uuid']},
                'file_name': 'default.lammps',
                'metadata_dtool_source_key': 'step_specific',
                'metadata_fw_dest_key': 'metadata->step_specific',
                'metadata_fw_source_key': 'metadata->step_specific',
            },
            'index_file': {
                'query': {'uuid': p['index_file_uuid']},
                'file_name': 'groups.ndx',
                'metadata_dtool_source_key': 'system',
                'metadata_fw_dest_key': 'metadata->system',
                'metadata_fw_source_key': 'metadata->system',
            }
        },
        integrate_push=True,
        description="SDS on Au(111) probe on substrate lateral sliding",
        owners=[{
            'name': 'Johannes Laurin Hörmann',
            'email': 'johannes.hoermann@imtek.uni-freiburg.de',
            'username': 'fr_jh1130',
            'orcid': '0000-0001-5867-695X'
        }],
        infile_prefix=prefix,
        machine='juwels',
        mode='production',
        system = {},
        step_specific={
            'probe_lateral_sliding': {
                'constant_indenter_velocity': p['constant_indenter_velocity'],
                'direction_of_linear_movement': p['direction_of_linear_movement'],
                'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the ore of the indenter
                'temperature': 298.0,
                'steps': p['steps'],
                'netcdf_frequency': p['netcdf_frequency'],
                'thermo_frequency': p['thermo_frequency'],
                'thermo_average_frequency': p['thermo_average_frequency'],
                'restart_frequency': p['restart_frequency'],
                
                'ewald_accuracy': 1.0e-4,
                'coulomb_cutoff': 8.0,
                'neigh_delay': 2,
                'neigh_every': 1,
                'neigh_check': True,
                'skin_distance': 3.0,
                
                'max_restarts': 100,
            },
            'filter_netcdf': {
                'group': 'indenter',
            },
            'dtool_push': {
                'dtool_target': f'/p/project/hfr21/hoermann4/dtool/PRODUCTION/{project_id}',
                'remote_dataset': None,
            }
        }
    )
    fp_files = wfg.push_infiles(fp)
    wf = wfg.build_wf()
    wf_list.append(wf)

    
# In[]:

# dump gernerated workflows to file

for i, wf in enumerate(wf_list):
    wf.to_file("wf_{:03d}.json".format(i), indent=4)
    
    
# In[]:
# 2022-08-26 all 278 queued