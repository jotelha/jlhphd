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

# index files from initial DPD equilibration from '2022-02-12-sds-on-au-111-probe-and-substrate-merge-and-approach'
index_file_input_datasets = [{'concentration': 0.5,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'e94f7715-e074-415e-90d6-721c3bfd2212'},
 {'concentration': 0.75,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'ec38bc5b-7031-4024-ad0e-2ff95d5ea18d'},
 {'concentration': 1.0,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'f42fadc3-6e5e-4a4b-b4eb-e8356e94a923'},
 {'concentration': 1.5,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'df59c36e-528d-4047-afc5-f88e98cc7cad'},
 {'concentration': 1.75,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': '418d726f-8118-4e3b-88db-b9b9271dbc8e'},
 {'concentration': 2.0,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'b53f0c2a-04ca-4b36-ae70-b9d751637af3'},
 {'concentration': 2.25,
  'x_shift': 0.0,
  'y_shift': 25.0,
  'uuid': '1135135f-d9b8-4cf9-b34c-4a829287fe08'},
 {'concentration': 2.75,
  'x_shift': 0.0,
  'y_shift': 25.0,
  'uuid': '411f7f22-baf6-48b4-bc11-d5ef4fed4560'},
 {'concentration': 2.25,
  'x_shift': 0.0,
  'y_shift': 0.0,
  'uuid': 'cbc8a45c-87da-4166-9a54-16f9cf91e9f0'}]

# extracted frames from 2022-05-17-sds-on-au-111-probe-and-substrate-approach-frame-extraction
probe_on_substrate_input_datasets = [
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 12.0,
        "uuid": "7a200f45-3b0b-4f9b-97d0-9c013d1423ba"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 14.0,
        "uuid": "19b85ad4-97a9-446d-8ac5-8ee95e61645b"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 45.0,
        "uuid": "57a57ed6-6476-4b39-8207-4317594b6a27"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 18.0,
        "uuid": "512dc4e6-9de3-4b0b-adeb-1d604d625b2c"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 20.0,
        "uuid": "1b4532bc-ba26-467d-9341-c99592870c9a"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 11.0,
        "uuid": "d0c15abb-00ca-4e67-807d-ac0d807c6096"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 19.0,
        "uuid": "45f2a7f2-0e30-47d9-b947-3d935b9885b9"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 50.0,
        "uuid": "bbd79cb0-4b7b-48e5-a58d-f9c0670cae2e"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 30.0,
        "uuid": "9694e227-6290-4252-99cb-3b5cb33fbbb4"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 13.0,
        "uuid": "e3c8aeca-fbb8-40a3-a22d-36470411c754"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 25.0,
        "uuid": "ede010dc-8ead-498f-8460-f7e6187a36c5"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 35.0,
        "uuid": "3b8e7bfa-dd8d-4487-9cdf-2ece49728626"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 17.0,
        "uuid": "7b4dcc56-eb1a-45c5-96a4-76a3c2051d0c"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 40.0,
        "uuid": "4ff0e40b-dc5d-40d2-ad1c-01261631b18b"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 16.0,
        "uuid": "a76f9659-221e-4f63-80e3-c6be0c8b4ed4"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 15.0,
        "uuid": "d7b5d413-fadc-48d5-b3c2-381410909dd0"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 25.0,
        "uuid": "5c95fb3e-da88-4a0e-9c59-672f46f154bd"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 19.0,
        "uuid": "867e8932-343f-45f5-a0b3-32a69ff117d0"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 18.0,
        "uuid": "fc87b819-e9a0-4034-9d01-8ba9f50776b3"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 20.0,
        "uuid": "906b7c6a-95a3-4278-8d41-d3f9955a2da5"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 14.0,
        "uuid": "6f1ada93-acbe-482c-b3cc-0ad6843fa7cd"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 12.0,
        "uuid": "81808f28-36fe-4f7e-ab55-c919618b5d18"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 45.0,
        "uuid": "5307be16-d9f0-4178-a32f-9268755e640d"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 11.0,
        "uuid": "23555dd2-b0a9-4827-882d-95505b2ea18a"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 40.0,
        "uuid": "1391b1cf-d6c6-4b5a-9b4c-7f2fbeadf726"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 35.0,
        "uuid": "e5a81818-0a4c-415c-9f0f-6ff811e4f3b5"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 13.0,
        "uuid": "cd829b6a-c670-488f-b667-675e55a3c27d"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 17.0,
        "uuid": "9a61dd87-b11c-426b-9d8b-3f38ddafc30e"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 30.0,
        "uuid": "93af1606-2684-4150-9c18-1ac0aba8652e"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 15.0,
        "uuid": "1285ac58-c5d4-491c-a26e-aadbda2867fa"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 50.0,
        "uuid": "771cff18-4810-47c8-a433-28c32cc7d57a"
    },
    {
        "nmolecules": 838,
        "concentration": 2.75,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 16.0,
        "uuid": "86f1dd49-c80c-45ae-aee9-682d23cde380"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 14.0,
        "uuid": "57c81a0f-78cd-4fa1-b691-e067f02ce6d4"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 30.0,
        "uuid": "4798407c-833b-493e-8ada-6e2fe85c1cda"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 19.0,
        "uuid": "6fee590e-7957-48ae-8244-5bb24b6bc2c6"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 16.0,
        "uuid": "b0894fb5-834f-45e9-851c-d27ba159fa59"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 12.0,
        "uuid": "fa8a2128-8759-47fc-a064-40b18dc70f4d"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 17.0,
        "uuid": "125e087a-18a2-48e7-b8af-46a68a3fa2ba"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 40.0,
        "uuid": "d6b28ae7-e7a9-4c47-b525-69d5b9859e2c"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 15.0,
        "uuid": "3e3ce03c-452f-4797-9ff1-ca9bb563b212"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 11.0,
        "uuid": "08ed9e55-5cd7-4c7c-97ea-a6d5ef6935dc"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 13.0,
        "uuid": "2a0c9b51-06a6-4bd0-a5c7-ecf5cbd48351"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 25.0,
        "uuid": "082defe6-8829-47ff-80a4-063dcbc8e3b2"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 18.0,
        "uuid": "6f622072-45b6-43b2-9d5a-bed2768bd9dd"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 35.0,
        "uuid": "2c5af19e-62a5-4413-804c-299d19320a0f"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 50.0,
        "uuid": "95c48b04-8148-4c81-9777-b8e49cd3c370"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 20.0,
        "uuid": "253674c1-dcf5-4ba1-8971-4792e9e7d9c9"
    },
    {
        "nmolecules": 681,
        "concentration": 2.25,
        "x_shift": 0.0,
        "y_shift": 25.0,
        "distance": 45.0,
        "uuid": "bba1d7e3-09e2-491c-9ad9-5a50d3facbf2"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 30.0,
        "uuid": "351bc5e2-e8c8-4482-ae04-e9e9ce9e97f3"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 40.0,
        "uuid": "55b54c9c-5675-462a-9ed0-357cbf6ca49a"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 50.0,
        "uuid": "2cd31fc3-f6eb-4b67-882b-a939b3d55282"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 18.0,
        "uuid": "0964200c-c86a-4820-90ec-e57dd9b7e88d"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 45.0,
        "uuid": "4ef641bb-90f6-42ad-9315-c228c9ffe9b0"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 25.0,
        "uuid": "5f5ca16c-2d47-4774-a94a-a0fc29a162ec"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 13.0,
        "uuid": "6a5ab77f-901d-4c58-a0ec-209cb432fe1c"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 16.0,
        "uuid": "7a8a61bb-f2d1-43c0-bfc1-7ba3f8fcb310"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 14.0,
        "uuid": "f709351f-57fb-4bd0-be6b-c1d32b1b2f19"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 15.0,
        "uuid": "80d99cb7-d10c-407f-b0ec-6a799476aaeb"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 35.0,
        "uuid": "c504fe87-2287-433d-87ed-e3c8534ff557"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 17.0,
        "uuid": "996197d8-0b50-4528-8781-610346ec3b5b"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 11.0,
        "uuid": "d9ec84d6-584b-4832-a56a-b1d38d2161c0"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 20.0,
        "uuid": "2b6ed602-0f85-40a9-8717-13196d0c5d00"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 12.0,
        "uuid": "06c4862b-3228-4e77-9acd-b405bafebeb9"
    },
    {
        "nmolecules": 603,
        "concentration": 2.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 19.0,
        "uuid": "7d82dd40-50e7-409f-a2dd-3e9449832322"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 14.0,
        "uuid": "3ab0354d-87dc-4937-aac6-64d2b1a962b5"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 19.0,
        "uuid": "c4a7de30-2610-40f4-9fe7-c439065e0b76"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 18.0,
        "uuid": "90f750b8-7450-4022-b88e-6c13b4538896"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 17.0,
        "uuid": "41dbd8a3-768c-4aa1-8251-737d3da3759b"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 40.0,
        "uuid": "7472c579-abde-45f6-9594-d1d67eaeccf2"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 45.0,
        "uuid": "51c82389-a06f-4ad4-8f57-4058f11e17b2"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 16.0,
        "uuid": "0c1cb68b-5fa6-4e5e-bc13-f93f239875c7"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 13.0,
        "uuid": "c46bef73-8ad4-41c4-acf6-6bb0095c693e"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 12.0,
        "uuid": "28822a88-127f-4366-8770-a5fcc1a8f516"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 25.0,
        "uuid": "f4ef502f-4277-4187-ac1d-70e35669d834"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 20.0,
        "uuid": "77982304-c862-479f-990f-7ac697c01b1a"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 35.0,
        "uuid": "2fbe808c-f53e-439f-93c8-ac385b7b66ca"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 30.0,
        "uuid": "2bd9f160-5540-4fca-8e04-0bb0e6d31440"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 15.0,
        "uuid": "36c19bea-9be4-4022-893f-f47db73872b8"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 50.0,
        "uuid": "57ce9fab-2270-4875-8dbe-bb6efbc98eba"
    },
    {
        "nmolecules": 525,
        "concentration": 1.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 11.0,
        "uuid": "e9a3cc8b-1a46-4b8a-994a-68cfaa36f737"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 12.0,
        "uuid": "fc8b4c09-1c77-490d-bed3-8e24f403dccf"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 45.0,
        "uuid": "78cfb72c-476e-419c-b574-c06f92d75abd"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 15.0,
        "uuid": "88d866ec-1782-495f-9d1d-d6d805eaf162"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 25.0,
        "uuid": "a8c57389-39bf-4608-b34b-5ee6b3e2a881"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 16.0,
        "uuid": "4b8a6970-300a-4853-ba92-43e5ba1cd902"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 30.0,
        "uuid": "00311d72-b3f8-4602-8171-ae0c94855713"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 19.0,
        "uuid": "3818d7d5-2b61-4a97-aa38-dda7179e2259"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 13.0,
        "uuid": "42d9ad9c-fbf4-40b1-b90c-fed530f29515"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 11.0,
        "uuid": "51a8e0c6-1032-4ba9-a987-d1b42d4e753c"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 17.0,
        "uuid": "94936dc1-0a7d-4965-b8bd-7c2402f1526e"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 18.0,
        "uuid": "bdfacf51-247e-4427-8702-4e5aa3f8a074"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 50.0,
        "uuid": "1d93567c-a195-49de-8ed3-041eacc403b2"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 14.0,
        "uuid": "bb4587ba-5564-4f65-aeeb-7f791be522f5"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 20.0,
        "uuid": "91cf1ffe-97d6-4248-8788-13f026fd80e4"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 35.0,
        "uuid": "275d6650-1362-4537-900e-6d080657a830"
    },
    {
        "nmolecules": 447,
        "concentration": 1.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 40.0,
        "uuid": "71269dea-a635-4ca5-986a-a934812fccae"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 15.0,
        "uuid": "2d240436-e283-4dca-8346-6afadff06bac"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 16.0,
        "uuid": "dacee93c-5184-4806-a27e-57248c288470"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 35.0,
        "uuid": "dda668e1-46cb-4972-9275-89e3a34d15a7"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 40.0,
        "uuid": "eec96b69-5862-48b9-bc64-cbed56aefda7"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 18.0,
        "uuid": "9113ab29-fbfd-4441-90ca-15b2431df3ef"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 11.0,
        "uuid": "fcd59181-a7dc-4579-be6a-faf18d539115"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 45.0,
        "uuid": "d342ccde-ab00-4b85-8cf5-3486d01db02e"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 13.0,
        "uuid": "e8e9f1d5-c330-4cca-bd95-0d21cbcae04c"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 30.0,
        "uuid": "6d7b6bb0-fd42-42c5-86ad-97041410f17c"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 20.0,
        "uuid": "94817e0d-6d08-45dc-958d-c2808ce576f8"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 12.0,
        "uuid": "f49db885-4b83-4194-9717-b83b52e1d39b"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 50.0,
        "uuid": "5512e71c-b78c-4b29-91bd-a16becddf02d"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 17.0,
        "uuid": "aacc5001-9703-4af1-9b01-23b7e92f2bbb"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 25.0,
        "uuid": "10131c38-57ea-4796-b2b2-7f46f5d864ec"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 14.0,
        "uuid": "74a20782-a36f-40ac-ac11-4fdb10e62930"
    },
    {
        "nmolecules": 313,
        "concentration": 1.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 19.0,
        "uuid": "7d39eec2-7f77-4b04-9426-ebfbc13af3b7"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 25.0,
        "uuid": "54d338b2-ef63-4d1a-96c2-17c0b0abf22a"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 18.0,
        "uuid": "97b634ec-a2f4-402c-9572-380fe8a953dd"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 16.0,
        "uuid": "88e3abd7-1b96-4f89-acb7-8b3fe9811ae9"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 11.0,
        "uuid": "5da66156-f518-470d-9b46-ad3d5835fb6b"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 17.0,
        "uuid": "598ef624-9c1a-4ba5-af50-3fb5cf7831b7"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 13.0,
        "uuid": "e38b34ad-aad1-4c2a-a059-867358b05f28"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 12.0,
        "uuid": "0750efa1-b64e-494e-8f3f-25d727eeb051"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 45.0,
        "uuid": "dfb5ff49-0a49-4610-9990-8f50251b2dbe"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 40.0,
        "uuid": "35a319ec-3d23-42e1-addb-f6150b64d407"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 20.0,
        "uuid": "2734bb58-ca03-42b8-87a8-c5538fab9ae5"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 30.0,
        "uuid": "3101396b-db2f-49b4-a2bb-e96169e3bf35"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 14.0,
        "uuid": "9d9ccfa1-a1db-430f-bcb8-0bc0b4a86d47"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 50.0,
        "uuid": "c2bb86f6-362e-4ff0-a634-1347ee6f70a9"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 35.0,
        "uuid": "c306f2d1-d1a3-4f9b-821e-3f7cfc089c19"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 19.0,
        "uuid": "c2ed19c8-d424-45cd-aa16-b785b306250d"
    },
    {
        "nmolecules": 235,
        "concentration": 0.75,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 15.0,
        "uuid": "f5ad3629-2bcc-4478-91ee-f0e0f8f56992"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 12.0,
        "uuid": "e5284da3-ef75-478f-84ab-d653fb0a2222"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 40.0,
        "uuid": "ccc2c9d1-b324-431b-b1be-20628b1e1efe"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 20.0,
        "uuid": "0842c675-6309-4347-8600-b9e65c58f9b1"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 16.0,
        "uuid": "79ad62fe-0787-43ea-ac1b-03073dca7bff"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 30.0,
        "uuid": "7f42c426-0442-4594-b878-899706e41cf6"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 18.0,
        "uuid": "839fd749-c04d-42e3-aa71-a3613caab459"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 35.0,
        "uuid": "193b332a-6ae9-43c2-a818-f5b873067309"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 14.0,
        "uuid": "56748752-cdef-44f1-ae49-3cc81ac0c8b1"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 17.0,
        "uuid": "4da09841-f21e-45ba-916d-09c5b325deac"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 11.0,
        "uuid": "7ead218f-7c21-47f2-9fd4-4b380d5a331c"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 25.0,
        "uuid": "c729dddc-e3a1-4550-9da2-bbaba8c5ae5c"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 13.0,
        "uuid": "271eb442-0345-4268-9d04-fefdfc0a5695"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 15.0,
        "uuid": "dacfd945-f411-4df7-8b6a-e91b58b33692"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 19.0,
        "uuid": "0e6c6d7d-ca26-4af9-bfb0-d41617fe744a"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 50.0,
        "uuid": "fb03cebe-e78e-474f-b021-ff13432f628b"
    },
    {
        "nmolecules": 156,
        "concentration": 0.5,
        "x_shift": 0.0,
        "y_shift": 0.0,
        "distance": 45.0,
        "uuid": "2111d7cf-12dd-4b5f-b753-0601b653316d"
    }
]

# In[]:
index_file_input_datasets_index_map = {
    (d["concentration"], d["x_shift"], d["y_shift"]): i for i, d in enumerate(index_file_input_datasets)}

for d in probe_on_substrate_input_datasets:
    d['index_file_uuid'] = index_file_input_datasets[
        index_file_input_datasets_index_map[(d["concentration"], d["x_shift"], d["y_shift"])]]['uuid']

probe_on_substrate_input_datasets_index_map = {
    (d["concentration"], d["x_shift"], d["y_shift"], d["distance"]): i for i, d in enumerate(probe_on_substrate_input_datasets) }

parameter_dict_list = [{**d} for d in probe_on_substrate_input_datasets]
# In[20]:

# SDS on Au(111)
from jlhpy.utilities.wf.probe_on_substrate.chain_wf_probe_on_substrate_lateral_sliding import WrapJoinAndDPDEquilibration

from jlhpy.utilities.wf.mappings import sds_lammps_type_atom_name_mapping
# remove all project files from filepad:
#     fp.delete_file_by_query({'metadata.project': project_id})

# index = probe_on_substrate_input_datasets_index_map[0,0,25.0]
# In[25]:

project_id = '2022-08-24-sds-on-au-111-probe-on-substrate-wrap-join-and-dpd-equilibration'

wf_list = []

for p in parameter_dict_list:
    wfg = WrapJoinAndDPDEquilibration(
        project_id=project_id,

        files_in_info={
            'data_file': {
                'query': {
                    'uuid': p['uuid'],
                    'base_uri':'s3://frct-simdata', # assure to get s3 entry, not outdated ecs
                },
                'file_name': 'default.lammps',
                'metadata_dtool_source_key': 'step_specific',
                'metadata_fw_dest_key': 'metadata->step_specific',
                'metadata_fw_source_key': 'metadata->step_specific',
            },
            'index_file': {
                'query': {
                    'uuid': p['index_file_uuid'],
                    'base_uri':'s3://frct-simdata', # assure to get s3 entry, not outdated ecs
                },
                'file_name': 'groups.ndx',
                'metadata_dtool_source_key': 'system',
                'metadata_fw_dest_key': 'metadata->system',
                'metadata_fw_source_key': 'metadata->system',
            }
        },
        integrate_push=True,
        description="SDS on Au(111) probe on substrate wrap-join and DPD equilibration after frame extraction",
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
            'wrap_join' : {
                'type_name_mapping': sds_lammps_type_atom_name_mapping
            },
            'equilibration': {
                'dpd': {
                    'freeze_substrate_layer': 14.0,  # freeze that slab at the substrate's bottom
                    'rigid_indenter_core_radius': 12.0,  # freeze that sphere at the core of the indenter
                    'temperature': 298.0,
                    'steps': 10000,
                    'netcdf_frequency': 100,
                    'thermo_frequency': 100,
                    'thermo_average_frequency': 100,
                    'restart_frequency': 100,
                    
                    'ewald_accuracy': 1.0e-4,
                    'coulomb_cutoff': 8.0,
                    'neigh_delay': 2,
                    'neigh_every': 1,
                    'neigh_check': True,
                    'skin_distance': 3.0,
                    
                    'max_restarts': 5,
                },
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
    
# In[]
    
# 2022-05-17 submitted all
