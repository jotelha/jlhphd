import json
from fireworks.utilities.filepad import FilePad

fp = FilePad(
    host='localhost',
    port=27018,
    database='fireworks-jhoermann',
    username='fireworks',
    password='fireworks')

content, doc = fp.get_file(identifier='646_SDS_on_AU_111_51x30x2_hemicylinders_with_counterion_10ns.lammps')
with open('metafile.txt','w') as metafile:
  json.dump( doc, metafile, skipkeys=True, indent=2, default=lambda d: str(d) )
# filepad entry carries 'ObjectID of type 'bson.objectid.ObjectId'
# serializable by simple str conversion
with open('datafile.lammps','wb') as datafile:
  datafile.write(content)
