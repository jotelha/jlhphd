#!/bin/bash
#
# visually compare two LAMMPS logs
#
# Example:
#
#   diff2html.sh no1.log no2.log out.html
#
# preliminary setup

# install npm (https://tecadmin.net/install-latest-nodejs-npm-on-ubuntu/)
#    sudo apt-get remove nodejs npm
#    sudo apt-get update
#    sudo apt-get upgrade
#    curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
#    sudo apt-get install nodejs
#    sudo npm install -g diff2html-cli

# python snippet to query logs from Fireworks filepad and output them to
# temporary files with their metadata attached:
#
#     from fireworks.utilities.filepad import FilePad
#     from tempfile import NamedTemporaryFile
#     import yaml
#
#     query = {
#         'metadata.mode': {'$in':['TRIAL','PRODUCTION'] },
#         'identifier': { '$regex': '.*log\.lammps$'},
#         'metadata.type': 'AFM',
#         'metadata.surfactant':     'SDS',
#         'metadata.sf_nmolecules':  646,
#         'metadata.sf_preassembly': 'monolayer',
#         'metadata.constant_indenter_velocity': -1.0e-4 }
#
#     fp = FilePad(
#         host='localhost',
#         port=27017,
#         database='fireworks',
#         username='fireworks',
#         password='fireworks')
#
#     files = fp.get_file_by_query(query)  # 2 files needed
#
#     n = 80
#     # prepend identifier and metadata to each log:
#     for (cont,doc) in files:
#         with NamedTemporaryFile(mode='r+',delete=False) as f:
#             f.write("#"*n+'\n')
#             f.write("[...]"+doc["identifier"][-n+5:]+'\n')
#             f.write("#"*n+'\n')
#             f.write(yaml.dump(doc["metadata"]))
#             f.write("#"*n+'\n')
#             f.write(cont.decode())
#             tmp_files.append(f.name)
#
# assume two LAMMPS log files and strip thermo output:
cat $1 | sed -e '/^Step/,/^Loop time/c\### [THERMO OUTPUT SKIPPED] ###' > 1.log
cat $2 | sed -e '/^Step/,/^Loop time/c\### [THERMO OUTPUT SKIPPED] ###' > 2.log

# line diff with (in most cases) all COMMON lines as well
diff -u999 1.log 2.log > 12.line.diff
diff2html --style side --input file --output stdout -- 12.line.diff > $3

# word diff not processed correctly:
# git diff --no-index --word-diff -- 1.log 2.log > 12.word.diff