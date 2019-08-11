// get all fw ids win wflow slected by an arbitrary fw id:

db.workflows.find({'nodes':{'$in':[15629]}},{'nodes':true})
db.workflows.find({'nodes':{'$in':[15629]}},{'_id':false,'nodes':true})

// query in one line from bash, extract only fw_ids:
mongo --quiet --port 27018 --authenticationDatabase fireworks-jhoermann \
    --username fireworks --password fireworks fireworks-jhoermann \
    --eval 'db.workflows.find({"nodes":{"$in":[15629]}},{"_id":false,"nodes":true})' \
  | sed -E 's/^\{.*\[([^]]*)\].*\}$/\1/' | sed 's/,//g'

mongo --quiet --port 27018 --authenticationDatabase fireworks-jhoermann \
    --username fireworks --password fireworks fireworks-jhoermann \
    --eval 'db.workflows.find({"nodes":{"$in":[15629]}},{"_id":false,"nodes":true})' \
  | sed -E 's/^\{.*\[([^]]*)\].*\}$/\1/' | xrags --deleimiter=,

// get ids of reserved fireworks in certain workflow
lpad get_fws --query '{"fw_id": {"$in": '"$(mongo --quiet --port 27018 --authenticationDatabase fireworks-jhoermann \
    --username fireworks --password fireworks fireworks-jhoermann \
    --eval 'db.workflows.find({"nodes":{"$in":[15629]}},{"_id":false,"nodes":true})' \
  | sed -E 's/^\{.*(\[[^]]*\]).*\}$/\1/')"' }, "state": "RESERVED" , "spec.metadata.step":"minimization"}' -d ids

// apply these ids to a command:
lpad get_fws --query '{"fw_id": {"$in": '"$(mongo --quiet --port 27018 --authenticationDatabase fireworks-jhoermann \
    --username fireworks --password fireworks fireworks-jhoermann \
    --eval 'db.workflows.find({"nodes":{"$in":[15629]}},{"_id":false,"nodes":true})' \
  | sed -E 's/^\{.*(\[[^]]*\]).*\}$/\1/')"' }, "state": "RESERVED" , "spec.metadata.step":"minimization"}' \
  -d ids | tr -d [], | xargs -n 1 echo

// execute for each id
for i in $( mongo --quiet --port 27018 --authenticationDatabase fireworks-jhoermann \
    --username fireworks --password fireworks fireworks-jhoermann \
    --eval 'db.workflows.find({"nodes":{"$in":[15629]}},{"_id":false,"nodes":true})' \
  | sed -E 's/^\{.*\[([^]]*)\].*\}$/\1/' | sed 's/,//g' )
do
  echo "Arg $i"
done

// find document
db.filepad.find(
  {
    identifier: {
      $regex: "substrate.*"
    },
    "metadata.sb_name": "AU_111_150Ang_cube"
  }
)

// find document
db.filepad.find(
  {'metadata.counterion': 'NA',
   'metadata.indenter.initial_radius': 25,
   'metadata.indenter.initial_radius_unit': 'Ang',
   'metadata.indenter.substrate': 'AU',
   'metadata.mode': 'PRODUCTION',
   'metadata.sb_base_length': 150,
   'metadata.sb_crystal_plane': 111,
   'metadata.sb_shape': 'cube',
   'metadata.solvent': 'H2O',
   'metadata.step': 'initial_config',
   'metadata.substrate': 'AU',
   'metadata.surfactant': 'SDS',
   'metadata.type': 'AFM'}
)

// update field
db.filepad.update(
  {'metadata.counterion': 'NA',
   'metadata.indenter.initial_radius': 25,
   'metadata.indenter.initial_radius_unit': 'Ang',
   'metadata.indenter.substrate': 'AU',
   'metadata.mode': 'PRODUCTION',
   'metadata.sb_base_length': 150,
   'metadata.sb_crystal_plane': 111,
   'metadata.sb_shape': 'cube',
   'metadata.solvent': 'H2O',
   'metadata.step': 'initial_config',
   'metadata.substrate': 'AU',
   'metadata.surfactant': 'SDS',
   'metadata.type': 'AFM'}, {
    $set: {
      "metadata.sb_in_dist": 30.0,
      "metadata.sb_in_dist_unit": "Ang" }
  }, {
    multi: true // update all matches
  }
)

db.filepad.update(
  { "metadata.counterion": "NA",
    "metadata.sb_base_length": 184,
    "metadata.sb_crystal_plane": 111,
    "metadata.sb_shape": "cube",
    "metadata.solvent": "H2O",
    "metadata.step": "initial_config",
    "metadata.substrate": "AU",
    "metadata.surfactant": "CTAB",
    "metadata.type": "interface"  }, {
    $set: {
      "metadata.counterion": "BR" }
  }, {
    multi: true // update all matches
  }
)

// update field
db.filepad.update(
  {
    identifier: {
      $regex: "substrate.*"
    },
    "metadata.sb_name": "AU_111_150Ang_cube"
  }, {
    $set: { "metadata.type": "substrate" }
  }, {
    multi: true // update all matches
  }
)

// not yet tested: update by aggregation pipeline:
db.runCommand(
  {
    update: "filepad",
    updates: [
      {
        q:  { },
        u: [
          {
            $match: {
              "identifier": {
                "$regex": "substrate.*"
                },
                "metadata.sb_name": "AU_111_150Ang_cube"
              }
          }, {
            $addFields: {
              "metadata.type": "substrate"
            }
          }
        ]
      }
    ],
    ordered: false,
    writeConcern: { w: "majority", wtimeout: 5000 }
  }
)
