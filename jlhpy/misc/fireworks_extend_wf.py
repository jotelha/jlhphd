# a sample on how to utilize FireWorks (and underlying pymongo) to
# modify existing workflows by appending matching partial worklows
# from a local Workflow yaml file
from fireworks.core.firework import Workflow
from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.wfb import extract_partial_workflow
import logging

logging.basicConfig(level=logging.INFO)

general_production_query = {
  "spec.metadata.mode":"PRODUCTION",
  "spec.metadata.step":"approach",
  "spec.metadata.surfactant":"SDS",
  "spec.metadata.sf_nmolecules": { "$gt": 0 },
  "spec.metadata.constant_indenter_velocity": { "$in": [-1.0e-4, -1.0e-5] },
  "name":{"$regex":".*production.*"},
  "state": "COMPLETED" }
general_postp_query = {
  "spec.metadata.mode":"PRODUCTION",
  "spec.metadata.step":"approach",
  "spec.metadata.surfactant":"SDS",
  "name":{"$regex":".*immediate post-processing"},
  "state": "COMPLETED" }

sweep_param_keys = [
  "spec.metadata.constant_indenter_velocity",
  "spec.metadata.sf_nmolecules",
  "spec.metadata.sf_preassembly" ]

step_to_append = "frame extraction"


logger = logging.getLogger(__name__)

def is_subset(sub, ref):
    """Checks whether dictionary sub is fully contained in ref"""
    for key, value in sub.items():
        if key in ref:
            if isinstance(sub[key], dict):
                if not is_subset(sub[key], ref[key]):
                    return False
            elif value != ref[key]:
                return False
        else:
            return False
    return True

def nested2plain(d, separator='.', prefix=''):
    """Converts nested dictionary to plain dictionary with keys concatenated.

    Args:
      d (dict)
      separator (str) : symbol or string used as seperator between nested keys
      prefix (str):     leave empty

    Returns:
      dict { str: obj }
    """
    r = {}
    assert isinstance(prefix,str)
    assert isinstance(separator,str)
    if prefix is not '':
        prefix = prefix + separator
    for k, v in d.items():
        if type(v) is dict:
            r.update( nested2plain(
                v, separator=separator, prefix=prefix + k) )
        else:
            r.update( {prefix + k: v} )
    return r

lp = LaunchPad.auto_load()

projection = dict({"_id": False}, **{ key: True for key in sweep_param_keys })

logger.info( "Primary query:   {}".format(general_production_query))
logger.info( "Projection:      {}".format(projection))

sweep_params_cursor = lp.fireworks.find( general_production_query, projection )

sweep_params = [ doc for doc in sweep_params_cursor ]
logger.info( "#Parameter sets: {}".format(len(sweep_params)))
logger.debug("Parameter sets:  {}".format(sweep_params))

# the uniqueness of parameter sets is not guaranteed here, thus check:
for i, p in enumerate(sweep_params):
    for j, q in enumerate(sweep_params[i+1:]):
        assert not is_subset(p,q) and not is_subset(q,p), \
            "Parameter sets {} and {} overlap!".format(p,q)

# construct mongo langage query from nested parameter sets
param_queries = [ nested2plain(param_set) for param_set in sweep_params ]
logger.debug("Param. queries:  {}".format(param_queries))

specific_production_queries = [ {**general_production_query, **param_query } for param_query in param_queries ]
logger.debug("Specific prod. fw queries:   {}".format(specific_production_queries))

production_fw_ids = [ lp.get_fw_ids(production_query) for production_query in specific_production_queries ]
unique_prod_fw_ids  = [ fw_ids[0] if len( fw_ids ) == 1 else None for fw_ids in production_fw_ids ]
logger.info("Unique prod. fw ids:  {}".format(unique_prod_fw_ids))

# find postprocessing steps in same workflow as completed production steps
# querying nodes directly instead of get_wf_by_fw_id much more perfomant
ref_wf_fws_list = []
for i, fw_id in enumerate(unique_prod_fw_ids):
    if fw_id:
        logger.debug("Querying associated WF for fw id no. {}: {}.".format(i, fw_id))
        res_cursor = lp.workflows.find(
            { "nodes": { "$in": [ fw_id ] } }, # in this case, matches if one element in nodes is equal to fw_id
            {"_id": False,"nodes": True} )
        assert res_cursor.count() == 1, "Found {} > 1 associated WFs.".format(res_cursor.count())
        ref_wf_fws_list.append( *[ n["nodes"] for n in res_cursor ] )
        logger.debug("Found {} FWs in associated WF.".format(len(ref_wf_fws_list[-1])))
    else:
        logger.error("Prod. fw id no. {}: {} not unique! IGNORED".format(i,prod_fw_ids[i]))
        raise ValueError()

specific_postp_queries = [ {
  "fw_id": { "$in": ref_wf_fws_list[i] },
  **general_postp_query,
  **param_query } for i, param_query in enumerate(param_queries) ]

logger.debug("Specific postp. fw queries:  {}".format(specific_postp_queries))

# sort to have latest fw first in list:
postp_fw_ids = [
    lp.get_fw_ids( postp_query, sort=[ ("updated_on",-1) ] ) for postp_query in specific_postp_queries ]
logger.info("Prod. fw ids:  {}".format(production_fw_ids))
logger.info("Postp. fw ids: {}".format(postp_fw_ids))

for prod_fw_id, associated_postp_fw_ids in zip(unique_prod_fw_ids,postp_fw_ids):
    if len( associated_postp_fw_ids ) > 1:
        logger.warn("Prod. fw id {} has more than one associated postp. fw id ({}). Most recent used.".format(prod_fw_id, associated_postp_fw_ids))
    elif len( associated_postp_fw_ids ) < 1:
        logger.warn("Prod. fw id {} has no associated postp. fw id (). Double-check!".format(prod_fw_id))

# unique_postp_fw_ids = [ fw_ids[0] if len( fw_ids ) == 1 else None for fw_ids in postp_fw_ids ]
unique_postp_fw_ids = [ fw_ids[0] for fw_ids in postp_fw_ids ]
logger.info("Unique postp. fw ids: {}".format(unique_postp_fw_ids))

wf = Workflow.from_file("build_nemo/wf.yaml")

# get all FWs in new Workflow that are to be appended
fws = [ fw for fw in wf.fws if step_to_append in fw.name ]
logger.info("Candidate fw ids for appending: {}".format( [ fw.fw_id for fw in fws ] ))

matched_fw_ids = []
for param_set in sweep_params:
    current_match = []
    for fw in fws:
        if is_subset(param_set["spec"], fw.spec):
            current_match.append(fw.fw_id)
    matched_fw_ids.append( current_match )
logger.info("Candidate fw ids matched against parameter sets: {}".format(matched_fw_ids))

# check whether matches are unique:
unique_matched_fw_ids = [ fw_ids[0] if len( fw_ids ) == 1 else None for fw_ids in matched_fw_ids ]
logger.info("Unique candidate fw ids matched against parameter sets: {}".format(unique_matched_fw_ids))

# construct partial workflows for matches
matched_partial_wfs = []
for fw_id in unique_matched_fw_ids:
    if fw_id:
        partial_wf = extract_partial_workflow(wf, fw_id)
    else:
        partial_wf = None

    matched_partial_wfs.append(partial_wf)

# construct dict of parent fw ids : new partial workflows
dependencies = {
  (prod_fw_id,postp_fw_id): matched_partial_wfs[i] \
    for i, (prod_fw_id,postp_fw_id) \
    in enumerate( zip(unique_prod_fw_ids,unique_postp_fw_ids) ) }

# finally, append matched extensions with
# for dep, subwf in dependencies.items(): lp.append_wf(subwf,dep)