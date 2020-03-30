# a sample on how to utilize FireWorks, FilePad (and underlying pymongo)
from fireworks.core.firework import Workflow
from fireworks.core.launchpad import LaunchPad
from fireworks.utilities.filepad import FilePad
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


projection = dict({"_id": False}, **{ key: True for key in sweep_param_keys })

lp = LaunchPad.auto_load()

fp_query = {
  "metadata.mode": "PRODUCTION",
  "metadata.step": "initial_config",
  "metadata.surfactant": "SDS",
  "metadata.initial_sb_in_dist": 30,
  "metadata.sf_nmolecules": { "$gt": 0 },
  "metadata.constant_indenter_velocity": { "$in": [-1.0e-4, -1.0e-5] } }

fp = FilePad.auto_load()

# fp.filepad.distinct("metadata.sb_in_dist", fq_query}