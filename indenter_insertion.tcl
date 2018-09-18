# {{header}}
vmdcon -info "tcl version [info patchlevel]"
source jlh_vmd.tcl
namespace import ::JlhVmd::*
use_{{surfactant}}
if { [catch {
  batch_process_pdb_visual {{interface_file}} {{indenter_file}} {{output_prefix}}
} errmsg ] } {
  vmdcon -err "error:  $errmsg"
  exit 1
}
exit 0
