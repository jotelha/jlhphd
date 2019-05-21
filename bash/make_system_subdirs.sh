#!/bin/bash
prefix={{prefix}}
sysdir=${prefix}/{{sysdir}}
newdir=${sysdir}/{{system}}

if [ ! -d "${sysdir}" ]; then
  mkdir "${sysdir}"
  echo "Directory '${sysdir}' did not exist. Created."
fi

if [ ! -d "${newdir}" ]; then
  mkdir "${newdir}"
  echo "Directory '${newdir}' did not exist. Created."
fi
echo "Process '${newdir}'"
ln -s "${prefix}/1_SDS.pdb" ${newdir}/ # single SDS molecule, necessary for gmx
ln -s "${prefix}/1_CTAB.pdb" ${newdir}/ # single CTAB molecule, necessary for gmx
ln -s "${prefix}/1_NA.pdb" ${newdir}/ # single NA molecule, necessary for packmol
ln -s "${prefix}/1_BR.pdb" ${newdir}/ # single BR molecule, necessary for packmol
ln -s "${prefix}/ionize.mdp" ${newdir}/ # dummy gmx parameter file, necessary for ionization
ln -s "${prefix}/par_all36_lipid_extended_stripped.prm" ${newdir}/ # CHARMM36 parameters
ln -s "${prefix}/top_all36_lipid_extended_stripped.rtf" ${newdir}/ # CHARMM36 topologies
ln -s "${prefix}/pdb_packmol2gmx.sh" ${newdir}/
exit 0
