# reuse previous aggregation pipeline
"""count components in molecular systems."""

import MDAnalysis as mda # here used for reading and analyzing gromacs trajectories


def count_pdb_components_by_resname(pdb, resname='AUM', **kwargs):
    """Count atoms and residues in pdb by resname.

    Parameters
    ----------
        pdb: str
            pdb file name
        resname: str, optional
            defaults: 'AUM'
        **kwargs:
            keyword arguments don't do anything

    Returns
    -------
        (int, int): atom count, residue count
    """

    mda_pdb = mda.Universe(pdb)

    components = mda_pdb.select_atoms('resname {}'.format(resname))
    natoms = len(components)
    nresidues = len(components.residues)
    return natoms, nresidues
