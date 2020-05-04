# reuse previous aggregation pipeline
"""compute rmsd."""

import MDAnalysis as mda # here used for reading and analyzing gromacs trajectories
import MDAnalysis.analysis.rms as mda_rms
import numpy as np

import datetime
import getpass
import socket


def atom_rmsd(gro, trr, out, atom_name='AU', **kwargs):
    """Computes time resolved rmsd of atom group identified by atom name.

    rmsd(t) = sqrt(1/N*sum_i=1^N w_i*(x_i(t)-x_i^rev)^2)

    see

    https://www.mdanalysis.org/mdanalysis/documentation_pages/analysis/rms.html

    Units in output textfile are default MDAnalysis units.

    https://www.mdanalysis.org/mdanalysis/documentation_pages/units.html

    Parameters
    ----------
        gro: str
            GROMACS gro coordinates file
        trr: str
            GROMACS trr trajectory file with N frames
        out: str
            output text file
        atom_name: str, optional
            defaults: 'AU'
        **kwargs:
            keyword arguments forwarded to  MDAnalysis.analysis.rms.RMSD

    Output
    ------
        out text file contains time [ps] and rmsd [Ang] in column vectors.
    """

    mda_trr = mda.Universe(gro, trr)

    atom_group = mda_trr.atoms[mda_trr.atoms.names == atom_name]

    rmsd_atom_group = mda_rms.RMSD(atom_group, ref_frame=0, **kwargs)
    rmsd_atom_group.run()

    # format of rmsd:
    # rmsdT = rmsd_atom_group.rmsd.T
    # frame = rmsdT[0]
    # time = rmsdT[1]
    # rmsd = rmsdT[2]

    data = rmsd_atom_group.rmsd[:,1:]  # time and rmsd in column vectors
    np.savetxt(out, data, fmt='%.8e',
        header='\n'.join((
            '{modulename:s}, {username:s}@{hostname:s}, {timestamp:s}'.format(
                modulename=__name__,
                username=getpass.getuser(),
                hostname=socket.gethostname(),
                timestamp=str(datetime.datetime.now()),
            ),
            'https://www.mdanalysis.org/mdanalysis/documentation_pages/analysis/rms.html',
            'rmsd(t) = sqrt(1/N*sum_i=1^N w_i*(x_i(t)-x_i^rev)^2)',
            'time [ps], rmsd [Ang]')))
