# -*- coding: utf-8 -*-
"""compute rdfs."""
import MDAnalysis as mda # here used for reading and analyzing gromacs trajectories
import MDAnalysis.analysis.rdf as mda_rdf
import numpy as np

import datetime
import getpass
import socket


def atom_atom_rdf(
        gro, trr, out, atom_name_a='AU', atom_name_b='S',
        interval=(0.0, 50.0), **kwargs):
    """Computes time resolved rdf between atom groups identified by atom name.

    https://www.mdanalysis.org/docs/documentation_pages/analysis/rdf.html

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
        atom_name_a, atom_name_b: str, optional
            defaults: 'AU' and 'S'
        interval: tuple or list, optional
            inner and outer cutoff for rdf. default (0.0,80.0)
        **kwargs:
            keyword arguments forwarded to  MDAnalysis.analysis.rdf.InterRDF

    Output
    ------
        out text file contains bins (1st data line), rdf (following data lines)

        bins: (M,) np.ndarray, centers of M bins
        rdf: (M,N) np.ndarray, rdf on M bins for N frames
    """

    mda_trr = mda.Universe(gro, trr)

    atom_group_a = mda_trr.atoms[mda_trr.atoms.names == atom_name_a]
    atom_group_b = mda_trr.atoms[mda_trr.atoms.names == atom_name_b]

    # rdf = mda_rdf.InterRDF(
    #    atom_group_a, atom_group_b, range=(0.0, 80.0), verbose=True)
    bins = []
    time_resolved_rdf = []
    for i in range(len(mda_trr.trajectory)):
        rdf = mda_rdf.InterRDF(
            atom_group_a, atom_group_b, range=interval, **kwargs)
        rdf.run(start=i, stop=i+1)
        # bins.append(rdf.bins.copy())
        time_resolved_rdf.append(rdf.rdf.copy())
    # bins = np.array(bins)
    # bins.append(rdf.bins.copy())
    # bins = np.atleast_2d(rdf.bins.copy())
    bins = rdf.bins.copy()
    # bins is the center cof a bin, see
    # https://www.mdanalysis.org/docs/_modules/MDAnalysis/analysis/rdf.html
    # self.bins = 0.5 * (edges[:-1] + edges[1:])
    time_resolved_rdf = np.array(time_resolved_rdf)

    # 1st dim is time (frame), 2nd dim is bin
    data = np.vstack((bins, time_resolved_rdf))
    np.savetxt(out, data, fmt='%.8e',
        header='\n'.join((
            '{modulename:s}, {username:s}@{hostname:s}, {timestamp:s}'.format(
                modulename=__name__,
                username=getpass.getuser(),
                hostname=socket.gethostname(),
                timestamp=str(datetime.datetime.now()),
            ),
            'https://www.mdanalysis.org/docs/documentation_pages/analysis/rdf.html',
            'g_ab(r)=(N_a N_b)^-1 sum_i=1^N_a sum_j=1^N_b <delta(|r_i-r_j|-r)>',
            'normalized to g_ab(r) -> 1 for r -> infty',
            'first line: bin centers [Ang], following lines: per-frame rdf')))
