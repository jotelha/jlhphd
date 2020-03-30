#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot side views with bounding sphere."""


class PlotSideViewsWithSpheres:
    """Plot side views with bounding sphere."""

    def plot_side_views_with_spheres(self,
            atoms=None, cc=None, R=None, figsize=(12,4), fig=None, ax=None):
        """
        Plots xy, yz and zx projections of atoms and sphere(s)

        Parameters
        ----------
        atoms: ase.atoms

        cc: (N,3) ndarray
            centers of spheres
        R:  (N,) ndarray
            radii of spheres
        figsize: 2-tuple, default (12,4)
        fig: matplotlib.figure, default None
        ax:  list of three matploblib.axes objects
        """
        import logging
        import numpy as np
        import matplotlib.pyplot as plt
        from ase.visualize.plot import plot_atoms  # has nasty offset issues
        from cycler import cycler  # here used for cycling through colors in plots

        if not cc:
            cc = self.C

        if not R:
            R = self.R

        if not atoms:
            atoms = self.atoms

        atom_radii = 0.5

        cc = np.array(cc,ndmin=2)
        self.logger.info("C({}) = {}".format(cc.shape,cc))
        R = np.array(R,ndmin=1)
        self.logger.info("R({}) = {}".format(R.shape,R))
        xmin = atoms.get_positions().min(axis=0)
        xmax = atoms.get_positions().max(axis=0)
        self.logger.info("xmin({}) = {}".format(xmin.shape,xmin))
        self.logger.info("xmax({}) = {}".format(xmax.shape,xmax))

        ### necessary due to ASE-internal atom position computations
        # see https://gitlab.com/ase/ase/blob/master/ase/io/utils.py#L69-82
        X1 = xmin - atom_radii
        X2 = xmax + atom_radii

        M = (X1 + X2) / 2
        S = 1.05 * (X2 - X1)

        scale = 1
        internal_offset = [ np.array(
            [scale * np.roll(M,i)[0] - scale * np.roll(S,i)[0] / 2,
             scale * np.roll(M,i)[1] - scale * np.roll(S,i)[1] / 2]) for i in range(3) ]

        atom_permut = [ atoms.copy() for i in range(3) ]

        for i, a in enumerate(atom_permut):
            a.set_positions( np.roll(a.get_positions(),i,axis=1) )

        rot      = ['0x,0y,0z']*3#,('90z,90x'),('90x,90y,0z')]
        label    = [ np.roll(np.array(['x','y','z'],dtype=str),i)[0:2] for i in range(3) ]

        # dim: sphere, view, coord
        center   = np.array([
            [ np.roll(C,i)[0:2] - internal_offset[i] for i in range(3) ] for C in cc ])

        self.logger.info("projected cc({}) = {}".format(center.shape,center))

        color_cycle = cycler(color=[
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
        circle   = [ [ plt.Circle( c , r, fill=False, **col) for c in C ] for C,r,col in zip(center,R,color_cycle) ]
        margin   = 1.1

        # dim: view, coord, minmax (i.e., 3,2,2)
        plot_bb = np.rollaxis( np.array(
            [ np.min(center - margin*(np.ones(center.T.shape)*R).T,axis=0),
              np.max(center + margin*(np.ones(center.T.shape)*R).T,axis=0) ] ).T, 1, 0)

        #plot_bb  = np.array( [ [
        #    [ [np.min(c[0]-margin*R[0]), np.max(c[0]+margin*R[0])],
        #      [np.min(c[1]-margin*R[0]), np.max(c[1]+margin*R[0])] ] ) for c in C ] for C,r in zip(center,R) ] )
        self.logger.info("projected bb({}) = {}".format(plot_bb.shape,plot_bb))

        if ax is None and self.ax is None:
            fig, ax = plt.subplots(1,3,figsize=figsize)
        elif ax is None:
            fig, ax = self.fig, self.ax

        (ax_xy, ax_xz, ax_yz)  = ax[:]
        self.logger.info("iterators len(atom_permut={}, len(ax)={}, len(rot)={}, len(circle)={}".format(
                len(atom_permut),len(ax),len(rot),len(circle)))

        #self.logger.info("len(circle)={}".format(len(circle))

        #for aa, a, r, C in zip(atom_permut,ax,rot,circle):
        for i, a in enumerate(ax):
            # rotation strings see https://gitlab.com/ase/ase/blob/master/ase/utils/__init__.py#L235-261
            plot_atoms(atom_permut[i],a,rotation=rot[i],radii=0.5,show_unit_cell=0,offset=(0,0))
            for j, c in enumerate(circle):
                self.logger.info("len(circle[{}])={}".format(j,len(c)))
                a.add_patch(c[i])

        for a,l,bb in zip(ax,label,plot_bb):
            a.set_xlabel(l[0])
            a.set_ylabel(l[1])
            a.set_xlim(*bb[0,:])
            a.set_ylim(*bb[1,:])

        self.fig = fig
        self.ax = ax
        return fig, ax

    def read(self, infile = None):
        import ase.io
        if not infile:
            infile = self.infile
        self.atoms = ase.io.read(infile, format='proteindatabank')

    def write(self, outfile = None):
        if not outfile:
            outfile = self.outfile
        self.fig.savefig(outfile)

    def from_file_to_file(self, infile=None, outfile=None, C=None, R=None):
        self.read(infile)
        self.plot_side_views_with_spheres(C=C, R=R)
        self.write(outfile)

    def __init__(self, infile=None, outfile=None, C=None, R=None, fig=None, ax=None):
        import logging
        self.logger = logging.getLogger(__name__)
        self.C = C
        self.R = R
        self.infile = infile
        self.outfile = outfile
        self.fig = fig
        self.ax = ax

        if infile:
            self.read()
        else:
            return

        if C and R:
            self.plot_side_views_with_spheres()
        else:
            return

        if outfile:
            self.write()
        else:
            return
