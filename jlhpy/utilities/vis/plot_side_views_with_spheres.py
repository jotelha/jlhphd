#!/usr/bin/env python
#
# plot_side_views_with_spheres.py
#
# Copyright (C) 2020 IMTEK Simulation
# Author: Johannes Hoermann, johannes.hoermann@imtek.uni-freiburg.de
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Plot side views with bounding sphere."""

# NOTE: for serializing and deserializing snippets with
# the utilities within wf.serialize and dill, formulate
# imports as below (otherwise, the builtin __import__
# will be missing when deserializing the functions)


def plot_side_views_with_spheres(
        atoms, cc, R, figsize=(12, 4), fig=None, ax=None):
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
    logging = __import__('logging')
    np = __import__('numpy')
    plt = __import__('matplotlib.pyplot').pyplot
    plot_atoms = __import__('ase.visualize.plot').visualize.plot.plot_atoms
    # from ase.visualize.plot import plot_atoms  # has nasty offset issues
    # from cycler import cycler  # here used for cycling through colors in plots
    cycler = __import__('cycler').cycler

    logger = logging.getLogger(__name__)

    atom_radii = 0.5

    cc = np.array(cc, ndmin=2)
    logger.info("C({}) = {}".format(cc.shape, cc))
    R = np.array(R, ndmin=1)
    logger.info("R({}) = {}".format(R.shape,R))
    xmin = atoms.get_positions().min(axis=0)
    xmax = atoms.get_positions().max(axis=0)
    logger.info("xmin({}) = {}".format(xmin.shape, xmin))
    logger.info("xmax({}) = {}".format(xmax.shape, xmax))

    ### necessary due to ASE-internal atom position computations
    # see https://gitlab.com/ase/ase/blob/master/ase/io/utils.py#L69-82
    X1 = xmin - atom_radii
    X2 = xmax + atom_radii

    M = (X1 + X2) / 2
    S = 1.05 * (X2 - X1)

    scale = 1
    internal_offset = [np.array(
        [scale * np.roll(M, i)[0] - scale * np.roll(S, i)[0] / 2,
         scale * np.roll(M, i)[1] - scale * np.roll(S, i)[1] / 2]) for i in range(3) ]

    atom_permut = [atoms.copy() for i in range(3)]

    for i, a in enumerate(atom_permut):
        a.set_positions(np.roll(a.get_positions(), i, axis=1))

    rot      = ['0x,0y,0z']*3#,('90z,90x'),('90x,90y,0z')]
    label    = [ np.roll(np.array(['x','y','z'],dtype=str),i)[0:2] for i in range(3) ]

    # dim: sphere, view, coord
    center   = np.array([
        [np.roll(C, i)[0:2]-internal_offset[i] for i in range(3)] for C in cc])

    logger.info("projected cc({}) = {}".format(center.shape,center))

    color_cycle = cycler(color=[
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
    circle = [ [ plt.Circle( c , r, fill=False, **col) for c in C ] for C,r,col in zip(center,R,color_cycle) ]
    margin = 1.1

    # dim: view, coord, minmax (i.e., 3,2,2)
    plot_bb = np.rollaxis( np.array(
        [ np.min(center - margin*(np.ones(center.T.shape)*R).T,axis=0),
          np.max(center + margin*(np.ones(center.T.shape)*R).T,axis=0) ] ).T, 1, 0)

    #plot_bb  = np.array( [ [
    #    [ [np.min(c[0]-margin*R[0]), np.max(c[0]+margin*R[0])],
    #      [np.min(c[1]-margin*R[0]), np.max(c[1]+margin*R[0])] ] ) for c in C ] for C,r in zip(center,R) ] )
    logger.info("projected bb({}) = {}".format(plot_bb.shape,plot_bb))

    if ax is None:
        fig, ax = plt.subplots(1,3,figsize=figsize)

    (ax_xy, ax_xz, ax_yz)  = ax[:]
    logger.info("iterators len(atom_permut={}, len(ax)={}, len(rot)={}, len(circle)={}".format(
            len(atom_permut),len(ax),len(rot),len(circle)))

    #logger.info("len(circle)={}".format(len(circle))

    #for aa, a, r, C in zip(atom_permut,ax,rot,circle):
    for i, a in enumerate(ax):
        # rotation strings see https://gitlab.com/ase/ase/blob/master/ase/utils/__init__.py#L235-261
        plot_atoms(atom_permut[i],a,rotation=rot[i],radii=0.5,show_unit_cell=0,offset=(0,0))
        for j, c in enumerate(circle):
            logger.info("len(circle[{}])={}".format(j,len(c)))
            a.add_patch(c[i])

    for a,l,bb in zip(ax,label,plot_bb):
        a.set_xlabel(l[0])
        a.set_ylabel(l[1])
        a.set_xlim(*bb[0,:])
        a.set_ylim(*bb[1,:])

    return fig, ax


def plot_side_views_with_spheres_via_ase(infile, outfile, C, R):
    ase = __import__('ase')
    atoms = ase.io.read(infile, format='proteindatabank')
    fig, ax = plot_side_views_with_spheres(atoms=atoms, cc=C, R=R)
    fig.savefig(outfile)

def plot_side_views_with_spheres_via_parmed(infile, outfile, C, R,
        atomic_number_replacements={}):
    pmd = __import__('parmed')
    ase = __import__('ase')
    pmd_structure = pmd.load_file(infile)
    ase_structure = ase.Atoms(
        numbers=[
            atomic_number_replacements[str(a.atomic_number)]
            if str(a.atomic_number) in atomic_number_replacements
            else a.atomic_number for a in pmd_structure.atoms],
        positions=pmd_structure.get_coordinates(0))
    fig, ax = plot_side_views_with_spheres(atoms=ase_structure, cc=C, R=R)
    fig.savefig(outfile)
