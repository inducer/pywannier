import math, cmath, sys, operator, random
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.algorithms as algo
import pylinear.matrix_tools as mtools
import pylinear.iteration as iteration

import scipy.optimize

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.stopwatch
import fempy.solver
import fempy.eoc
import fempy.tools as tools
import fempy.integration
import fempy.mesh_function
import fempy.visualization as visualization

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc

crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))

for crystal in crystals:
    periodicity_nodes = pc.findPeriodicityNodes(crystal.Mesh, 
                                                crystal.Lattice.DirectLatticeBasis)

    for band_index in range(len(crystal.Bands)):
        band = crystal.Bands[band_index]
        pband = crystal.PeriodicBands[band_index]

        for k_index in crystal.KGrid:
            k = crystal.KGrid[k_index]
            mode = band[k_index][1]
            pmode = pband[k_index][1]
            for dependent_node, (gv, independent_nodes) in periodicity_nodes.iteritems():
                my_coord_sum = -(dependent_node.Coordinates + gv)
                my_sum = -mode[dependent_node]
                my_periodic_sum = -pmode[dependent_node]
                for node, weight in independent_nodes:
                    my_coord_sum += weight * node.Coordinates
                    my_sum += weight * cmath.exp(-1j * mtools.sp(gv, k)) \
                              * mode[node]
                    my_periodic_sum += weight * pmode[node]

                assert mtools.norm2(my_coord_sum) < 1e-9
                assert abs(my_sum) < 1e-9
                assert abs(my_periodic_sum) < 1e-9

