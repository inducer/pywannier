import math, cmath, sys, operator, random
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.linear_algebra as la
import pylinear.operation as op

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

job = fempy.stopwatch.Job("loading")
crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
job.done()

for crystal in crystals:
    periodicity_nodes = pc.find_periodicity_nodes(crystal.Mesh, 
                                                  crystal.BoundaryShapeSection,
                                                  crystal.Lattice.DirectLatticeBasis)

    job = fempy.stopwatch.Job("checking")
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
                    my_sum += weight * cmath.exp(-1j * (gv*k)) \
                              * mode[node]
                    my_periodic_sum += weight * pmode[node]

                assert op.norm_2(my_coord_sum) < 1e-9
                assert abs(my_sum) < 1e-9
                assert abs(my_periodic_sum) < 1e-9
    job.done()
    print "OK"

