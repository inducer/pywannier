import math, cmath, sets, sys
import cPickle as pickle

import pytools
import pytools.stopwatch as stopwatch

# Numerics imports ------------------------------------------------------------
import pylinear.array as num

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.geometry
import fempy.solver

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc
 



def verify_eigenmodes(crystal, spc):
    periodicity_nodes = pc.find_periodicity_nodes(crystal.Mesh, 
                                                  crystal.BoundaryShapeSection,
                                                  crystal.Lattice.DirectLatticeBasis)

    eigensolver = fempy.solver.LaplacianEigenproblemSolver(crystal.Mesh, 
                                                            constrained_nodes = periodicity_nodes,
                                                            g = crystal.Epsilon, 
                                                            typecode = num.Complex,
                                                            given_number_assignment = crystal.NodeNumberAssignment)

    for k_index in crystal.KGrid.enlarge_at_both_boundaries():
        k = crystal.KGrid[k_index]
        eigensolver.setup_constraints(pc.get_floquet_constraints(periodicity_nodes, k))
        for i, (evalue, emode) in enumerate(crystal.Modes[k_index]):
            err = eigensolver.compute_eigenpair_residual(evalue, emode) / spc(emode, emode)
            #print i, k_index, err
            assert eigensolver.compute_eigenpair_residual(evalue, emode) < 1e-9

def run():
    job = stopwatch.Job("loading")
    crystals = pickle.load(file(",,crystal.pickle", "rb"))
    job.done()

    for crystal in crystals:
        spc = fempy.mesh_function.ScalarProductCalculator(crystal.NodeNumberAssignment,
                                                          crystal.MassMatrix)
        verify_eigenmodes(crystal, spc)
    print "OK"

if __name__ == "__main__":
    run()
