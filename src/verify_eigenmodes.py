import math, cmath, sets, sys
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.geometry
import fempy.stopwatch
import fempy.solver
import fempy.tools as tools

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc
 



def verifyEigenmodes(crystal, spc):
    periodicity_nodes = pc.findPeriodicityNodes(crystal.Mesh, 
                                                crystal.BoundaryShapeSection,
                                                crystal.Lattice.DirectLatticeBasis)

    eigensolver = fempy.solver.tLaplacianEigenproblemSolver(crystal.Mesh, 
                                                            constrained_nodes = periodicity_nodes,
                                                            g = crystal.Epsilon, 
                                                            typecode = num.Complex,
                                                            given_number_assignment = crystal.NodeNumberAssignment)

    for k_index in crystal.KGrid.enlargeAtBothBoundaries():
        k = crystal.KGrid[k_index]
        eigensolver.setupConstraints(pc.getFloquetConstraints(periodicity_nodes, k))
        for i, (evalue, emode) in enumerate(crystal.Modes[k_index]):
            err = eigensolver.computeEigenpairResidual(evalue, emode) / spc(emode, emode)
            #print i, k_index, err
            assert eigensolver.computeEigenpairResidual(evalue, emode) < 1e-9

def run():
    job = fempy.stopwatch.tJob("loading")
    crystals = pickle.load(file(",,crystal.pickle", "rb"))
    job.done()

    for crystal in crystals:
        spc = fempy.mesh_function.tScalarProductCalculator(crystal.NodeNumberAssignment,
                                                          crystal.MassMatrix)
        verifyEigenmodes(crystal, spc)
    print "OK"

if __name__ == "__main__":
    run()
