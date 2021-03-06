import math, cmath, sys, operator
import cPickle as pickle

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.stopwatch
import fempy.solver
import fempy.eoc
import fempy.tools as tools
import fempy.integration
import fempy.mesh_function
import fempy.visualization as visualization
import fempy.spatial_btree as spatial_btree

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc

job = fempy.stopwatch.tJob("loading")
crystals = pickle.load(file(",,crystal.pickle", "rb"))
job.done()

def getBCResidual(mode, node1, node2, k, pnodes):
    if node1 in pnodes and node2 in pnodes:
        node1_grid_vector = pnodes[node1][0]
        node2_grid_vector = pnodes[node2][0]
        if node1_grid_vector is not node2_grid_vector:
            return 0
        gv = node1_grid_vector
        floquet_factor = cmath.exp(1j * (gv*k))

        def fii(point):
            return tools.absSquared(
                (mode.getGradient(point + gv)*gv)
                - (floquet_factor * mode.getGradient(point)*gv))
        result = fempy.integration.integrateAlongLine(
            node1.Coordinates, node2.Coordinates, fii)
        return result
    else:
        return 0

eoc_rec = fempy.eoc.tEOCRecorder()
for crystal in crystals[:]:
    periodicity_nodes = pc.findPeriodicityNodes(crystal.Mesh, 
                                                crystal.BoundaryShapeSection,
                                                crystal.Lattice.DirectLatticeBasis)

    visualization.writeGnuplotMesh(crystal.Mesh, ",,mesh.data")

    b_edges = fempy.mesh.getBoundaryEdges(crystal.Mesh)

    boundary_error = 0.
    for index in list(crystal.KGrid)[1:]:
        k = crystal.KGrid[index]
        for evalue, mode in crystal.Modes[index][0:1]:
            for node1, node2 in b_edges["floquet"]:
                boundary_error += getBCResidual(mode, node1, node2,
                                                k, periodicity_nodes)

        print k, boundary_error
    eoc_rec.addDataPoint(len(crystal.Mesh.elements())**0.5,
                         boundary_error ** 0.5)

print "Boundary normal derivative EOC:", eoc_rec.estimateOrderOfConvergence()
eoc_rec.writeGnuplotFile(",,boundary_eoc.data")
