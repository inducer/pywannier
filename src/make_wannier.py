import math, sys
import cmath
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools

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




def findNearestNode(mesh, point):
    return tools.argmin(mesh.dofManager(),
                              lambda node: tools.norm2(node.Coordinates-point))

job = fempy.stopwatch.tJob("loading")
crystals = pickle.load(file(",,crystal.pickle", "rb"))
job.done()

crystal = crystals[0]

periodicity_nodes = pc.findPeriodicityNodes(crystal.Mesh, 
                                            crystal.Lattice.DirectLatticeBasis)

for k_index in crystal.KGrid:
    k = crystal.KGrid[k_index]
    for evalue, mode in crystal.Modes[k_index]:
        for gv, main_node, other_weights_and_nodes in periodicity_nodes:
            my_sum = mode[main_node]
            for node, weight in other_weights_and_nodes:
                node_val = mode[node]
                my_sum += -weight * cmath.exp(-1j * mtools.sp(gv, k)) * node_val
            if abs(my_sum) > 1e-9:
                print "WARNING: BC check failed by", abs(my_sum)
                print k, main_node.Coordinates, gv
                raw_input()
sys.exit()

job = fempy.stopwatch.tJob("localizing bands")
bands = pc.findBands(crystal)
job.done()

rl = crystal.Lattice.ReciprocalLattice
k_track = [0*rl[0],
           0.5*rl[0],
           0.5*(rl[0]+rl[1]),
           0*rl[0]]
#pc.writeBandDiagram(",,band_diagram.data", crystal, bands,
                    #tools.interpolateVectorList(k_track, 30))

job = fempy.stopwatch.tJob("verifying bcs")
for i, band in enumerate(bands):
    for k_index in crystal.KGrid:
        k = crystal.KGrid[k_index]
        mode = band[k_index][1]
        
        for gv, main_node, other_weights_and_nodes in periodicity_nodes:
            my_sum = mode[main_node]
            print "START:", main_node.Coordinates, my_sum
            for node, weight in other_weights_and_nodes:
                print "FACTOR", weight, cmath.exp(-1j * mtools.sp(gv, k))
                my_sum += -weight * cmath.exp(-1j * mtools.sp(gv, k)) * mode[node]
                print "NEXT:", node.Coordinates, my_sum
            if abs(my_sum) > 1e-9:
                print "WARNING: BC check failed by", abs(my_sum)
                print i, k, main_node.Coordinates, gv
                raw_input()
            else:
                print
job.done()

sys.exit(0)



multicell_grid = tools.tFiniteGrid(origin = num.array([0.,0.], num.Float),
                                   grid_vectors = crystal.Lattice.DirectLatticeBasis,
                                   limits = [(-2,2)] * 2)



job = fempy.stopwatch.tJob("computing wannier functions")
wannier_functions = []
for n, band in enumerate(bands):
    print "computing band", n
    this_wannier_function = {}
    for multicell_index in multicell_grid:
        R = multicell_grid[multicell_index]
        def function_in_integral(k_index):
            k = crystal.KGrid[k_index]
            return cmath.exp(1.j * mtools.sp(k, R)) * band[k_index][1]

        this_wannier_function[multicell_index] = \
                                               tools.getParallelogramVolume(crystal.Lattice.DirectLatticeBasis) \
                                               / (2*math.pi) ** len(crystal.Lattice.DirectLatticeBasis) \
                                               * fempy.integration.integrateOnTwoDimensionalGrid(
            crystal.KGrid, function_in_integral)
    wannier_functions.append(this_wannier_function)
job.done()

for n, band in enumerate(bands):
    omv = []
    for multicell_index in multicell_grid:
        R = multicell_grid[multicell_index]
        omv.append((R, crystal.Mesh, wannier_functions[n][multicell_index].real))

    job = fempy.stopwatch.tJob("visualizing band %d" % n)
    visualization.visualizeSeveralMeshes("vtk", 
                                         (",,result.vtk", ",,result_grid.vtk"), 
                                         omv)
    job.done()
    raw_input("[enter] for next: ")
