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
import fempy.tools as tools
import fempy.integration
import fempy.visualization as visualization

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc




def findNearestNode(mesh, point):
    return tools.argmin(mesh.dofManager(),
                              lambda node: tools.norm2(node.Coordinates-point))

job = fempy.stopwatch.tJob("loading")
crystal = pickle.load(file(",,crystal.pickle", "rb"))
job.done()

job = fempy.stopwatch.tJob("localizing bands")
bands = pc.findBands(crystal)
job.done()

#job = fempy.stopwatch.tJob("normalizing bloch functions")
#dlb = crystal.Lattice.DirectLatticeBasis
#bottom_left_node_number = findNearestNode(crystal.Mesh, -0.5*(dlb[0]+dlb[1]))
#for key in crystal.Modes.genuineKeys():
#    for value, vector in crystal.Modes[key]:
#        vector *= 1/vector[bottom_left_node_number]
#job.done()


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
