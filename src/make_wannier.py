import math, cmath, sys, operator
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

sp = fempy.mesh_function.tScalarProductCalculator(crystal.ScalarProduct)
job = fempy.stopwatch.tJob("normalizing modes")
for key in crystal.KGrid:
    norms = []
    for index, (evalue, emode) in enumerate(crystal.Modes[key]):
        norm_squared = sp(emode, emode)
        assert abs(norm_squared.imag) < 1e-10
        emode *= 1 / math.sqrt(norm_squared.real)
job.done()

job = fempy.stopwatch.tJob("localizing bands")
bands = pc.findBands(crystal)
job.done()

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

if True:
    for n, band in enumerate(bands):
        offsets_and_mesh_functions = []
        for multicell_index in multicell_grid:
            R = multicell_grid[multicell_index]
            offsets_and_mesh_functions.append((R, wannier_functions[n][multicell_index].real))
            
        job = fempy.stopwatch.tJob("visualizing band %d" % n)
        visualization.visualizeSeveralMeshes("vtk", 
                                             (",,result.vtk", ",,result_grid.vtk"), 
                                             offsets_and_mesh_functions)
        job.done()
        raw_input("[enter] for next: ")

sys.exit(0)
def addTuple(t1, t2):
    return tuple([t1v + t2v for t1v, t2v in zip(t1, t2)])

# bands to use in computation
selected_bands = bands[:]
n_bands = len(bands)

k_grid_index_increments = [(1,0), (0,1)]

# corresponds to M in Marzari's paper.
# indexed by k_index and an index into k_grid_index_increments
scalar_products = {}

job = fempy.stopwatch.tJob("computing scalar products")
for gbi in crystal.KGrid.gridBlockIndices():
    for kgi_index, index_inc in enumerate(k_grid_index_increments):
        other_index = addTuple(gbi, index_inc)
        mat = num.zeros((n_bands, n_bands), num.Complex)
        for i in range(n_bands):
            for j in range(n_bands):
                ev_i, em_i = bands[i][gbi]
                ev_j, em_j = bands[j][other_index]
                mat[i,j] = sp(em_i, em_j)
        scalar_products[gbi, kgi_index] = mat
job.done()

