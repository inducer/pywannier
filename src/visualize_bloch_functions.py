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

dlb = crystal.Lattice.DirectLatticeBasis
for band_index, band in enumerate(bands):
    for k_index in crystal.KGrid:
        k = crystal.KGrid[k_index]
        print "band = ", band_index, "k =",k
        
        offsets_and_mesh_functions = []
        for multicell_index in multicell_grid:
            R = multicell_grid[multicell_index]

            my_mode = cmath.exp(1.j * mtools.sp(k,R)) * band[k_index][1]
            offsets_and_mesh_functions.append((R, my_mode.real))
        visualization.visualizeSeveralMeshes("vtk", 
                                             (",,result.vtk", ",,result_grid.vtk"), 
                                             offsets_and_mesh_functions)
        value = raw_input("[b<enter> for next band, enter for next k in band]:")
        if value == "b":
            break
