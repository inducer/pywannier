import math
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




job = fempy.stopwatch.tJob("loading")
crystal = pickle.load(file(",,crystal.pickle", "rb"))
job.done()

job = fempy.stopwatch.tJob("localizing bands")
bands = pc.findBands(crystal)
job.done()

job = fempy.stopwatch.tJob("computing wannier functions")
multicell_grid = tools.tFiniteGrid(origin = num.array([0.,0.], num.Float),
                                   grid_vectors = crystal.Lattice.DirectLatticeBasis,
                                   limits = [(0,4)] * 2)

wannier_functions = []
for n, band in enumerate(bands):
    print "computing band", n
    this_wannier_function = {}
    for multicell_index in multicell_grid.asSequence().getAllIndices():
        R = multicell_grid[multicell_index]
        def function_in_integral(k_index):
            k = multicell_grid[k_index]
            return cmath.exp(-1.j * mtools.sp(k, R)) * band[k_index][1]

        this_wannier_function[multicell_index] = fempy.integration.integrateOnTwoDimensionalGrid(
            crystal.KGrid, function_in_integral)
    wannier_functions.append(this_wannier_function)
job.done()


for n, band in enumerate(bands):
    print "visualizing band", n
    omv = []
    for multicell_index in multicell_grid.asSequence().getAllIndices():
        R = multicell_grid[multicell_index]
        omv.append((R, crystal.Mesh, wannier_functions[n][multicell_index].real))
    visualization.visualize("vtk", (",,result.vtk", ",,result_grid.vtk"), omv)
    raw_input()
