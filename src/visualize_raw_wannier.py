import math, cmath, sys, operator
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.array as num

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



def findNearestNode(mesh, point): return tools.argmin(mesh.dofManager(),
                              lambda node: tools.norm2(node.Coordinates-point))

job = fempy.stopwatch.Job("loading")
crystals = pickle.load(file(",,crystal.pickle", "rb"))
job.done()

crystal = crystals[0]

sp = fempy.mesh_function.tScalarProductCalculator(crystal.NodeNumberAssignment,
                                                  crystal.MassMatrix)
job = fempy.stopwatch.Job("normalizing modes")
for key in crystal.KGrid:
    norms = []
    for index, (evalue, emode) in enumerate(crystal.Modes[key]):
        norm_squared = sp(emode, emode)
        assert abs(norm_squared.imag) < 1e-10
        emode *= 1 / math.sqrt(norm_squared.real)
job.done()

job = fempy.stopwatch.Job("finding bands")
bands = pc.find_bands(crystal, crystal.Modes, sp)
job.done()

multicell_grid = tools.FiniteGrid(origin = num.array([0.,0.], num.Float),
                                  grid_vectors = crystal.Lattice.DirectLatticeBasis,
                                  limits = [(-2,2)] * 2)

def compute_wanniers(crystal, bands):
    job = fempy.stopwatch.Job("computing wannier functions")
    wannier_functions = []
    for n, band in enumerate(bands):
        print "computing band", n
        this_wannier_function = {}
        for multicell_index in multicell_grid:
            R = multicell_grid[multicell_index]
            def function_in_integral(k_index):
                k = crystal.KGrid[k_index]
                return cmath.exp(1.j*k*R) * band[k_index][1]

            this_wannier_function[multicell_index] = \
                                                   tools.get_parallelogram_volume(crystal.Lattice.DirectLatticeBasis) \
                                                   / (2*math.pi) ** len(crystal.Lattice.DirectLatticeBasis) \
                                                   * fempy.integration.integrateOnTwoDimensionalGrid(
                crystal.KGrid, function_in_integral)
        wannier_functions.append(this_wannier_function)
    job.done()
    return wannier_functions

wannier_functions = compute_wanniers(crystal, bands)
for n, band in enumerate(bands):
    offsets_and_mesh_functions = []
    for multicell_index in multicell_grid:
        R = multicell_grid[multicell_index]
        offsets_and_mesh_functions.append((R, wannier_functions[n][multicell_index].real))
            
    job = fempy.stopwatch.Job("visualizing band %d" % n)
    visualization.visualizeSeveralMeshes("vtk", 
                                             (",,result.vtk", ",,result_grid.vtk"), 
                                             offsets_and_mesh_functions)
    job.done()
    raw_input("[enter] for next: ")
