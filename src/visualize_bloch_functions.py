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





job = fempy.stopwatch.tJob("loading")
crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
job.done()

crystal = crystals[0]

node_number_assignment = crystal.Modes[0,0][0][1].numberAssignment()
spc = fempy.mesh_function.tScalarProductCalculator(node_number_assignment,
                                                   crystal.ScalarProduct)
multicell_grid = tools.tFiniteGrid(origin = num.array([0.,0.], num.Float),
                                   grid_vectors = crystal.Lattice.DirectLatticeBasis,
                                   limits = [(-2,2)] * 2)

dlb = crystal.Lattice.DirectLatticeBasis
for band_index, band in enumerate(crystal.Bands):
    for k_index in crystal.KGrid:
        k = crystal.KGrid[k_index]
        break_k = False
        while True:
            print "band = ", band_index, "k =",k
            value = raw_input("[p for psi, u for u, b for next band, empty for next]:")
            if value == "":
                break
            if value == "b":
                break_k = True
                break

            if value in ["p", "u"]:
                f_on_grid = {}
                offsets_and_mesh_functions = []
                for multicell_index in multicell_grid:
                    R = multicell_grid[multicell_index]

                    if value == "p":
                        my_mode = cmath.exp(1.j * mtools.sp(k, R)) * band[k_index][1]
                    else:
                        my_mode = crystal.PeriodicBands[band_index][k_index][1]
                    f_on_grid[multicell_index] = my_mode.imaginary
                pc.visualizeGridFunction(multicell_grid, f_on_grid)

        if break_k:
            break

