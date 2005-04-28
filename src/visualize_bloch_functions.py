import math, sys
import cmath
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





job = fempy.stopwatch.Job("loading")
crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
job.done()

crystal = crystals[0]

multicell_grid = tools.FiniteGrid(origin = num.array([0.,0.], num.Float),
                                  grid_vectors = crystal.Lattice.DirectLatticeBasis,
                                  limits = [(-2,2)] * 2)

dlb = crystal.Lattice.DirectLatticeBasis
for band_index, band in enumerate(crystal.Bands):
    for k_index in crystal.KGrid:
        k = crystal.KGrid[k_index]
        break_k = False
        while True:
            print "band = ", band_index, "k =",k
            my_evalue, bloch_mode = band[k_index]
            my_evalue2, periodic_mode = crystal.PeriodicBands[band_index][k_index]
            assert abs(my_evalue-my_evalue2) < 1e-10
            print "eigenvalue", my_evalue
            value = raw_input("[p for psi, u for u, b for next band, empty for next]:")
            if value == "":
                break
            if value == "b":
                break_k = True
                break

            if value in ["p", "u"]:
                f_on_grid = {}
                for multicell_index in multicell_grid:
                    R = multicell_grid[multicell_index]

                    if value == "p":
                        my_mode = bloch_mode * cmath.exp(1.j*k*R)
                    else:
                        my_mode = periodic_mode
                    f_on_grid[multicell_index] = my_mode.imaginary
                pc.visualize_grid_function(multicell_grid, f_on_grid)

        if break_k:
            break

