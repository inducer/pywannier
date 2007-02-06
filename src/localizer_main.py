import math, cmath, random
import cPickle as pickle

import pytools
import pytools.grid
import pytools.stopwatch as stopwatch

# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.linear_algebra as la
import pylinear.computation as comp
import pylinear.toybox as toybox
import pylinear.randomized as randomized
import pylinear.iteration as iteration

import scipy.optimize

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.solver
import fempy.eoc
import fempy.integration
import fempy.mesh_function

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc

from localizer_tools import *
from localizer_preproc import *
from localizer_marzari import MarzariSpreadMinimizer
from localizer_conventional import ConventionalSpreadMinimizer
from localizer_postproc import *




def run():
    debug_mode = raw_input("enable debug mode? [n]") == "y"
    ilevel_str = raw_input("interactivity level? [0]")
    interactivity_level = (ilevel_str) and int(ilevel_str) or 0
    #random.seed(200)
    random.seed(2000)

    job = stopwatch.Job("loading")
    crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
    job.done()

    crystal = crystals[-1] 

    assert abs(integrate_over_k_grid(
        crystal.KGrid, 
        lambda _, k: cmath.exp(1j*(k * num.array([5.,17.]))))) < 1e-10
    assert abs(1-integrate_over_k_grid(
        crystal.KGrid, 
        lambda _, k: cmath.exp(1j*(k * num.array([0.,0.]))))) < 1e-10

    sp = fempy.mesh_function.ScalarProductCalculator(crystal.NodeNumberAssignment,
                                                     crystal.MassMatrix)
                                                      
    gaps, clusters = pc.analyze_band_structure(crystal.Bands)
    print "Gaps:", gaps
    print "Clusters:", clusters

    bands = crystal.Bands[0:4]
    pbands = crystal.PeriodicBands[0:4]

    job = stopwatch.Job("guessing initial mix")
    mix_matrix = guess_initial_mix_matrix(crystal, bands, sp)
    job.done()

    #minimizer = MarzariSpreadMinimizer(crystal, sp, debug_mode, interactivity_level)
    minimizer = ConventionalSpreadMinimizer(crystal, sp, debug_mode, interactivity_level)
    mix_matrix = minimizer.minimize_spread(bands, pbands, mix_matrix)

    mixed_bands = compute_mixed_bands(crystal, bands, mix_matrix)

    wannier_grid = pytools.grid.FiniteGrid(
        origin = num.array([0.,0.]),
        grid_vectors = crystal.Lattice.DirectLatticeBasis,
        limits = [(-1,2)] * 2)

    wanniers = compute_wanniers(crystal, mixed_bands, wannier_grid)

    for n, wf in enumerate(wanniers):
        print "average phase deviation (0..1) band ", n, ":", average_phase_deviation(wannier_grid, wf)

    for n, w in enumerate(wanniers):
        print "wannier func number ", n
        wf = {}
        for wi in wannier_grid:
            wf[wi] = w[wi].real
        pc.visualize_grid_function(wannier_grid, wf)
        raw_input("[hit enter when done viewing]")

if __name__ == "__main__":
    run()

