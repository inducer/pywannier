import math, cmath

import pytools
import pytools.grid
import pytools.stopwatch as stopwatch

# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.computation as comp
import pylinear.iteration as iteration

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc
from localizer_tools import *
from localizer_preproc import compute_mixed_bands, compute_mixed_periodic_bands




def compute_wannier(crystal, wannier_grid, band):
    this_wf = {}
    for wannier_index in wannier_grid:
        R = wannier_grid[wannier_index]
        def function_in_integral(k_index, k):
            k = crystal.KGrid[k_index]
            return cmath.exp(1.j * (k*R)) * band[k_index][1]

        this_wf[wannier_index] = integrate_over_k_grid(crystal.KGrid, 
                                                           function_in_integral)
    return this_wf




def compute_wanniers(crystal, bands, wannier_grid):
    job = stopwatch.Job("computing wannier functions")
    wannier_functions = []

    for n, band in enumerate(bands):
        this_wf = compute_wannier(crystal, wannier_grid, band)
        wannier_functions.append(this_wf)

    job.done()
    return wannier_functions




def average_phase_deviation(multicell_grid, func_on_multicell_grid):
    my_sum = 0
    for gi in multicell_grid:
        my_sum += sum(func_on_multicell_grid[gi].vector())
    avg_phase_term = my_sum / abs(my_sum)

    my_phase_diff_sum = 0.
    n = 0
    n_total = 0
    for gi in multicell_grid:
        fvec = func_on_multicell_grid[gi].vector() / avg_phase_term
            
        for z in fvec:
            if abs(z) >= 1e-2:
                my_phase_diff_sum += abs(cmath.log(z).imag)
                n += 1
            n_total += 1
    return my_phase_diff_sum / (n * math.pi), n, n_total




def visualize_wannier(crystal, bands, mix_matrix, basename, index=0):
    mixed_bands = compute_mixed_bands(crystal, bands, mix_matrix)

    wannier_grid = pytools.grid.FiniteGrid(
        origin = num.array([0.,0.]),
        grid_vectors = crystal.Lattice.DirectLatticeBasis,
        limits = [(-1,2)] * 2)

    my_wannier = compute_wannier(crystal, wannier_grid, mixed_bands[index])

    wf = {}
    for wi in wannier_grid:
        wf[wi] = my_wannier[wi].real

    pc.visualize_grid_function(wannier_grid, wf, basename=basename)

