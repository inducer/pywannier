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





def findAbsMax(mesh_funcs):
    max_val = 0
    max_idx = None

    for i in range(len(mesh_funcs[0].vector())):
        this_val = 0
        for mf in mesh_funcs:
            this_val += abs(mf.vector()[i])
        if this_val > max_val:
            max_idx = i
            max_val = this_val
    return max_idx




def normalizePhaseAt(mf, i):
    return mf * (mf.vector()[i] / abs(mf.vector()[i]))
    

            

job = fempy.stopwatch.tJob("loading")
crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
job.done()

crystal = crystals[0]

multicell_grid = tools.tFiniteGrid(origin = num.array([0.,0.], num.Float),
                                   grid_vectors = crystal.Lattice.DirectLatticeBasis,
                                   limits = [(low, (high-low)/2+low) 
                                             for low, high in crystal.KGrid.limits()])

dlb = crystal.Lattice.DirectLatticeBasis
for band_index, pband in enumerate(crystal.PeriodicBands):
    print "band number", band_index
    f_on_grid = {}
    for mcell_index in multicell_grid:
        k_index = tuple([(el-low)*2+low 
                         for el, (low, high) in zip(mcell_index, multicell_grid.limits())])
        print "k =", crystal.KGrid[k_index]
        f_on_grid[mcell_index] = pband[k_index][1].real
    abs_max_index = findAbsMax(f_on_grid.values())
    for mcell_index in multicell_grid:
        f_on_grid[mcell_index] = normalizePhaseAt(f_on_grid[mcell_index], abs_max_index)

    pc.visualizeGridFunction(multicell_grid, f_on_grid)
    raw_input("[enter for next band]")

