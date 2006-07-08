import math, cmath, sys, operator
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.computation as comp

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

def computeWanniers(crystal, bands):
    job = fempy.stopwatch.tJob("computing wannier functions")
    wannier_functions = []
    for n, band in enumerate(bands):
        print "computing band", n
        this_wannier_function = {}
        for multicell_index in multicell_grid:
            R = multicell_grid[multicell_index]
            def function_in_integral(k_index):
                k = crystal.KGrid[k_index]
                return cmath.exp(1.j * (k*R)) * band[k_index][1]

            this_wannier_function[multicell_index] = \
                                                   tools.getParallelogramVolume(crystal.Lattice.DirectLatticeBasis) \
                                                   / (2*math.pi) ** len(crystal.Lattice.DirectLatticeBasis) \
                                                   * fempy.integration.integrateOnTwoDimensionalGrid(
                crystal.KGrid, function_in_integral)
        wannier_functions.append(this_wannier_function)
    job.done()
    return wannier_functions

if False:
    wannier_functions = computeWanniers(crystal, bands)
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

#sys.exit(0)
def addTuple(t1, t2):
    return tuple([t1v + t2v for t1v, t2v in zip(t1, t2)])

# bands to use in computation
n_bands = 6 #len(bands)

k_grid_index_increments = [tuple(i) 
                           for i in tools.enumerateBasicDirections(2)]
k_grid_increments = [crystal.KGrid[kgi] - crystal.KGrid[0,0]
                     for kgi in k_grid_index_increments]
k_weights = [1/(2. * comp.norm_2_squared(kgi))
             for kgi in k_grid_increments]

# corresponds to M in Marzari's paper.
# indexed by k_index and an index into k_grid_index_increments
scalar_products = {}

job = fempy.stopwatch.tJob("computing scalar products")
for ii in crystal.KGrid.chopBothBoundaries():
    for kgii_index, kgii in enumerate(k_grid_index_increments):
        other_index = addTuple(ii, kgii)
        mat = num.zeros((n_bands, n_bands), num.Complex)
        for i in range(n_bands):
            for j in range(n_bands):
                ev_i, em_i = bands[i][ii]
                ev_j, em_j = bands[j][other_index]
                mat[i,j] = sp(em_i, em_j)
        scalar_products[ii, kgii] = mat
job.done()

# Omega_I according to (34) in [Marzari97]
# Bogus because
# - N was left out
omega_i_34 = 0.
for ii in crystal.KGrid.chopBothBoundaries():
    for kgii_index, kgii in enumerate(k_grid_index_increments):
        my_sum = 0
        for i in range(n_bands):
            for j in range(n_bands):
                my_sum += abs(scalar_products[ii, kgii][i,j])**2
        omega_i_34 += k_weights[kgii_index] * (n_bands - my_sum)
print omega_i_34

# Omega_I computed naively

# Bogus because
# - N was left out
# - Infinite sum over R is just cut off

def computeRSquaredEV(band_index):
    result = 0.
    for ii in crystal.KGrid.chopBothBoundaries():
        for kgii_index, kgii in enumerate(k_grid_index_increments):
            my_m = scalar_products[ii, kgii][band_index, band_index]
            result += k_weights[kgii_index] * \
                      (1-abs(my_m)**2+(cmath.log(my_m)).imag**2)
    return result
    
def computeWeirdScalarProduct(k, r, m, n):
    sp = tools.sp
    result = num.zeros((2,), num.Complex)
    for ii in crystal.KGrid.chopBothBoundaries():
        k = crystal.KGrid[ii]
        for kgii_index, kgii in enumerate(k_grid_index_increments):
            result += k_weights[kgii_index] * cmath.exp(1j * sp(k, r)) \
                      * num.asarray(k_grid_increments[kgii_index], num.Complex) \
                      * (scalar_products[ii, kgii][m,n] - tools.delta(m, n))
    return 1.j * result

big_multicell_grid = tools.tFiniteGrid(origin = num.array([0.,0.], num.Float),
                                       grid_vectors = crystal.Lattice.DirectLatticeBasis,
                                       limits = [(-2,2)] * 2)
omega_i_naive = 0.
for n in range(n_bands):
    print "band",n
    omega_i_naive += computeRSquaredEV(n)
    for r_index in big_multicell_grid:
        r = big_multicell_grid[r_index]
        for m in range(n_bands):
            omega_i_naive -= tools.norm2(computeWeirdScalarProduct(r, m, n))

print omega_i_naive


