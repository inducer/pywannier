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

def addTuple(t1, t2):
    return tuple([t1v + t2v for t1v, t2v in zip(t1, t2)])

# bands to use in computation
n_bands = 6 #len(bands)
dimensions = 2

k_grid_index_increments = []
for i in range(dimensions):
    direction = [0.] * dimensions
    direction[i] = 1
    k_grid_index_increments.append(tuple(direction))
    direction = [0.] * dimensions
    direction[i] = -1
    k_grid_index_increments.append(tuple(direction))

k_grid_increments = [crystal.KGrid[kgii] - crystal.KGrid[0,0]
                     for kgii in k_grid_index_increments]

k_weights = [0.5 / mtools.norm2squared(kgi)
             for kgi in k_grid_increments]

# verify k_weights
for i in range(dimensions):
    for j in range(dimensions):
        my_sum = 0
        for kgi_index, kgi in enumerate(k_grid_increments):
            my_sum += k_weights[kgi_index]*kgi[i]*kgi[j]
        assert my_sum == tools.delta(i, j)

# corresponds to M in Marzari's paper.
# indexed by k_index and an index into k_grid_index_increments
scalar_products = {}

job = fempy.stopwatch.tJob("computing scalar products")
for ii in crystal.KGrid.innerIndices():
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

# minimization algorithm ------------------------------------
# FIXME -> gridBlockIndices, wraparound
gradient = {}
N = len(list(crystal.KGrid.innerIndices()))

k_weight_sum = sum(k_weights)
alpha = 1

mix_matrix = tools.tDictionaryWithDefault(lambda x: num.identity(n_bands, num.Complex))
mm = num.matrixmultiply
iteration = 0

while True:
    print "--------------------------------------------------"
    print "ITERATION %d" % iteration
    print "--------------------------------------------------"
    gradient_norm = 0.
    job = fempy.stopwatch.tJob("updating scalar products")
    current_scalar_products = {}
    for ii in crystal.KGrid.innerIndices():
        for kgii in k_grid_index_increments:
            current_scalar_products[ii, kgii] = mm(
                num.hermite(mix_matrix[ii, kgii]), mm(scalar_products[ii, kgii], 
                                         mix_matrix[ii, kgii]))
    job.done()

    job = fempy.stopwatch.tJob("computing wannier centers")
    wannier_centers = []
    for n in range(n_bands):
        result = num.zeros((2,), num.Complex)
        for ii in crystal.KGrid.innerIndices():
            for kgii_index, kgii in enumerate(k_grid_index_increments):
                result -= k_weights[kgii_index] \
                          * num.asarray(k_grid_increments[kgii_index], num.Complex) \
                          * cmath.log(scalar_products[ii, kgii][n,n]).imag
        result /= N
        print n, result
        wannier_centers.append(result)
    job.done()

    job = fempy.stopwatch.tJob("computing gradient")
    for ii in crystal.KGrid.innerIndices():
        k = crystal.KGrid[ii]
        result = num.zeros((n_bands, n_bands), num.Complex)
        for kgii_index, kgii in enumerate(k_grid_index_increments):
            m = current_scalar_products[ii, kgii]
            m_diagonal = num.diagonal(m)
            r = num.transpose(num.multiply(num.transpose(m), m_diagonal))
            r_tilde = num.divide(m, m_diagonal)
            
            q = num.asarray(num.log(m_diagonal).imaginary, num.Complex)
            for n in range(n_bands):
                q[n] += mtools.sp(k_grid_increments[kgii_index], wannier_centers[n])
            t = num.transpose(num.multiply(num.transpose(r_tilde), q))

            a_r = (r-num.hermite(r))/2
            s_t = (t+num.hermite(t))/2j

            result += 4 * k_weights[kgii_index] * (a_r + s_t)
        gradient = result
        gradient_norm += mtools.frobeniusNorm(gradient)
        dW = alpha / (k_weight_sum * 4) * gradient
        exp_dW = mtools.matrixExpByDiagonalization(dW)

        for kgii_index, kgii in enumerate(k_grid_index_increments):
            mix_matrix[ii, kgii] = mm(mix_matrix[ii, kgii], exp_dW)
    job.done()

    print gradient_norm
    iteration += 1
