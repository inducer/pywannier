import math, cmath, sys, operator
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools

import scipy.optimize

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

# tools -----------------------------------------------------------------------
def addTuple(t1, t2):
    return tuple([t1v + t2v for t1v, t2v in zip(t1, t2)])

def matrixToList(num_mat):
    if len(num_mat.shape) == 1:
        return [x for x in num_mat]
    else:
        return [matrixToList(x) for x in num_mat]

def vectorToKDependentMatrix(vec):
    matrix = {}
    index = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        subvec = vec[index:index+n_bands*n_bands]
        matrix[ii] = num.array(Numeric.reshape(subvec, (n_bands, n_bands)))
        index += n_bands*n_bands
    return matrix

def kDependentMatrixToVector(matrix):
    vec = Numeric.zeros((N * n_bands * n_bands,), Numeric.Complex)
    index = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        subvec = Numeric.reshape(Numeric.array(matrixToList(matrix[ii])), 
                                 (n_bands*n_bands,))
        vec[index:index+n_bands*n_bands] = subvec
        index += n_bands*n_bands
    return vec

# K space weights -------------------------------------------------------------
dimensions = 2

k_grid_index_increments = []
for i in range(dimensions):
    direction = [0] * dimensions
    direction[i] = 1
    k_grid_index_increments.append(tuple(direction))
    direction = [0] * dimensions
    direction[i] = -1
    k_grid_index_increments.append(tuple(direction))

k_grid_increments = [crystal.KGrid[kgii] - crystal.KGrid[0,0]
                     for kgii in k_grid_index_increments]

k_weights = [0.5 / mtools.norm2squared(kgi)
             for kgi in k_grid_increments]
k_weight_sum = sum(k_weights)

# verify...
for i in range(dimensions):
    for j in range(dimensions):
        my_sum = 0
        for kgi_index, kgi in enumerate(k_grid_increments):
            my_sum += k_weights[kgi_index]*kgi[i]*kgi[j]
        assert my_sum == tools.delta(i, j)

# Marzari-relevant functionality ----------------------------------------------
n_bands = 6 # 2 .. len(bands), 1 does not work (FIXME?)
N = tools.product(crystal.KGrid.gridIntervalCounts())

def computeOffsetScalarProducts(crystal, bands):
    scalar_products = {}
    for ii in crystal.KGrid.chopUpperBoundary():
        for kgii_index, kgii in enumerate(k_grid_index_increments):
            other_index = addTuple(ii, kgii)
            mat = num.zeros((n_bands, n_bands), num.Complex)
            for i in range(n_bands):
                for j in range(n_bands):
                    ev_i, em_i = bands[i][ii]
                    ev_j, em_j = bands[j][other_index]
                    mat[i,j] = sp(em_i, em_j)
            scalar_products[ii, kgii] = mat
    return scalar_products

def updateOffsetScalarProducts(scalar_products, mix_matrix):
    mm = num.matrixmultiply

    current_scalar_products = {}
    for ii in crystal.KGrid.chopUpperBoundary():
        for kgii in k_grid_index_increments:
            current_scalar_products[ii, kgii] = mm(
                mix_matrix[ii], 
                mm(scalar_products[ii, kgii], 
                   num.hermite(mix_matrix[addTuple(ii, kgii)])))
    return current_scalar_products

def wannierCenters(scalar_products):
    wannier_centers = []
    for n in range(n_bands):
        result = num.zeros((2,), num.Float)
        for ii in crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(k_grid_index_increments):
                result -= k_weights[kgii_index] \
                          * k_grid_increments[kgii_index] \
                          * cmath.log(scalar_products[ii, kgii][n,n]).imag
        result /= N
        wannier_centers.append(result)
    return wannier_centers

def spreadFunctional(scalar_products):
    wannier_centers = wannierCenters(scalar_products)

    total_spread_f = 0
    for n in range(n_bands):
        mean_r_squared = 0
        for ii in crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(k_grid_index_increments):
                mean_r_squared += k_weights[kgii_index] \
                          * (1 - abs(scalar_products[ii, kgii][n,n])**2 
                             + cmath.log(scalar_products[ii, kgii][n,n]).imag**2)
        mean_r_squared /= N
        total_spread_f += mean_r_squared - mtools.norm2squared(wannier_centers[n])
    return total_spread_f

def omegaI(scalar_products):
    omega_i = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        for kgii_index, kgii in enumerate(k_grid_index_increments):
            omega_i += k_weights[kgii_index] \
                       * (n_bands - mtools.frobeniusNormSquared(scalar_products[ii, kgii]))
    return omega_i / N

def omegaOD(scalar_products):
    def frobeniusNormOffDiagonalSquared(a):
        result = 0
        for i,j in a.indices():
            if i != j:
                result += abs(a[i,j])**2
        return result

    omega_od = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        for kgii_index, kgii in enumerate(k_grid_index_increments):
            omega_od += k_weights[kgii_index] \
                       * (frobeniusNormOffDiagonalSquared(scalar_products[ii, kgii]))
    return omega_od / N

def omegaD(scalar_products, wannier_centers = None):
    if wannier_centers is None:
        wannier_centers = wannierCenters(scalar_products)

    omega_d = 0.
    for n in range(n_bands):
        for ii in crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(k_grid_index_increments):
                b = k_grid_increments[kgii_index]

                omega_d += k_weights[kgii_index] \
                           * (cmath.log(scalar_products[ii,kgii][n,n]).imag \
                              + mtools.sp(wannier_centers[n], b))**2
    return omega_d / N

def spreadFunctional2(scalar_products, wannier_centers = None):
    return omegaI(scalar_products) + \
           omegaOD(scalar_products) + \
           omegaD(scalar_products, wannier_centers)

def spreadFunctionalGradient(scalar_products, wannier_centers = None):
    mm = num.matrixmultiply

    if wannier_centers is None:
        wannier_centers = wannierCenters(scalar_products)

    gradient = {}
    for ii in crystal.KGrid.chopUpperBoundary():
        k = crystal.KGrid[ii]
        result = num.zeros((n_bands, n_bands), num.Complex)
        for kgii_index, kgii in enumerate(k_grid_index_increments):
            m = scalar_products[ii, kgii]
            m_diagonal = num.diagonal(m)
            r = num.multiply(m, m_diagonal)
            r_tilde = num.divide(m, m_diagonal)
            
            q = num.asarray(num.log(m_diagonal).imaginary, num.Complex)
            for n in range(n_bands):
                q[n] += mtools.sp(k_grid_increments[kgii_index], wannier_centers[n])
            t = num.multiply(r_tilde, q)

            a_r = num.transpose(r-num.hermite(r))/2
            s_t = num.transpose(t+num.hermite(t))/2j

            result -= 4. * k_weights[kgii_index] * (a_r - s_t)
        gradient[ii] = result
    return gradient

def getMixMatrix(prev_mix_matrix, factor, gradient):
    mm = num.matrixmultiply

    temp_mix_matrix = pc.makeKPeriodicLookupStructure(crystal.KGrid)
    for ii in crystal.KGrid.chopUpperBoundary():
        dW = factor * gradient[ii]
        exp_dW = mtools.matrixExpByDiagonalization(dW)
        temp_mix_matrix[ii] = mm(prev_mix_matrix[ii], exp_dW)
    return temp_mix_matrix

def run():
    mm = num.matrixmultiply

    iteration = 0

    orig_sps = computeOffsetScalarProducts(crystal, bands)
    print "sf1", spreadFunctional(orig_sps)
    print "sf2", spreadFunctional2(orig_sps)

    mix_matrix = {}
    for ii in crystal.KGrid.chopUpperBoundary():
        mix_matrix[ii] = num.identity(n_bands, num.Complex)
    mix_matrix = pc.makeKPeriodicLookupStructure(crystal.KGrid, mix_matrix)

    while True:
        print "--------------------------------------------------"
        print "ITERATION %d" % iteration
        print "--------------------------------------------------"
        
        sps = updateOffsetScalarProducts(orig_sps, mix_matrix)
        print "spread_func:", spreadFunctional(sps)
        gradient = spreadFunctionalGradient(sps)

        def minfunc(x):
            temp_mix_matrix = getMixMatrix(mix_matrix, x, gradient)
            temp_sps = updateOffsetScalarProducts(orig_sps, temp_mix_matrix)
            result = spreadFunctional(temp_sps)
            print "try", x, result
            return result

        xmin = scipy.optimize.brent(minfunc, brack = (-1e-1, 1e-1))

        mix_matrix = getMixMatrix(mix_matrix, xmin, gradient)

        iteration += 1

run()
