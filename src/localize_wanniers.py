import math, cmath, sys, operator, random
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools
import pylinear.iteration as iteration

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

# tools -----------------------------------------------------------------------
def addTuple(t1, t2):
    return tuple([t1v + t2v for t1v, t2v in zip(t1, t2)])

def matrixToList(num_mat):
    if len(num_mat.shape) == 1:
        return [x for x in num_mat]
    else:
        return [matrixToList(x) for x in num_mat]

def vectorToKDependentMatrix(crystal, vec):
    matrix = pc.makeKPeriodicLookupStructure(crystal.KGrid)
    index = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        subvec = vec[index:index+n_bands*n_bands]
        matrix[ii] = num.array(Numeric.reshape(subvec, (n_bands, n_bands)))
        index += n_bands*n_bands
    return matrix

def makeRandomKDependentSkewHermitianMatrix(crystal, size, tc):
    matrix = pc.makeKPeriodicLookupStructure(crystal.KGrid)
    for ii in crystal.KGrid.chopUpperBoundary():
        matrix[ii] = mtools.makeRandomSkewHermitianMatrix(size, tc)
    return matrix

def kDependentMatrixToVector(crystal, matrix):
    N = tools.product(crystal.KGrid.gridIntervalCounts())
    vec = Numeric.zeros((N * n_bands * n_bands,), Numeric.Complex)
    index = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        subvec = Numeric.reshape(Numeric.array(matrixToList(matrix[ii])), 
                                 (n_bands*n_bands,))
        vec[index:index+n_bands*n_bands] = subvec
        index += n_bands*n_bands
    return vec

# K space weights -------------------------------------------------------------
class tKSpaceDirectionalWeights:
    def __init__(self, crystal):
        self.KGridIndexIncrements = []
        dimensions = len(crystal.Lattice.DirectLatticeBasis)
        for i in range(dimensions):
            direction = [0] * dimensions
            direction[i] = 1
            self.KGridIndexIncrements.append(tuple(direction))
            direction = [0] * dimensions
            direction[i] = -1
            self.KGridIndexIncrements.append(tuple(direction))

        self.KGridIncrements = [crystal.KGrid[kgii] - crystal.KGrid[0,0]
                                for kgii in self.KGridIndexIncrements]

        self.KWeights = [0.5 / mtools.norm2squared(kgi)
                         for kgi in self.KGridIncrements]

        # verify...
        for i in range(dimensions):
            for j in range(dimensions):
                my_sum = 0
                for kgi_index, kgi in enumerate(self.KGridIncrements):
                    my_sum += self.KWeights[kgi_index]*kgi[i]*kgi[j]
                assert my_sum == tools.delta(i, j)

# Marzari-relevant functionality ----------------------------------------------
def computeOffsetScalarProducts(crystal, bands, dir_weights, scalar_product_calculator):
    n_bands = len(bands)
    scalar_products = {}
    for ii in crystal.KGrid.chopUpperBoundary():
        for kgii_index, kgii in enumerate(dir_weights.KGridIndexIncrements):
            other_index = addTuple(ii, kgii)
            mat = num.zeros((n_bands, n_bands), num.Complex)
            for i in range(n_bands):
                for j in range(n_bands):
                    ev_i, em_i = bands[i][ii]
                    ev_j, em_j = bands[j][other_index]
                    mat[i,j] = scalar_product_calculator(em_i, em_j)
            scalar_products[ii, kgii] = mat
    return scalar_products

def updateOffsetScalarProducts(crystal, dir_weights, scalar_products, mix_matrix):
    mm = num.matrixmultiply

    current_scalar_products = {}
    for ii in crystal.KGrid.chopUpperBoundary():
        for kgii in dir_weights.KGridIndexIncrements:
            current_scalar_products[ii, kgii] = mm(
                mix_matrix[ii], 
                mm(scalar_products[ii, kgii], 
                   num.hermite(mix_matrix[addTuple(ii, kgii)])))
    return current_scalar_products

def wannierCenters(n_bands, crystal, dir_weights, scalar_products):
    wannier_centers = []
    for n in range(n_bands):
        result = num.zeros((2,), num.Float)
        for ii in crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(dir_weights.KGridIndexIncrements):
                result -= dir_weights.KWeights[kgii_index] \
                          * dir_weights.KGridIncrements[kgii_index] \
                          * cmath.log(scalar_products[ii, kgii][n,n]).imag
        result /= tools.product(crystal.KGrid.gridIntervalCounts())
        wannier_centers.append(result)
    return wannier_centers

def spreadFunctional(n_bands, crystal, dir_weights, scalar_products):
    wannier_centers = wannierCenters(n_bands, crystal, dir_weights, scalar_products)

    total_spread_f = 0
    for n in range(n_bands):
        mean_r_squared = 0
        for ii in crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(dir_weights.KGridIndexIncrements):
                mean_r_squared += dir_weights.KWeights[kgii_index] \
                          * (1 - abs(scalar_products[ii, kgii][n,n])**2 
                             + cmath.log(scalar_products[ii, kgii][n,n]).imag**2)
        mean_r_squared /= tools.product(crystal.KGrid.gridIntervalCounts())
        total_spread_f += mean_r_squared - mtools.norm2squared(wannier_centers[n])
    return total_spread_f

def badWannierCenters(n_bands, crystal, dir_weights, scalar_products):
    wannier_centers = []
    for n in range(n_bands):
        result = num.zeros((2,), num.Complex)
        for ii in crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(dir_weights.KGridIndexIncrements):
                result -= 1j* dir_weights.KWeights[kgii_index] \
                          * num.asarray(k_grid_increments[kgii_index], num.Complex) \
                          * (scalar_products[ii, kgii][n,n] - 1)
        result /= tools.product(crystal.KGrid.gridIntervalCounts())
        wannier_centers.append(result)
    return wannier_centers

def badSpreadFunctional(n_bands, crystal, dir_weights, scalar_products):
    wannier_centers = badWannierCenters(n_bands, crystal, dir_weights, scalar_products)

    total_spread_f = 0
    for n in range(n_bands):
        mean_r_squared = 0
        for ii in crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(dir_weights.KGridIndexIncrements):
                mean_r_squared += dir_weights.KWeights[kgii_index] \
                          * (2 - 2 * scalar_products[ii, kgii][n,n].real)
        mean_r_squared /= tools.product(crystal.KGrid.gridIntervalCounts())
        total_spread_f += mean_r_squared - mtools.norm2squared(wannier_centers[n])
    return total_spread_f

def omegaI(n_bands, crystal, dir_weights, scalar_products):
    omega_i = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        for kgii_index, kgii in enumerate(dir_weights.KGridIndexIncrements):
            omega_i += dir_weights.KWeights[kgii_index] \
                       * (n_bands - mtools.frobeniusNormSquared(scalar_products[ii, kgii]))
    return omega_i / tools.product(crystal.KGrid.gridIntervalCounts())

def omegaOD(crystal, dir_weights, scalar_products):
    def frobeniusNormOffDiagonalSquared(a):
        result = 0
        for i,j in a.indices():
            if i != j:
                result += abs(a[i,j])**2
        return result

    omega_od = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        for kgii_index, kgii in enumerate(dir_weights.KGridIndexIncrements):
            omega_od += dir_weights.KWeights[kgii_index] \
                       * (frobeniusNormOffDiagonalSquared(scalar_products[ii, kgii]))
    return omega_od / tools.product(crystal.KGrid.gridIntervalCounts())

def omegaD(n_bands, crystal, dir_weights, scalar_products, wannier_centers = None):
    if wannier_centers is None:
        wannier_centers = wannierCenters(n_bands, crystal, dir_weights, scalar_products)

    omega_d = 0.
    for n in range(n_bands):
        for ii in crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(dir_weights.KGridIndexIncrements):
                b = dir_weights.KGridIncrements[kgii_index]

                omega_d += dir_weights.KWeights[kgii_index] \
                           * (cmath.log(scalar_products[ii,kgii][n,n]).imag \
                              + mtools.sp(wannier_centers[n], b))**2
    return omega_d / tools.product(crystal.KGrid.gridIntervalCounts())

def spreadFunctionalViaOmegas(n_bands, crystal, dir_weights, scalar_products, wannier_centers = None):
    return omegaI(n_bands, crystal, dir_weights, scalar_products) + \
           omegaOD(crystal, dir_weights, scalar_products) + \
           omegaD(n_bands, crystal, dir_weights, scalar_products, wannier_centers)

def spreadFunctionalGradient(n_bands, crystal, dir_weights, scalar_products, wannier_centers = None):
    mm = num.matrixmultiply

    if wannier_centers is None:
        wannier_centers = wannierCenters(n_bands, crystal, dir_weights, scalar_products)

    gradient = {}
    for ii in crystal.KGrid.chopUpperBoundary():
        k = crystal.KGrid[ii]
        result = num.zeros((n_bands, n_bands), num.Complex)
        for kgii_index, kgii in enumerate(dir_weights.KGridIndexIncrements):
            m = scalar_products[ii, kgii]
            m_diagonal = num.diagonal(m)
            r = num.multiply(m, m_diagonal)
            r_tilde = num.divide(m, m_diagonal)
            
            q = num.asarray(num.log(m_diagonal).imaginary, num.Complex)
            for n in range(n_bands):
                q[n] += mtools.sp(dir_weights.KGridIncrements[kgii_index], 
                                  wannier_centers[n])
            t = num.multiply(r_tilde, q)

            a_r = (num.transpose(r)-num.conjugate(r))/2
            s_t = (num.transpose(t)+num.conjugate(t))/2j

            result -= 4. * dir_weights.KWeights[kgii_index] * (a_r - s_t)
        gradient[ii] = result
    return gradient

def getMixMatrix(crystal, prev_mix_matrix, factor, gradient):
    mm = num.matrixmultiply

    temp_mix_matrix = pc.makeKPeriodicLookupStructure(crystal.KGrid)
    for ii in crystal.KGrid.chopUpperBoundary():
        dW = factor * gradient[ii]
        # assert dW skew-hermitian
        assert mtools.frobeniusNorm(dW + num.hermite(dW)) < 1e-15

        exp_dW = mtools.matrixExpByDiagonalization(dW)
        # assert exp_dW unitary
        assert mtools.frobeniusNorm(mm(exp_dW, num.hermite(exp_dW)) 
                                    - num.identity(exp_dW.shape[0], exp_dW.typecode())) < 1e-12

        temp_mix_matrix[ii] = mm(prev_mix_matrix[ii], exp_dW)
    return temp_mix_matrix

def minimizeSpread(crystal, bands, scalar_product_calculator, mix_matrix):
    mm = num.matrixmultiply
    dir_weights = tKSpaceDirectionalWeights(crystal)
    orig_sps = computeOffsetScalarProducts(crystal, bands, dir_weights, 
                                           scalar_product_calculator)

    observer = iteration.makeObserver(stall_thresh = 1e-5, max_stalls = 3)
    observer.reset()
    try:
        while True:
            sps = updateOffsetScalarProducts(crystal, dir_weights, orig_sps, mix_matrix)
            oi, od, ood = omegaI(len(bands), crystal, dir_weights, sps), \
                          omegaD(len(bands), crystal, dir_weights, sps), \
                          omegaOD(crystal, dir_weights, sps)
            sf = oi+od+ood
            print "spread_func:", sf, oi, od, ood
            observer.addDataPoint(sf)

            gradient = spreadFunctionalGradient(len(bands), crystal, dir_weights, sps)
            #gradient = makeRandomKDependentSkewHermitianMatrix(crystal, len(bands), num.Complex)

            assert abs(spreadFunctional(len(bands), crystal, dir_weights, sps) - sf) < 1e-10

            def minfunc(x):
                temp_mix_matrix = getMixMatrix(crystal, mix_matrix, x, gradient)
                temp_sps = updateOffsetScalarProducts(crystal, dir_weights, orig_sps, temp_mix_matrix)
                result = spreadFunctional(len(bands), crystal, dir_weights, temp_sps)
                return result

            def badminfunc(x):
                temp_mix_matrix = getMixMatrix(crystal, mix_matrix, x, gradient)
                temp_sps = updateOffsetScalarProducts(crystal, dir_weights, orig_sps, temp_mix_matrix)
                result = badSpreadFunctional(len(bands), crystal, dir_weights, temp_sps)
                return result

            def gradfunc(x):
                temp_mix_matrix = getMixMatrix(crystal, mix_matrix, x, gradient)
                temp_sps = updateOffsetScalarProducts(crystal, dir_weights, orig_sps, temp_mix_matrix)
                new_grad = spreadFunctionalGradient(len(bands), crystal, dir_weights, temp_sps)
                sp = 0.
                for ii in crystal.KGrid.chopUpperBoundary():
                    sp += mtools.entrySum(num.multiply(gradient[ii], num.conjugate(new_grad[ii])))
                return sp.real

            xmin = scipy.optimize.brent(minfunc, brack = (0, 1e-1))
            #print "GRAD_PLOT"
            #tools.writeGnuplotGraph(gradfunc, 0, 3 * xmin, 
                                    #steps = 200, progress = True, fname = ",,sf_grad.data")
            #print "SF_PLOT"
            #tools.writeGnuplotGraph(minfunc, 0, 3 * xmin, 
                                    #steps = 200, progress = True, fname = ",,sf.data")
            #raw_input("see plot:")

            mix_matrix = getMixMatrix(crystal, mix_matrix, xmin, gradient)
    except iteration.tIterationStalled:
        pass
    return mix_matrix

def computeMixedBands(crystal, bands, mix_matrix):
    result = []
    for n in range(len(bands)):
        band = pc.makeKPeriodicLookupStructure(crystal.KGrid)

        for ii in crystal.KGrid.chopUpperBoundary():
            my_sum = 0 * bands[n][ii][1]
            for i in range(len(bands)):
                my_sum += mix_matrix[ii][n, i] * bands[i][ii][1]
            # set eigenvalue to 0 since there is no meaning attached to it
            band[ii] = 0, my_sum
        result.append(pc.tBand(crystal, band))

    return result

def computeWanniers(crystal, bands, wannier_grid):
    job = fempy.stopwatch.tJob("computing wannier functions")
    wannier_functions = []
    for n, band in enumerate(bands):
        print "computing band", n
        this_wannier_function = {}
        for wannier_index in wannier_grid:
            R = wannier_grid[wannier_index]
            def function_in_integral(k_index):
                k = crystal.KGrid[k_index]
                return cmath.exp(1.j * mtools.sp(k, R)) * band[k_index][1]

            this_wannier_function[wannier_index] = tools.getParallelogramVolume(crystal.Lattice.DirectLatticeBasis) \
                                                   / (2*math.pi) ** len(crystal.Lattice.DirectLatticeBasis) \
                                                   * fempy.integration.integrateOnTwoDimensionalGrid(
                crystal.KGrid, function_in_integral)
        wannier_functions.append(this_wannier_function)
    job.done()
    return wannier_functions

def visualizeGridFunction(multicell_grid, func_on_multicell_grid):
    offsets_and_mesh_functions = []
    for multicell_index in multicell_grid:
        R = multicell_grid[multicell_index]
        offsets_and_mesh_functions.append((R, func_on_multicell_grid[multicell_index].real))
    visualization.visualizeSeveralMeshes("vtk", 
                                         (",,result.vtk", ",,result_grid.vtk"), 
                                         offsets_and_mesh_functions)

def integrateOverKGrid(crystal, f):
    integral = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        integral += f(crystal.KGrid[ii])
    return integral / tools.product(crystal.KGrid.gridIntervalCounts())

def phaseVariance(multicell_grid, func_on_multicell_grid):
    my_sum = 0
    for gi in wannier_grid:
        my_sum += sum(func_on_multicell_grid[gi].vector())
    avg_phase_term = my_sum / abs(my_sum)

    my_phase_diff_sum = 0.
    n = 0
    for gi in multicell_grid:
        fvec = func_on_multicell_grid[gi].vector() / avg_phase_term
            
        for z in fvec:
            my_phase_diff_sum += cmath.log(z).imag**2
            n += 1
    return math.sqrt(my_phase_diff_sum / n)

def generateIntegerTuplesBelow(n, length, least = 0):
    assert length >= 0
    if length == 0:
        yield []
    else:
        for i in range(least, n):
            for base in generateIntegerTuplesBelow(n, length-1, least):
                yield [i] + base

def generateAllIntegerTuples(length, least = 0):
    assert length >= 0
    current_max = least
    while True:
        for max_pos in range(length):
            for prebase in generateIntegerTuplesBelow(current_max, max_pos, least):
                for postbase in generateIntegerTuplesBelow(current_max+1, length-max_pos-1, least):
                    yield prebase + [current_max] + postbase
        current_max += 1
            
def generateGaussians(crystal):
    for l in generateAllIntegerTuples(2,1):
        div_x, div_y = tuple(l)
        print l
        dlb = crystal.Lattice.DirectLatticeBasis
        h_x = dlb[0] / div_x
        h_y = dlb[1] / div_y

        def resfunc(point):
            result = 0
            for idx_x in range(div_x):
                y_result = 0
                for idx_y in range(div_y):
                    y_result += math.exp(-20*div_y**2*mtools.sp(dlb[1], point-(idx_y+.5)*h_y+dlb[1]/2)**2)
                result += y_result * \
                          math.exp(-20*div_x**2*mtools.sp(dlb[0], point-(idx_x+.5)*h_x+dlb[0]/2)**2)
            return result
        yield resfunc
    
def run():
    random.seed()

    crystals = pickle.load(file(",,crystal.pickle", "rb"))
    crystal = crystals[0]

    node_number_assignment = crystal.Modes[0,0][0][1].numberAssignment()

    it = generateGaussians(crystal)
    for gaussian in it:
        mf = fempy.mesh_function.discretizeFunction(crystal.Mesh, gaussian, 
                                                    number_assignment = node_number_assignment)
        visualization.visualize("vtk", (",,result.vtk", ",,result_grid.vtk"), mf)
        raw_input("look here:")



    assert abs(integrateOverKGrid(crystal, 
                                  lambda k: cmath.exp(1j*mtools.sp(k, num.array([5.,17.]))))) < 1e-10
    assert abs(1- integrateOverKGrid(crystal, 
                                     lambda k: cmath.exp(1j*mtools.sp(k, num.array([0.,0.]))))) < 1e-10

    sp = fempy.mesh_function.tScalarProductCalculator(node_number_assignment,
                                                      crystal.ScalarProduct)
                                                      
    pc.normalizeModes(crystal, sp)

    bands = pc.findBands(crystal)
    gaps, clusters = pc.analyzeBandStructure(bands)
    bands = bands[1:6]

    mix_matrix = pc.makeKPeriodicLookupStructure(crystal.KGrid, mix_matrix)
    for ii in crystal.KGrid.chopUpperBoundary():
        mix_matrix[ii] = num.identity(len(bands), num.Complex)
        #mix_matrix[ii] = mtools.makeRandomOrthogonalMatrix(n_bands, num.Complex)

    mix_matrix = minimizeSpread(crystal, bands, sp, mix_matrix)

    mixed_bands = computeMixedBands(crystal, bands, mix_matrix)

    wannier_grid = tools.tFiniteGrid(origin = num.array([0.,0.]),
                                     grid_vectors = crystal.Lattice.DirectLatticeBasis,
                                     limits = [(-3,3)] * 2)

    wanniers = computeWanniers(crystal, mixed_bands, wannier_grid)

    for n, wf in enumerate(wanniers):
        print "band", n, ":", phaseVariance(wannier_grid, wanniers)

    for n, w in enumerate(wanniers):
        print "wannier func number ", n
        visualizeGridFunction(wannier_grid, w)
        raw_input("[hit enter when done viewing]")

if __name__ == "__main__":
    run()



