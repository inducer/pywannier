import math, cmath, sys, random, Numeric
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
def matrixToList(num_mat):
    if len(num_mat.shape) == 1:
        return [x for x in num_mat]
    else:
        return [matrixToList(x) for x in num_mat]

def vectorToKDependentMatrix(crystal, n_bands, vec):
    matrix = {}
    index = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        subvec = vec[index:index+n_bands*n_bands]
        matrix[ii] = num.array(Numeric.reshape(subvec, (n_bands, n_bands)))
        index += n_bands*n_bands
    return matrix

def makeRandomKDependentSkewHermitianMatrix(crystal, size, tc):
    matrix = {}
    for ii in crystal.KGrid.chopUpperBoundary():
        matrix[ii] = mtools.makeRandomSkewHermitianMatrix(size, tc)
    return matrix

def kDependentMatrixToVector(crystal, n_bands, matrix):
    N = tools.product(crystal.KGrid.gridIntervalCounts())
    vec = Numeric.zeros((N * n_bands * n_bands,), Numeric.Complex)
    index = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        subvec = Numeric.reshape(Numeric.array(matrixToList(matrix[ii])), 
                                 (n_bands*n_bands,))
        vec[index:index+n_bands*n_bands] = subvec
        index += n_bands*n_bands
    return vec

def kDependentMatrixScalarProduct(k_grid, a1, a2):
    sp = 0.
    for ii in k_grid.chopUpperBoundary():
        sp += mtools.entrySum(num.multiply(a1[ii], num.conjugate(a2[ii])))
    return sp

def operateOnKDependentMatrix(k_grid, a, m_op):
    result = {}
    for ii in k_grid:
        result[ii] = m_op(a[ii])
    return result

def operateOnKDependentMatrices(k_grid, a1, a2, m_op):
    result = {}
    for ii in k_grid:
        result[ii] = m_op(a1[ii], a2[ii])
    return result

class tContinuityAwareArg:
    def __init__(self, last_value_dict = {}):
        self.LastValue = last_value_dict.copy()

    def copy(self):
        return tContinuityAwareArg(self.LastValue)

    def __call__(self, z, association = None):
        return cmath.log(z).imag
        try:
            last_v = self.LastValue[association]
            new_result = cmath.log(z).imag
            last_band = math.floor((last_v+math.pi) / (2*math.pi))
            last_band_center = last_band * 2 * math.pi

            centers = [last_band_center, last_band_center + 2*math.pi,
                       last_band_center - 2*math.pi]
            least_i = tools.argmin(centers,
                                   lambda center: abs(center+new_result - last_v))
            result = centers[least_i] + new_result
            self.LastValue[association] = result
            return result

        except KeyError:
            result = cmath.log(z).imag
            self.LastValue[association] = result
            return result


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
class tMarzariSpreadMinimizer:
    def __init__(self, crystal, spc):
        self.Crystal = crystal
        self.KWeights = tKSpaceDirectionalWeights(crystal)
        self.ScalarProductCalculator = spc

    def computeOffsetScalarProducts(self, pbands):
        n_bands = len(pbands)
        scalar_products = {}

        pbands2 = [pc.makeKPeriodicLookupStructure(self.Crystal.KGrid, pband)
                  for pband in pbands]

        for i in range(len(pbands)):
            a = pbands[i][0,7][0]
            b = pbands[i][0,0][0]
            print a,b
        raw_input()

        for ii in self.Crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                added_tuple = tools.addTuples(ii, kgii)
                reduced_tuple = self.Crystal.KGrid.chopUpperBoundary().reducePeriodically(
                    added_tuple)
                if added_tuple != reduced_tuple:
                    print "reduced", added_tuple, "->", reduced_tuple
                    print "added:", added_tuple, self.Crystal.KGrid[added_tuple]
                    print "reduced:", reduced_tuple, self.Crystal.KGrid[reduced_tuple]

                mat = num.zeros((n_bands, n_bands), num.Complex)
                for i in range(n_bands):
                    for j in range(n_bands):
                        mat[i,j] = self.ScalarProductCalculator(pbands[i][ii][1], 
                                                                pbands[j][reduced_tuple][1])
                        if added_tuple != reduced_tuple:
                            print "1****"
                            reduced_ev, reduced_band = pbands[j][reduced_tuple]
                            print "2****"
                            added_ev, added_band = pbands2[j][added_tuple]
                            print "evs", added_ev, reduced_ev
                            errvec = (reduced_band-added_band).vector()
                            error = mtools.norm2(errvec)
                            sp = self.ScalarProductCalculator(reduced_band, added_band)
                            if error > 1e-15:
                                print i,j, error, abs(sp)
                                raw_input()
                scalar_products[ii, kgii] = mat
        return scalar_products

    def updateOffsetScalarProducts(self, scalar_products, mix_matrix):
        mm = num.matrixmultiply

        current_scalar_products = {}
        for ii in self.Crystal.KGrid.chopUpperBoundary():
            for kgii in self.KWeights.KGridIndexIncrements:
                current_scalar_products[ii, kgii] = mm(
                    mix_matrix[ii], 
                    mm(scalar_products[ii, kgii], 
                       num.hermite(mix_matrix[tools.addTuples(ii, kgii)])))
        return current_scalar_products

    def wannierCenters(self, n_bands, scalar_products, arg):
        wannier_centers = []
        for n in range(n_bands):
            result = num.zeros((2,), num.Float)
            for ii in self.Crystal.KGrid.chopUpperBoundary():
                for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                    result -= self.KWeights.KWeights[kgii_index] \
                              * self.KWeights.KGridIncrements[kgii_index] \
                              * arg(scalar_products[ii, kgii][n,n], 
                                    (ii, kgii, n))
            result /= tools.product(self.Crystal.KGrid.gridIntervalCounts())
            wannier_centers.append(result)
        return wannier_centers

    def spreadFunctional(self, n_bands, scalar_products, arg):
        wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        total_spread_f = 0
        for n in range(n_bands):
            mean_r_squared = 0
            for ii in self.Crystal.KGrid.chopUpperBoundary():
                for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                    mean_r_squared += self.KWeights.KWeights[kgii_index] \
                                      * (1 - abs(scalar_products[ii, kgii][n,n])**2 
                                         + arg(scalar_products[ii, kgii][n,n], 
                                               (ii, kgii, n))**2)
            mean_r_squared /= tools.product(self.Crystal.KGrid.gridIntervalCounts())
            total_spread_f += mean_r_squared - mtools.norm2squared(wannier_centers[n])
        return total_spread_f

    def badWannierCenters(self, n_bands, scalar_products):
        wannier_centers = []
        for n in range(n_bands):
            result = num.zeros((2,), num.Complex)
            for ii in self.Crystal.KGrid.chopUpperBoundary():
                for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                    result -= 1j* self.KWeights.KWeights[kgii_index] \
                              * num.asarray(self.KWeights.KGridIncrements[kgii_index], num.Complex) \
                              * (scalar_products[ii, kgii][n,n] - 1)
            result /= tools.product(self.Crystal.KGrid.gridIntervalCounts())
            wannier_centers.append(result)
        return wannier_centers

    def badSpreadFunctional(self, n_bands, scalar_products):
        wannier_centers = self.badWannierCenters(n_bands, scalar_products)

        total_spread_f = 0
        for n in range(n_bands):
            mean_r_squared = 0
            for ii in self.Crystal.KGrid.chopUpperBoundary():
                for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                    mean_r_squared += self.KWeights.KWeights[kgii_index] \
                              * (2 - 2 * scalar_products[ii, kgii][n,n].real)
            mean_r_squared /= tools.product(self.Crystal.KGrid.gridIntervalCounts())
            total_spread_f += mean_r_squared - mtools.norm2squared(wannier_centers[n])
        return total_spread_f

    def omegaI(self, n_bands, scalar_products):
        omega_i = 0
        for ii in self.Crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                omega_i += self.KWeights.KWeights[kgii_index] \
                           * (n_bands - mtools.frobeniusNormSquared(scalar_products[ii, kgii]))
        return omega_i / tools.product(self.Crystal.KGrid.gridIntervalCounts())

    def omegaOD(self, scalar_products):
        def frobeniusNormOffDiagonalSquared(a):
            result = 0
            for i,j in a.indices():
                if i != j:
                    result += abs(a[i,j])**2
            return result

        omega_od = 0
        for ii in self.Crystal.KGrid.chopUpperBoundary():
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                omega_od += self.KWeights.KWeights[kgii_index] \
                           * (frobeniusNormOffDiagonalSquared(scalar_products[ii, kgii]))
        return omega_od / tools.product(self.Crystal.KGrid.gridIntervalCounts())

    def omegaD(self, n_bands, scalar_products, arg, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        omega_d = 0.
        for n in range(n_bands):
            for ii in self.Crystal.KGrid.chopUpperBoundary():
                for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                    b = self.KWeights.KGridIncrements[kgii_index]

                    omega_d += self.KWeights.KWeights[kgii_index] \
                               * (arg(scalar_products[ii,kgii][n,n], 
                                      (ii, kgii, n)) \
                                  + mtools.sp(wannier_centers[n], b))**2
        return omega_d / tools.product(self.Crystal.KGrid.gridIntervalCounts())

    def spreadFunctionalViaOmegas(self, n_bands, scalar_products, arg, wannier_centers = None):
        return self.omegaI(n_bands, scalar_products) + \
               self.omegaOD(scalar_products) + \
               self.omegaD(n_bands, scalar_products, arg, wannier_centers)

    def spreadFunctionalGradient(self, n_bands, scalar_products, arg, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        gradient = {}
        for ii in self.Crystal.KGrid.chopUpperBoundary():
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                m = scalar_products[ii, kgii]
                m_diagonal = num.diagonal(m)
                r = num.multiply(m, num.conjugate(m_diagonal))

                if True: # DEBUG
                    r2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r2[i,j] = m[i,j] * m[j,j].conjugate()
                    assert mtools.frobeniusNorm(r-r2) < 1e-15

                r_tilde = num.divide(m, m_diagonal)

                if True: # DEBUG
                    r_tilde2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r_tilde2[i,j] = m[i,j] / m[j,j]
                    assert mtools.frobeniusNorm(r_tilde-r_tilde2) < 1e-15

                q = num.zeros((n_bands,), num.Complex)
                for n in range(n_bands):
                    q[n] = arg(m_diagonal[n], (ii, kgii, n))

                for n in range(n_bands):
                    q[n] += mtools.sp(self.KWeights.KGridIncrements[kgii_index], 
                                      wannier_centers[n])
                t = num.multiply(r_tilde, q)
                if True: # DEBUG
                    t2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            t2[i,j] = r_tilde[i,j] * q[j]
                    assert mtools.frobeniusNorm(t-t2) < 1e-15

                # re and im are not literal: watch out for the transpose
                re_r = (num.transpose(r)-num.conjugate(r))/2
                im_t = (num.transpose(t)+num.conjugate(t))/2j

                result += 4. * self.KWeights.KWeights[kgii_index] * (-re_r + im_t)
            gradient[ii] = result
        return gradient

    def spreadFunctionalGradientOmegaD(self, n_bands, scalar_products, arg, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        gradient = {}
        for ii in self.Crystal.KGrid.chopUpperBoundary():
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                m = scalar_products[ii, kgii]
                m_diagonal = num.diagonal(m)

                r_tilde = num.divide(m, m_diagonal)

                q = num.zeros((n_bands,), num.Complex)
                for n in range(n_bands):
                    q[n] = arg(m_diagonal[n], (ii, kgii, n))

                for n in range(n_bands):
                    q[n] += mtools.sp(self.KWeights.KGridIncrements[kgii_index], 
                                      wannier_centers[n])
                t = num.multiply(r_tilde, q)
                im_t = (num.transpose(t)+num.conjugate(t))/2j

                result += 4. * self.KWeights.KWeights[kgii_index] * im_t
            gradient[ii] = result
        return gradient

    def spreadFunctionalGradientOmegaOD(self, n_bands, scalar_products, arg, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        gradient = {}
        for ii in self.Crystal.KGrid.chopUpperBoundary():
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                m = scalar_products[ii, kgii]
                m_diagonal = num.diagonal(m)
                r = num.multiply(m, num.conjugate(m_diagonal))

                re_r = (num.transpose(r)-num.conjugate(r))/2

                result += 4. * self.KWeights.KWeights[kgii_index] * (-re_r)
            gradient[ii] = result
        return gradient

    def getMixMatrix(self, prev_mix_matrix, factor, gradient):
        mm = num.matrixmultiply

        temp_mix_matrix = pc.makeKPeriodicLookupStructure(self.Crystal.KGrid)
        for ii in self.Crystal.KGrid.chopUpperBoundary():
            dW = factor * gradient[ii]

            assert mtools.isSkewHermitian(dW)
            exp_dW = mtools.matrixExpByDiagonalization(dW)
            assert mtools.isUnitary(exp_dW)

            temp_mix_matrix[ii] = mm(prev_mix_matrix[ii], exp_dW)
        return temp_mix_matrix

    def testSPUpdater(self, pbands, mix_matrix):
        job = fempy.stopwatch.tJob("self-test")
        sps_original = self.computeOffsetScalarProducts(pbands)
        sps_updated = self.updateOffsetScalarProducts(sps_original, mix_matrix)
        mixed_bands = computeMixedBands(self.Crystal, pbands, mix_matrix)
        sps_direct = self.computeOffsetScalarProducts(mixed_bands)

        arg = tContinuityAwareArg()
        sf1 = self.spreadFunctional(len(pbands), sps_updated, arg.copy())
        sf2 = self.spreadFunctional(len(pbands), sps_direct, arg.copy())
        assert abs(sf1-sf2) < 1e-10

        for k_index in self.Crystal.KGrid.chopUpperBoundary():
            for kgii in self.KWeights.KGridIndexIncrements:
                assert mtools.frobeniusNorm(sps_direct[k_index, kgii]
                                            - sps_updated[k_index, kgii]) < 1e-9
        job.done()

    def minimizeSpread(self, pbands, mix_matrix):
        mm = num.matrixmultiply
        for ii in self.Crystal.KGrid.chopUpperBoundary():
            assert mtools.isUnitary(mix_matrix[ii], threshold = 1e-5)

        #self.testSPUpdater(pbands, mix_matrix)

        job = fempy.stopwatch.tJob("preparing minimization")
        orig_sps = self.computeOffsetScalarProducts(pbands)
        job.done()
        raw_input()

        oi = self.omegaI(len(pbands), orig_sps)
        arg = tContinuityAwareArg()

        observer = iteration.makeObserver(stall_thresh = 1e-5, max_stalls = 3)
        observer.reset()
        try:
            while True:
                sps = self.updateOffsetScalarProducts(orig_sps, mix_matrix)
                assert abs(oi - self.omegaI(len(pbands), sps)) < 1e-5
                od, ood = self.omegaD(len(pbands), sps, arg), \
                          self.omegaOD(sps)
                sf = oi+od+ood
                print "spread_func:", sf, oi, od, ood
                observer.addDataPoint(sf)

                gradient = self.spreadFunctionalGradient(len(pbands), sps, arg)
                #gradient = makeRandomKDependentSkewHermitianMatrix(crystal, len(pbands), num.Complex)

                assert abs(self.spreadFunctional(len(pbands), sps, arg) - sf) < 1e-5

                def testDerivs(x):
                    print "--------------------------"
                    print x
                    print "--------------------------"
                    temp_mix_matrix = self.getMixMatrix(mix_matrix, x, gradient)
                    temp_sps = self.updateOffsetScalarProducts(orig_sps, temp_mix_matrix)

                    for ii in self.Crystal.KGrid.chopUpperBoundary():
                        for kgii in self.KWeights.KGridIndexIncrements:
                            added_tup = self.Crystal.KGrid.chopUpperBoundary().\
                                        reducePeriodically(tools.addTuples(ii, kgii))
                            if added_tup != tools.addTuples(ii, kgii):
                                print "reduced", tools.addTuples(ii, kgii), "->", added_tup

                            new_m = temp_sps[ii,kgii]
                            dm1 = new_m - orig_sps[ii,kgii]
                            m = orig_sps[ii,kgii]
                            m_plusb = num.hermite(orig_sps[added_tup, tools.negateTuple(kgii)])
                            print mtools.frobeniusNorm(m - m_plusb) 
                            assert mtools.frobeniusNorm(m - m_plusb) < 1e-13
                            dw = x * gradient[ii]
                            dw_plusb = x * gradient[added_tup]

                            dm2 = mm(dw, m) + mm(m, num.hermite(dw))
                            print "dm2-dm1", mtools.frobeniusNorm(dm2-dm1)

                            dm3 = mm(dw, m) + num.hermite(mm(dw_plusb, m_plusb))
                            print "dm3-dm1", mtools.frobeniusNorm(dm3-dm1)
                testDerivs(1e-7)
                testDerivs(1e-6)
                testDerivs(1e-5)
                testDerivs(1e-4)
                testDerivs(1e-3)
                testDerivs(1e-2)
                testDerivs(1e-1)
                raw_input()

                def minfunc(x):
                    temp_mix_matrix = self.getMixMatrix(mix_matrix, x, gradient)
                    temp_sps = self.updateOffsetScalarProducts(orig_sps, temp_mix_matrix)

                    result = self.spreadFunctional(len(pbands), temp_sps, arg)
                    print x, result
                    return result

                def plotfunc(x):
                    temp_mix_matrix = self.getMixMatrix(mix_matrix, x, gradient)
                    temp_sps = self.updateOffsetScalarProducts(orig_sps, temp_mix_matrix)

                    new_grad = self.spreadFunctionalGradient(len(pbands), temp_sps, arg)
                    new_grad_od = self.spreadFunctionalGradientOmegaOD(len(pbands), temp_sps, arg)
                    new_grad_d = self.spreadFunctionalGradientOmegaD(len(pbands), temp_sps, arg)
                    sp = kDependentMatrixScalarProduct(self.Crystal.KGrid, new_grad, gradient)
                    sp_od = kDependentMatrixScalarProduct(self.Crystal.KGrid, new_grad_od, gradient)
                    sp_d = kDependentMatrixScalarProduct(self.Crystal.KGrid, new_grad_d, gradient)

                    od = self.omegaD(len(pbands), temp_sps, arg)
                    ood = self.omegaOD(temp_sps)
                    return od, ood, oi+od+ood, sp.real, sp_od.real, sp_d.real
                           
                xmin = scipy.optimize.brent(minfunc, brack = (0, 1e-1))

                tools.write1DGnuplotGraphs(plotfunc, -3*xmin, 3 * xmin, 
                                           steps = 200, progress = True)
                raw_input("see plot:")

                mix_matrix = self.getMixMatrix(mix_matrix, xmin, gradient)
        except iteration.tIterationStalled:
            pass
        return mix_matrix

def computeMixedBands(crystal, bands, mix_matrix):
    result = []
    for n in range(len(bands)):
        band = {}
        for ii in crystal.KGrid:
            # set eigenvalue to 0 since there is no meaning attached to it
            band[ii] = 0, tools.linearCombination(mix_matrix[ii][n],
                                                  [bands[i][ii][1] for i in range(len(bands))])
        result.append(band)
    return result

def computeWanniers(crystal, bands, wannier_grid):
    job = fempy.stopwatch.tJob("computing wannier functions")
    wannier_functions = []
    for n, band in enumerate(bands):
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

def integrateOverKGrid(crystal, f):
    integral = 0
    for ii in crystal.KGrid.chopUpperBoundary():
        integral += f(crystal.KGrid[ii])
    return integral / tools.product(crystal.KGrid.gridIntervalCounts())

def averagePhaseDeviation(multicell_grid, func_on_multicell_grid):
    my_sum = 0
    for gi in multicell_grid:
        my_sum += sum(func_on_multicell_grid[gi].vector())
    avg_phase_term = my_sum / abs(my_sum)

    my_phase_diff_sum = 0.
    n = 0
    for gi in multicell_grid:
        fvec = func_on_multicell_grid[gi].vector() / avg_phase_term
            
        for z in fvec:
            my_phase_diff_sum += abs(cmath.log(z).imag)
            n += 1
    return my_phase_diff_sum / (n * math.pi)

def generateHierarchicGaussians(crystal, node_number_assignment, typecode):
    for l in tools.generateAllIntegerTuples(2,1):
        div_x, div_y = tuple(l)
        dlb = crystal.Lattice.DirectLatticeBasis
        h_x = dlb[0] / div_x
        h_y = dlb[1] / div_y

        def gaussian(point):
            result = 0
            for idx_x in range(div_x):
                y_result = 0
                for idx_y in range(div_y):
                    y_result += math.exp(-20*div_y**2*mtools.sp(dlb[1], point-(idx_y+.5)*h_y+dlb[1]/2)**2)
                result += y_result * \
                          math.exp(-20*div_x**2*mtools.sp(dlb[0], point-(idx_x+.5)*h_x+dlb[0]/2)**2)
            return result
        yield fempy.mesh_function.discretizeFunction(crystal.Mesh, 
                                                     gaussian, 
                                                     typecode,
                                                     node_number_assignment)

def generateRandomGaussians(crystal, node_number_assignment, typecode):
    dlb = crystal.Lattice.DirectLatticeBasis
    while True:
        center_coords = [random.uniform(0.1, 0.9) for i in range(len(dlb))]
        center = tools.linearCombination(center_coords, dlb) \
                 - 0.5*tools.generalSum(dlb)

        sigma = num.zeros((len(dlb), len(dlb)), num.Float)
        for i in range(len(dlb)):
            max_width = min(1-center_coords[i], center_coords[i])
            sigma[i,i] = random.uniform(0.1, max_width)
        sigma_inv = la.inverse(sigma)
            
        # FIXME this is dependent on dlb actually being unit vectors
        def gaussian(point):
            arg = num.matrixmultiply(sigma_inv, point - center)
            return math.exp(-mtools.norm2squared(arg))

        yield fempy.mesh_function.discretizeFunction(crystal.Mesh, 
                                                     gaussian, 
                                                     typecode,
                                                     node_number_assignment)
    
def guessInitialMixMatrix(crystal, node_number_assignment, bands, sp):
    # generate the gaussians
    gaussians = []
    gaussian_it = generateRandomGaussians(crystal, node_number_assignment, num.Complex)
    for n in range(len(bands)):
        gaussians.append(gaussian_it.next())

    # project the gaussians
    projected_bands = []
    projected_bands_co = []
    for n in range(len(bands)):
        projected_band = {}
        projected_band_co = {}

        for ii in crystal.KGrid.chopUpperBoundary():
            mf = fempy.mesh_function.discretizeFunction(
                crystal.Mesh, lambda x: 0., num.Complex, 
                number_assignment = node_number_assignment)
            coordinates = num.zeros((len(bands),), num.Complex)
            for m in range(len(bands)):
                coordinates[m] = sp(gaussians[n], bands[m][ii][1])
                mf += coordinates[m] * bands[m][ii][1]
            projected_band[ii] = mf
            projected_band_co[ii] = coordinates
        projected_bands.append(projected_band)
        projected_bands_co.append(projected_band_co)

    # orthogonalize the projected gaussians
    mix_matrix = pc.makeKPeriodicLookupStructure(crystal.KGrid)
    for ii in crystal.KGrid.chopUpperBoundary():
        # calculate scalar products
        my_sps = num.zeros((len(bands), len(bands)), num.Complex)
        for n in range(len(bands)):
            for m in range(m+1):
                my_sp = sp(projected_bands[n][ii], projected_bands[m][ii])
                my_sps[n,m] = my_sp
                my_sps[m,n] = my_sp.conjugate()

        inv_sqrt_my_sps = la.inverse(la.cholesky_decomposition(my_sps))

        mix_matrix[ii] = num.zeros((len(bands), len(bands)), num.Complex)
        for n in range(len(bands)):
            # determine and compute correct mixture of projected bands
            mix_matrix[ii][n] = tools.linearCombination(inv_sqrt_my_sps[n], 
                                                        [projected_bands_co[i][ii] 
                                                         for i in range(len(bands))])
                
    return mix_matrix

def run():
    random.seed(10)

    job = fempy.stopwatch.tJob("loading")
    crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
    job.done()

    crystal = crystals[-1]

    node_number_assignment = crystal.Modes[0,0][0][1].numberAssignment()

    assert abs(integrateOverKGrid(crystal, 
                                  lambda k: cmath.exp(1j*mtools.sp(k, num.array([5.,17.]))))) < 1e-10
    assert abs(1- integrateOverKGrid(crystal, 
                                     lambda k: cmath.exp(1j*mtools.sp(k, num.array([0.,0.]))))) < 1e-10

    sp = fempy.mesh_function.tScalarProductCalculator(node_number_assignment,
                                                      crystal.ScalarProduct)
                                                      
    gaps, clusters = pc.analyzeBandStructure(crystal.Bands)

    bands = crystal.Bands[1:6]
    pbands = crystal.PeriodicBands[1:6]

    job = fempy.stopwatch.tJob("guessing initial mix")
    mix_matrix = guessInitialMixMatrix(crystal, 
                                       node_number_assignment, 
                                       bands,
                                       sp)
    job.done()

    minimizer = tMarzariSpreadMinimizer(crystal, sp)
    mix_matrix = minimizer.minimizeSpread(pbands, mix_matrix)

    mixed_bands = computeMixedBands(crystal, bands, mix_matrix)

    wannier_grid = tools.tFiniteGrid(origin = num.array([0.,0.]),
                                     grid_vectors = crystal.Lattice.DirectLatticeBasis,
                                     limits = [(-3,3)] * 2)

    wanniers = computeWanniers(crystal, mixed_bands, wannier_grid)

    for n, wf in enumerate(wanniers):
        print "average phase deviation (0..1) band ", n, ":", averagePhaseDeviation(wannier_grid, wf)

    for n, w in enumerate(wanniers):
        print "wannier func number ", n
        wf = {}
        for wi in wannier_grid:
            wf[wi] = w[wi].real
        pc.visualizeGridFunction(wannier_grid, wf)
        raw_input("[hit enter when done viewing]")

if __name__ == "__main__":
    run()
