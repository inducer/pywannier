import math, cmath, sys, random
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




class tDictionaryOfMatrices(tools.tDictionaryOfArithmeticTypes):
    def conjugate():
        result = tDictionaryOfMatrices()
        for key in self:
            result[key] = num.conjugate(self[key])
        return result
    
    def hermite():
        result = tDictionaryOfMatrices()
        for key in self:
            result[key] = num.hermite(self[key])
        return result
        
    def transpose():
        result = tDictionaryOfMatrices()
        for key in self:
            result[key] = num.transpose(self[key])
        return result
        
    def _getreal():
        result = tDictionaryOfMatrices()
        for key in self:
            result[key] = self[key].real
        return result

    def _getimaginary():
        result = tDictionaryOfMatrices()
        for key in self:
            result[key] = self[key].real
        return result

    def matrixMultiply(self, other):
        return self.binaryOperator(other, num.matrixmultiply)

    real = property(_getreal)
    imaginary = property(_getimaginary)




# tools -----------------------------------------------------------------------
def frobeniusNormOffDiagonalSquared(a):
    result = 0
    for i,j in a.indices():
        if i != j:
            result += abs(a[i,j])**2
    return result

def matrixToList(num_mat):
    if len(num_mat.shape) == 1:
        return [x for x in num_mat]
    else:
        return [matrixToList(x) for x in num_mat]

def makeRandomKDependentSkewHermitianMatrix(crystal, size, tc):
    matrix = tKDependentMatrix
    for k_index in crystal.KGrid:
        matrix[k_index] = mtools.makeRandomSkewHermitianMatrix(size, tc)
    return matrix

def gradScalarProduct(a1, a2):
    rp = num.multiply(a1.real, a2.real)
    ip = num.multiply(a1.imaginary, a2.imaginary)
    return mtools.entrySum(rp) + mtools.entrySum(ip)

def kDependentMatrixGradientScalarProduct(k_grid, a1, a2):
    # we can't use complex multiplication since our "complex" number
    # here is just a convenient notation for a gradient, so the real
    # and imaginary parts have to stay separate.
    sp = 0.
    for k_index in k_grid:
        rp = num.multiply(a1[k_index].real, a2[k_index].real)
        ip = num.multiply(a1[k_index].imaginary, a2[k_index].imaginary)
        sp += mtools.entrySum(rp) + mtools.entrySum(ip)
    return sp / k_grid.gridPointCount()

def operateOnKDependentMatrix(k_grid, a, m_op):
    result = {}
    for k_index in k_grid:
        result[k_index] = m_op(a[k_index])
    return result

def operateOnKDependentMatrices(k_grid, a1, a2, m_op):
    result = {}
    for k_index in k_grid:
        result[k_index] = m_op(a1[k_index], a2[k_index])
    return result

def frobeniusNormOnKDependentMatrices(k_grid, mat):
    result = 0
    for k_index in k_grid:
        result += mtools.frobeniusNorm(mat[k_index])
    return result

def symmetricPart(matrix):
    return 0.5*(matrix+num.transpose(matrix))

def skewSymmetricPart(matrix):
    return 0.5*(matrix-num.transpose(matrix))

class tSimpleArg:
    def __init__(self):
        pass

    def reset(self):
        pass

    def copy(self):
        return self

    def __call__(self, z, association = None):
        return cmath.log(z).imag

class tStatsCountingArg:
    def __init__(self, k_grid, dir_weights):
        self.KGrid = k_grid
        self.KWeights = dir_weights
        self._reset()

    def _reset(self):
        self.Applications = 0
        self.Violations = 0
        self.ViolationMap = tools.tDictionaryWithDefault(
            lambda x: tools.tDictionaryWithDefault(
            lambda x: 0))

    def reset(self):
        if self.Applications:
            print "violations in past term %d out of %d (%f%%)" % \
                  (self.Violations, self.Applications, 
                   100. * self.Violations / self.Applications)

            vios_drawn = 0
            for n, vio_map in self.ViolationMap.iteritems():
                statsf = file(",,violations-%d.data" % n, "w")
                for k_index in self.KGrid:
                    for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                        where = self.KGrid[k_index] + self.KWeights.KGridIncrements[kgii_index] * 0.2
                        vios_here = self.ViolationMap[n][k_index, kgii]
                        vios_drawn += vios_here
                        statsf.write("%f\t%f\t%d\n" %(where[0], where[1], vios_here))
                statsf.close()
            assert vios_drawn == self.Violations
            raw_input("violation maps drawn [enter]:")
        self._reset()


    def copy(self):
        return self

    def __call__(self, z, association = None):
        result = cmath.log(z).imag
        self.Applications += 1
        if abs(result/math.pi) > 0.5:
            self.Violations += 1
            k_index, kgii, n = association
            self.ViolationMap[n][k_index, kgii] += 1
        return result

class tBoinkArg:
    def __init__(self, last_value_dict = {}):
        self.LastValue = last_value_dict.copy()

    def reset(self):
        self.LastValue = {}

    def copy(self):
        return tBoinkArg(self.LastValue)

    def __call__(self, z, association = None):
        result = cmath.log(z).imag

        use_randadd = abs(result/math.pi) > 0.5
        have_randadd = association in self.LastValue
        #assert not (not use_randadd and have_randadd)
        if use_randadd:
            if have_randadd:
                randadd = self.LastValue[association]
            else:
                randadd = random.randint(-1,1) * 2 * math.pi
                self.LastValue[association] = randadd
            return result + randadd
        else:
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
    def __init__(self, crystal, spc, debug_mode = True, interactivity_level = 0):
        self.Crystal = crystal
        self.KWeights = tKSpaceDirectionalWeights(crystal)
        self.ScalarProductCalculator = spc
        self.DebugMode = debug_mode
        self.InteractivityLevel = interactivity_level

    def computeOffsetScalarProducts(self, pbands):
        n_bands = len(pbands)
        scalar_products = {}

        for k_index in self.Crystal.KGrid:
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                added_tuple = self.Crystal.KGrid.reducePeriodically(
                    tools.addTuples(k_index, kgii))
                trace = not self.Crystal.KGrid.isWithinBounds(added_tuple)
                trace = False

                mat = num.zeros((n_bands, n_bands), num.Complex)
                for i in range(n_bands):
                    for j in range(n_bands):
                        if trace and i==j:
                            em1 = pbands[i][k_index][1]
                            em2 = pbands[i][added_tuple][1]
                            value = self.ScalarProductCalculator(pbands[i][k_index][1], 
                                                                 pbands[j][added_tuple][1])
                            print (em1-em2).vector()[0:5]
                            print k_index, i,mtools.norm2((em1-em2).vector()), value
                        mat[i,j] = self.ScalarProductCalculator(pbands[i][k_index][1], 
                                                                pbands[j][added_tuple][1])
                if trace:
                    print "trace", num.diagonal(mat)

                scalar_products[k_index, kgii] = mat

        self.checkScalarProducts(scalar_products)
        return scalar_products

    def checkInitialScalarProducts(self, scalar_products):
        if not self.DebugMode:
            return

        # verify that M^{k,b} = M^{-k,-b}^*
        # (which is only valid for M calculated from original u's, not for
        # later mixtures, hence "initial")
        for k_index in self.Crystal.KGrid:
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is not None:
                    assert mtools.frobeniusNorm(
                        scalar_products[k_index, kgii] 
                        - num.conjugate(scalar_products[
                        pc.invertKIndex(self.Crystal.KGrid, k_index),
                        tools.negateTuple(kgii)])) < 1e-15

    def updateOffsetScalarProducts(self, scalar_products, mix_matrix):
        mm = num.matrixmultiply

        new_scalar_products = {}
        for k_index in self.Crystal.KGrid:
            for kgii in self.KWeights.KGridIndexIncrements:
                added_tuple = self.Crystal.KGrid.reducePeriodically(
                    tools.addTuples(k_index, kgii))
                if self.DebugMode:
                    assert mtools.unitarietyError(mix_matrix[k_index]) < 1e-8
                    assert mtools.unitarietyError(mix_matrix[added_tuple]) < 1e-8

                new_scalar_products[k_index, kgii] = mm(
                    mix_matrix[k_index], 
                    mm(scalar_products[k_index, kgii], 
                       num.hermite(mix_matrix[added_tuple])))
        self.checkScalarProducts(new_scalar_products)
        return new_scalar_products

    def checkScalarProducts(self, scalar_products):
        if self.DebugMode:
            # make sure M^{k,b} = (M^{k+b,-b})^H actually holds
            for k_index in self.Crystal.KGrid:
                for kgii in self.KWeights.KGridIndexIncrements:
                    if scalar_products[k_index, kgii] is None:
                        raise RuntimeError, "None is actually used in scalar products"

                    added_tup = tools.addTuples(k_index, kgii)
                    neg_kgii = tools.negateTuple(kgii)

                    m = scalar_products[k_index, kgii]
                    n_bands = m.shape[0]

                    if (added_tup, neg_kgii) in scalar_products:
                        m_plusb = num.hermite(scalar_products[added_tup, neg_kgii])
                        assert mtools.frobeniusNorm(m - m_plusb) < 1e-13

        if self.InteractivityLevel >= 2:
            n_bands = scalar_products[(0,0), (1,0)].shape[0]
            # analyze arguments of diagonal entries
            magfiles = [file(",,magnitude-%d.data" % n, "w") for n in range(n_bands)]
            argfiles = [file(",,arguments-%d.data" % n, "w") for n in range(n_bands)]
            for k_index in self.Crystal.KGrid:
                for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                    if scalar_products[k_index, kgii] is None:
                        continue

                    where = self.Crystal.KGrid[k_index] + self.KWeights.KGridIncrements[kgii_index] * 0.2
                    m = scalar_products[k_index, kgii]
                    m_diagonal = num.diagonal(m)
                    for i, z in enumerate(m_diagonal):
                        arg = cmath.log(m[i,i]).imag
                        magfiles[i].write("%f\t%f\t%f\n" %(where[0], where[1], abs(m[i,i])))
                        argfiles[i].write("%f\t%f\t%f\n" %(where[0], where[1], arg))
            for af in argfiles:
                af.close()
            for mf in magfiles:
                mf.close()
            raw_input("[magnitude/argument plot ready]")

    def wannierCenters(self, n_bands, scalar_products, arg):
        wannier_centers = []
        for n in range(n_bands):
            result = num.zeros((2,), num.Float)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                for k_index in self.Crystal.KGrid:
                    if scalar_products[k_index, kgii] is None:
                        continue

                    result -= self.KWeights.KWeights[kgii_index] \
                              * self.KWeights.KGridIncrements[kgii_index] \
                              * arg(scalar_products[k_index, kgii][n,n], 
                                    (k_index, kgii, n))
            result /= self.Crystal.KGrid.gridPointCount()
            wannier_centers.append(result)
        return wannier_centers

    def spreadFunctional(self, n_bands, scalar_products, arg):
        wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        total_spread_f = 0
        for n in range(n_bands):
            mean_r_squared = 0
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                for k_index in self.Crystal.KGrid:
                    if scalar_products[k_index, kgii] is None:
                        continue

                    mean_r_squared += self.KWeights.KWeights[kgii_index] \
                                      * (1 - abs(scalar_products[k_index, kgii][n,n])**2 
                                         + arg(scalar_products[k_index, kgii][n,n], 
                                               (k_index, kgii, n))**2)
            mean_r_squared /= self.Crystal.KGrid.gridPointCount()
            total_spread_f += mean_r_squared - mtools.norm2squared(wannier_centers[n])
        return total_spread_f

    def badWannierCenters(self, n_bands, scalar_products):
        # without the series-changing corrections by Marzari
        wannier_centers = []
        for n in range(n_bands):
            result = num.zeros((2,), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                for k_index in self.Crystal.KGrid:
                    if scalar_products[k_index, kgii] is None:
                        continue

                    result -= 1j* self.KWeights.KWeights[kgii_index] \
                              * num.asarray(self.KWeights.KGridIncrements[kgii_index], num.Complex) \
                              * (scalar_products[k_index, kgii][n,n] - 1)
            result /= self.Crystal.KGrid.gridPointCount()
            wannier_centers.append(result)
        return wannier_centers

    def badSpreadFunctional(self, n_bands, scalar_products):
        # without the series-changing corrections by Marzari
        wannier_centers = self.badWannierCenters(n_bands, scalar_products)

        total_spread_f = 0
        for n in range(n_bands):
            mean_r_squared = 0
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                for k_index in self.Crystal.KGrid:
                    if scalar_products[k_index, kgii] is None:
                        continue

                    mean_r_squared += self.KWeights.KWeights[kgii_index] \
                              * (2 - 2 * scalar_products[k_index, kgii][n,n].real)
            mean_r_squared /= self.Crystal.KGrid.gridPointCount()
            total_spread_f += mean_r_squared - mtools.norm2squared(wannier_centers[n])
        return total_spread_f

    def omegaI(self, n_bands, scalar_products):
        omega_i = 0
        for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
            for k_index in self.Crystal.KGrid:
                if scalar_products[k_index, kgii] is None:
                    continue

                omega_i += self.KWeights.KWeights[kgii_index] \
                           * (n_bands - mtools.frobeniusNormSquared(scalar_products[k_index, kgii]))
        return omega_i / self.Crystal.KGrid.gridPointCount()

    def omegaOD(self, scalar_products):
        omega_od = 0
        for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
            for k_index in self.Crystal.KGrid:
                if scalar_products[k_index, kgii] is None:
                    continue

                omega_od += self.KWeights.KWeights[kgii_index] \
                           * (frobeniusNormOffDiagonalSquared(scalar_products[k_index, kgii]))
        return omega_od / self.Crystal.KGrid.gridPointCount()

    def omegaD(self, n_bands, scalar_products, arg, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        omega_d = 0.
        for n in range(n_bands):
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                for k_index in self.Crystal.KGrid:
                    if scalar_products[k_index, kgii] is None:
                        continue

                    b = self.KWeights.KGridIncrements[kgii_index]

                    omega_d += self.KWeights.KWeights[kgii_index] \
                               * (arg(scalar_products[k_index,kgii][n,n], 
                                      (k_index, kgii, n)) \
                                  + mtools.sp(wannier_centers[n], b))**2
        return omega_d / self.Crystal.KGrid.gridPointCount()

    def spreadFunctionalViaOmegas(self, n_bands, scalar_products, arg, wannier_centers = None):
        return self.omegaI(n_bands, scalar_products) + \
               self.omegaOD(scalar_products) + \
               self.omegaD(n_bands, scalar_products, arg, wannier_centers)

    def spreadFunctionalGradient(self, n_bands, scalar_products, arg, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        gradient = {}
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)

                # Omega_OD part
                r = num.multiply(m, num.conjugate(m_diagonal))

                if self.DebugMode:
                    r2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r2[i,j] = m[i,j] * m[j,j].conjugate()
                    assert mtools.frobeniusNorm(r-r2) < 1e-15

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (-num.asarray(skewSymmetricPart(num.transpose(r.real)), num.Complex)
                           +1j*num.asarray(symmetricPart(r.imaginary), num.Complex))

                # Omega_D part
                r_tilde = num.divide(m, m_diagonal)

                if self.DebugMode:
                    r_tilde2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r_tilde2[i,j] = m[i,j] / m[j,j]
                    assert mtools.frobeniusNorm(r_tilde-r_tilde2) < 1e-13

                q = num.zeros((n_bands,), num.Complex)
                for n in range(n_bands):
                    q[n] = arg(m_diagonal[n], (k_index, kgii, n))

                for n in range(n_bands):
                    q[n] += mtools.sp(self.KWeights.KGridIncrements[kgii_index], 
                                      wannier_centers[n])
                t = num.multiply(r_tilde, q)
                if self.DebugMode:
                    t2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            t2[i,j] = r_tilde[i,j] * q[j]
                    assert mtools.frobeniusNorm(t-t2) < 1e-15

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (num.asarray(skewSymmetricPart(num.transpose(t.imaginary)), num.Complex)
                           +1j*num.asarray(symmetricPart(t.real), num.Complex))

            gradient[k_index] = result
        return gradient

    def spreadFunctionalGradientOmegaD(self, n_bands, scalar_products, arg, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        gradient = {}
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)

                r_tilde = num.divide(m, m_diagonal)

                q = num.zeros((n_bands,), num.Complex)
                for n in range(n_bands):
                    #print "mein_q", ii, kgii, n, arg(m_diagonal[n], (ii, kgii, n)) / math.pi
                    q[n] = arg(m_diagonal[n], (k_index, kgii, n))

                for n in range(n_bands):
                    q[n] += mtools.sp(self.KWeights.KGridIncrements[kgii_index], 
                                      wannier_centers[n])
                t = num.multiply(r_tilde, q)

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (num.asarray(skewSymmetricPart(num.transpose(t.imaginary)), num.Complex)
                           +1j*num.asarray(symmetricPart(t.real), num.Complex))

            gradient[k_index] = result
        return gradient

    def spreadFunctionalGradientOmegaOD(self, n_bands, scalar_products, arg):
        gradient = {}
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)
                r = num.multiply(m, num.conjugate(m_diagonal))

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (-num.asarray(skewSymmetricPart(num.transpose(r.real)), num.Complex)
                           +1j*num.asarray(symmetricPart(r.imaginary), num.Complex))
            gradient[k_index] = result
        return gradient

    def getMixMatrix(self, prev_mix_matrix, factor, gradient):
        mm = num.matrixmultiply

        temp_mix_matrix = {}
        for k_index in self.Crystal.KGrid:
            dW = factor * gradient[k_index]
            if self.DebugMode:
                assert mtools.skewHermiticityError(dW) < 1e-13

            exp_dW = mtools.matrixExpByDiagonalization(dW)
            #exp_dW = num.identity(dW.shape[0], dW.typecode()) + dW
            if self.DebugMode:
                assert mtools.unitarietyError(exp_dW) < 1e-13

            temp_mix_matrix[k_index] = mm(exp_dW, prev_mix_matrix[k_index])

        return temp_mix_matrix

    def testSPUpdater(self, pbands, mix_matrix):
        if not self.DebugMode:
            return

        job = fempy.stopwatch.tJob("self-test")
        sps_original = self.computeOffsetScalarProducts(pbands)
        sps_updated = self.updateOffsetScalarProducts(sps_original, mix_matrix)
        mixed_bands = computeMixedPeriodicBands(self.Crystal, pbands, mix_matrix)
        sps_direct = self.computeOffsetScalarProducts(mixed_bands)

        for k_index in self.Crystal.KGrid:
            for kgii in self.KWeights.KGridIndexIncrements:
                if sps_direct[k_index, kgii] is not None:
                    assert mtools.frobeniusNorm(sps_direct[k_index, kgii]
                                               - sps_updated[k_index, kgii]) < 1e-9

        arg = tContinuityAwareArg()
        sf1 = self.spreadFunctional(len(pbands), sps_updated, arg.copy())
        sf2 = self.spreadFunctional(len(pbands), sps_direct, arg.copy())
        assert abs(sf1-sf2) < 1e-10

        job.done()

    def minimizeOmegaODByCodiagonalization(self, raw_scalar_products, mix_matrix):
        """scalar_products are understood to be before application of the
        mix_matrix specified.
        """
        mm = num.matrixmultiply
        sps = self.updateOffsetScalarProducts(raw_scalar_products, mix_matrix)

        if self.DebugMode:
            print "od before pre", self.omegaOD(sps)

        new_mix_matrix = {}
        omega_od_matrices = []
        for k_index in self.Crystal.KGrid:
            for kgii in self.KWeights.KGridIndexIncrements:
                if sps[k_index, kgii] is not None:
                    omega_od_matrices.append(sps[k_index, kgii].copy())

        job = fempy.stopwatch.tJob("pre-minimization")
        q, diag_mats, tol = mtools.codiagonalize(omega_od_matrices)
        job.done()

        for k_index in self.Crystal.KGrid:
            new_mix_matrix[k_index] = mm(num.hermite(q), mix_matrix[k_index])

        if self.DebugMode:
            sps_post = self.updateOffsetScalarProducts(raw_scalar_products, new_mix_matrix)
            print "od after pre", self.omegaOD(sps_post)
        return new_mix_matrix

    def minimizeSpread(self, pbands, mix_matrix):
        mm = num.matrixmultiply
        if self.DebugMode:
            for ii in self.Crystal.KGrid:
                assert mtools.unitarietyError(mix_matrix[ii]) < 5e-3

        self.testSPUpdater(pbands, mix_matrix)

        job = fempy.stopwatch.tJob("computing scalar products")
        orig_sps = self.computeOffsetScalarProducts(pbands)
        self.checkInitialScalarProducts(orig_sps)
        job.done()

        oi = self.omegaI(len(pbands), orig_sps)
        arg = tSimpleArg()
        #arg = tStatsCountingArg(
            #self.Crystal.KGrid, self.KWeights)
        #arg = tContinuityAwareArg()
        #arg = tBoinkArg()

        #mix_matrix = self.minimizeOmegaODByCodiagonalization(orig_sps, mix_matrix)

        observer = iteration.makeObserver(stall_thresh = 1e-4, max_stalls = 3)
        observer.reset()
        try:
            while True:
                arg.reset()
                sps = self.updateOffsetScalarProducts(orig_sps, mix_matrix)
                if self.DebugMode:
                    assert abs(oi - self.omegaI(len(pbands), sps)) < 1e-5
                od, ood = self.omegaD(len(pbands), sps, arg), \
                          self.omegaOD(sps)
                sf = oi+od+ood
                print "spread_func:", sf, oi, od, ood
                observer.addDataPoint(sf)

                gradient = self.spreadFunctionalGradient(len(pbands), sps, arg)
                #gradient = makeRandomKDependentSkewHermitianMatrix(crystal, len(pbands), num.Complex)

                if self.DebugMode:
                    assert abs(self.spreadFunctional(len(pbands), sps, arg) - sf) < 1e-5

                def testDerivs(x):
                    print_count = 4
                    print "--------------------------"
                    print x
                    print "--------------------------"
                    temp_mix_matrix = self.getMixMatrix(mix_matrix, x, gradient)
                    temp_sps = self.updateOffsetScalarProducts(orig_sps, temp_mix_matrix)

                    gpc = self.Crystal.KGrid.gridPointCount()
                    before_oiod = (self.omegaI(len(pbands), sps) \
                                  + self.omegaOD(sps)) * gpc
                    after_oiod = (self.omegaI(len(pbands), temp_sps) \
                                  + self.omegaOD(temp_sps)) * gpc

                    before_oiod2 = 0
                    after_oiod2 = 0
                    doiod3 = 0
                    doiod4 = 0
                    doiod5 = 0
                    doiod7 = 0

                    kdep_dw = {}
                    for k_index in self.Crystal.KGrid:
                        kdep_dw[k_index] = x * gradient[k_index]

                    doiod6 = gpc * kDependentMatrixGradientScalarProduct(
                        self.Crystal.KGrid,
                        kdep_dw,
                        self.spreadFunctionalGradientOmegaOD(len(pbands), 
                                                             sps, arg))

                    for k_index in self.Crystal.KGrid:
                        for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                            added_tup = self.Crystal.KGrid.reducePeriodically(tools.addTuples(k_index, kgii))

                            w_b = self.KWeights.KWeights[kgii_index]

                            new_m = temp_sps[k_index,kgii]
                            m = sps[k_index,kgii]
                            dw = x * gradient[k_index]

                            dw_plusb = x * gradient[added_tup]
                            m_plusb = num.hermite(m)
                            dm1 = new_m - m
                            dm2 = mm(dw, m) + mm(m, num.hermite(dw_plusb))

                            before_oiod_here = w_b * (len(pbands)-tools.norm2squared(num.diagonal(m)))
                            after_oiod_here = w_b * (len(pbands)-tools.norm2squared(num.diagonal(new_m)))

                            before_oiod2 += before_oiod_here
                            after_oiod2 += after_oiod_here
                            doiod_here2  = after_oiod_here-before_oiod_here
                            doiod_here2b = w_b * (+tools.norm2squared(num.diagonal(m))
                                              -tools.norm2squared(num.diagonal(new_m)))
                            assert abs(doiod_here2 - doiod_here2b) < 1e-11

                            doiod_here2c = 2* w_b * sum(num.multiply(-num.diagonal(new_m)+num.diagonal(m),
                                                                     num.conjugate(num.diagonal(m)))).real

                            r = num.multiply(m, num.conjugate(num.diagonal(m)))
                            doiod_here3 = -4*w_b*num.trace(mm(dw, r)).real
                            doiod3 += doiod_here3

                            doiod_here4 = -2*w_b*mtools.sp(
                                num.diagonal(dm2),
                                num.diagonal(m)).real
                            doiod4 += doiod_here4

                            doiod_here5 = -2*w_b*(sum(num.multiply(num.diagonal(mm(dw, m)),
                                                                   num.diagonal(num.conjugate(m)))) 
                                                  + sum(num.multiply(num.diagonal(num.conjugate(mm(dw_plusb, m_plusb))),
                                                                     num.diagonal(num.conjugate(m))))).real
                            doiod5 += doiod_here5
                            assert abs(doiod_here4-doiod_here5) < 1e-11

                            ssym_re_r_t = skewSymmetricPart(num.transpose(r.real))
                            sym_im_r = symmetricPart(r.imaginary)
                            re_grad_od = 4 * w_b * (-ssym_re_r_t )
                            im_grad_od = 4 * w_b * sym_im_r
                            doiod7_here = gradScalarProduct(dw.real, re_grad_od) \
                                          + gradScalarProduct(dw.imaginary, im_grad_od)
                            doiod7 += doiod7_here

                            if print_count:
                                #print k_index, kgii
                                #print "dw", mtools.frobeniusNorm(dw)
                                #print "dm1", mtools.frobeniusNorm(dm1)
                                #print "dm2", mtools.frobeniusNorm(dm2)
                                print "dm2-dm1", \
                                      mtools.frobeniusNorm(dm2-dm1) \
                                      / mtools.frobeniusNorm(dm1)
                                print "doiod_here", doiod_here2, doiod_here4
                                print "b and c", doiod_here2b, doiod_here2c
                                print_count -= 1

                    assert abs(before_oiod-before_oiod2) < 1e-9
                    assert abs(after_oiod-after_oiod2) < 1e-9
                    assert abs(doiod4-doiod5) < 1e-11
                    assert abs(doiod3-doiod5) < 1e-11
                    #assert abs(doiod6-doiod7) < 1e-11
                    print "doiod total", after_oiod-before_oiod, doiod3, \
                          doiod6, doiod7

                #testDerivs(1e-6)
                #testDerivs(1e-5)
                #testDerivs(1e-4)
                #testDerivs(1e-3)
                #raw_input()

                def minfunc(x):
                    temp_mix_matrix = self.getMixMatrix(mix_matrix, x, gradient)
                    temp_sps = self.updateOffsetScalarProducts(orig_sps, temp_mix_matrix)

                    result = self.spreadFunctional(len(pbands), temp_sps, arg)
                    if self.DebugMode:
                        print x, result
                    return result

                def plotfunc(x):
                    temp_mix_matrix = self.getMixMatrix(mix_matrix, x, gradient)
                    temp_sps = self.updateOffsetScalarProducts(orig_sps, temp_mix_matrix)

                    new_grad_od = self.spreadFunctionalGradientOmegaOD(len(pbands), temp_sps, arg)
                    new_grad_d = self.spreadFunctionalGradientOmegaD(len(pbands), temp_sps, arg)
                    sp_od = kDependentMatrixGradientScalarProduct(self.Crystal.KGrid, new_grad_od, gradient)
                    sp_d = kDependentMatrixGradientScalarProduct(self.Crystal.KGrid, new_grad_d, gradient)
                    sp = sp_od + sp_d

                    oi_here = self.omegaI(len(pbands), temp_sps)
                    od = self.omegaD(len(pbands), temp_sps, arg)
                    ood = self.omegaOD(temp_sps)
                    return od, oi_here+ood, oi_here+od+ood, sp, sp_od, sp_d
                           
                step = 0.5/(4*sum(self.KWeights.KWeights))

                if self.InteractivityLevel and (raw_input("see plot? y/n [n]:") == "y"):
                    tools.write1DGnuplotGraphs(plotfunc, -5*step, 5 * step, 
                                               steps = 100, progress = True)
                    raw_input("see plot:")

                xmin = scipy.optimize.brent(minfunc, brack = (0, step))
                # Marzari's fixed step
                #xmin = step

                mix_matrix = self.getMixMatrix(mix_matrix, xmin, gradient)
        except iteration.tIterationStalled:
            pass
        return mix_matrix

def computeMixedBands(crystal, bands, mix_matrix):
    # WARNING! Don't be tempted to insert symmetry code in here, since
    # mix_matrix is of potentially unknown symmetry.

    result = []
    for n in range(len(bands)):
        band = {}

        for k_index in crystal.KGrid:
            # set eigenvalue to 0 since there is no meaning attached to it
            band[k_index] = 0, tools.linearCombination(mix_matrix[k_index][n],
                                                       [bands[i][k_index][1] 
                                                        for i in range(len(bands))])
        result.append(band)
    return result

def computeMixedPeriodicBands(crystal, pbands, mix_matrix):
    # WARNING! Don't be tempted to insert symmetry code in here, since
    # mix_matrix is of potentially unknown symmetry.

    result = []
    for n in range(len(pbands)):
        pband = {}

        for k_index in crystal.KGrid.enlargeAtBothBoundaries():
            reduced_k_index = crystal.KGrid.reducePeriodically(k_index)

            # set eigenvalue to 0 since there is no meaning attached to it
            pband[k_index] = 0.j, tools.linearCombination(mix_matrix[reduced_k_index][n],
                                                          [pbands[i][k_index][1] 
                                                           for i in range(len(pbands))])
        result.append(pband)
    return result

def integrateOverKGrid(k_grid, f):
    return (1./ k_grid.gridPointCount()) \
           * tools.generalSum([f(k_index, k_grid[k_index])
                               for k_index in k_grid])

def computeWanniers(crystal, bands, wannier_grid):
    job = fempy.stopwatch.tJob("computing wannier functions")
    wannier_functions = []

    for n, band in enumerate(bands):
        this_wf = {}
        for wannier_index in wannier_grid:
            R = wannier_grid[wannier_index]
            def function_in_integral(k_index, k):
                k = crystal.KGrid[k_index]
                return cmath.exp(1.j * mtools.sp(k, R)) * band[k_index][1]

            this_wf[wannier_index] = integrateOverKGrid(crystal.KGrid, 
                                                        function_in_integral)
        wannier_functions.append(this_wf)
    job.done()
    return wannier_functions

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

def generateHierarchicGaussians(crystal, typecode):
    for l in tools.generateAllPositiveIntegerTuples(2,1):
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
                                                     crystal.NodeNumberAssignment)

def generateRandomGaussians(crystal, typecode):
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
                                                     crystal.NodeNumberAssignment)
    
def guessInitialMixMatrix(crystal, bands, sp):
    # generate the gaussians
    gaussians = []
    gaussian_it = generateRandomGaussians(crystal, num.Complex)
    for n in range(len(bands)):
        gaussians.append(gaussian_it.next())

    # project the gaussians
    projected_bands = []
    projected_bands_co = []
    for n in range(len(bands)):
        projected_band = {}
        projected_band_co = {}

        for k_index in crystal.KGrid:
            mf = fempy.mesh_function.discretizeFunction(
                crystal.Mesh, lambda x: 0., num.Complex, 
                number_assignment = crystal.NodeNumberAssignment)
            coordinates = num.zeros((len(bands),), num.Complex)
            for m in range(len(bands)):
                coordinates[m] = sp(gaussians[n], bands[m][k_index][1])
                mf += coordinates[m] * bands[m][k_index][1]
            projected_band[k_index] = mf
            projected_band_co[k_index] = coordinates
        projected_bands.append(projected_band)
        projected_bands_co.append(projected_band_co)

    # orthogonalize the projected gaussians
    mix_matrix = {}
    for k_index in crystal.KGrid:
        # calculate scalar products
        my_sps = num.zeros((len(bands), len(bands)), num.Complex)
        for n in range(len(bands)):
            for m in range(m+1):
                my_sp = sp(projected_bands[n][k_index], projected_bands[m][k_index])
                my_sps[n,m] = my_sp
                my_sps[m,n] = my_sp.conjugate()

        inv_sqrt_my_sps = la.inverse(la.cholesky_decomposition(my_sps))

        mix_matrix[k_index] = num.zeros((len(bands), len(bands)), num.Complex)
        for n in range(len(bands)):
            # determine and compute correct mixture of projected bands
            mix_matrix[k_index][n] = tools.linearCombination(
                inv_sqrt_my_sps[n], 
                [projected_bands_co[i][k_index] 
                 for i in range(len(bands))])
                
    return mix_matrix

def run():
    #print "WARNING: Id+dW still enabled"
    debug_mode = raw_input("enable debug mode? [n]") == "y"
    interactivity_level = int(raw_input("interactivity level? [0]"))
    random.seed(10)

    job = fempy.stopwatch.tJob("loading")
    crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
    job.done()

    crystal = crystals[-1]

    assert abs(integrateOverKGrid(
        crystal.KGrid, 
        lambda k_index, k: cmath.exp(1j*mtools.sp(k, num.array([5.,17.]))))) < 1e-10
    assert abs(1- integrateOverKGrid(
        crystal.KGrid, 
        lambda k_index, k: cmath.exp(1j*mtools.sp(k, num.array([0.,0.]))))) < 1e-10

    sp = fempy.mesh_function.tScalarProductCalculator(crystal.NodeNumberAssignment,
                                                      crystal.MassMatrix)
                                                      
    #gaps, clusters = pc.analyzeBandStructure(crystal.Bands)

    bands = crystal.Bands[1:6]
    pbands = crystal.PeriodicBands[1:6]

    job = fempy.stopwatch.tJob("guessing initial mix")
    mix_matrix = guessInitialMixMatrix(crystal, 
                                       bands,
                                       sp)
    job.done()

    minimizer = tMarzariSpreadMinimizer(crystal, sp, debug_mode, interactivity_level)
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
