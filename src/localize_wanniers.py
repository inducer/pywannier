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
            result[key] = self[key].H
        return result
        
    def transpose():
        result = tDictionaryOfMatrices()
        for key in self:
            result[key] = self[key].T
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
    T = property(transpose)
    H = property(hermite)




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
    return 0.5*(matrix+matrix.T)

def skewSymmetricPart(matrix):
    return 0.5*(matrix-matrix.T)

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

def minimizeByGradientDescent(x, f, grad, x_plus_alpha_grad, step):
    observer = iteration.makeObserver(min_change = 1e-3, max_unchanged = 3)
    observer.reset()

    last_fval = f(x)
    try:
        while True:
            grad_here = grad(start)
            def minfunc(alpha):
                return f(x_plus_alpha_grad(x, alpha, grad_here))
            alpha, fval, iter, funcalls  = scipy.optimize.brent(
                minfunc, brack = (0, -step), full_output = True)
            observer.addDataPoint(fval)
            print "Target value: %f (D:%f) - %d calls in last step - step size %f" % (
                fval, fval - last_fval, funcalls, alpha)
            last_fval = fval

            x = x_plus_alpha_grad(x, alpha, grad_here)
    except iteration.tIterationStalled:
        pass
    except iteration.tIterationStopped:
        pass
    return start

def minimizeByCG(x, f, grad, x_plus_alpha_grad, step, sp, log_filenames = None):
    # from Shewchuk's paper, p. 48
    # Polak-Ribi`ere with implicit restart

    d = last_r = -grad(x)
    observer = iteration.makeObserver(min_change = 1e-5, max_unchanged = 3)
    observer.reset()

    last_fval = f(x)

    if log_filenames is not None:
        target_log = file(log_filenames[0], "w")
        step_log = file(log_filenames[1], "w")
        step_count = 0

        target_log.write("%d\t%f\n" % (step_count, last_fval))

    try:
        while True:
            def minfunc(alpha):
                return f(x_plus_alpha_grad(x, alpha, d))
            alpha, fval, iter, funcalls = scipy.optimize.brent(
                minfunc, brack = (0, step), full_output = True)
            observer.addDataPoint(fval)
            print "Target value: %f (D:%f) - %d calls in last step - step size %f" % (
                fval, fval - last_fval, funcalls, alpha)
            last_fval = fval

            x = x_plus_alpha_grad(x, alpha, d)
            r = -grad(x)
            beta = max(0, sp(r, r - last_r)/sp(last_r, last_r))
            d = r + beta * d
            last_r = r

            if log_filenames is not None:
                step_log.write("%d\t%f\n" % (step_count, alpha))
                step_count += 1
                target_log.write("%d\t%f\n" % (step_count, last_fval))

    except iteration.tIterationStalled:
        pass
    except iteration.tIterationStopped:
        pass
    return x

def minimizeByFixedStep(x, f, grad, x_plus_alpha_grad, step, sp):
    d = -grad(x)
    observer = iteration.makeObserver(min_change = 1e-3, max_unchanged = 3)
    observer.reset()

    last_fval = f(x)
    try:
        while True:
            alpha = step
            fval = f(x_plus_alpha_grad(x, alpha, d))
            print "Target value: %f (D:%f)" % (fval, fval - last_fval)
            observer.addDataPoint(fval)
            last_fval = fval

            x = x_plus_alpha_grad(x, alpha, d)
            d = -grad(x)
    except iteration.tIterationStalled:
        print "Continuing with fine-grained CG"
        return minimizeByCG(x, f, grad, x_plus_alpha_grad, step, sp, 
                            (",,cg_target_log.data", ",,cg_step_log.data"))
    except iteration.tIterationStopped:
        pass
    return x

# K space weights -------------------------------------------------------------
class tKSpaceDirectionalWeights:
    def __init__(self, crystal):
        self.HalfTheKGridIndexIncrements = []
        dimensions = len(crystal.Lattice.DirectLatticeBasis)
        for i in range(dimensions):
            direction = [0] * dimensions
            direction[i] = 1
            self.HalfTheKGridIndexIncrements.append(tuple(direction))

        self.KGridIndexIncrements = self.HalfTheKGridIndexIncrements + \
                                    [tools.negateTuple(kgii) 
                                     for kgii in self.HalfTheKGridIndexIncrements]

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
                assert abs(my_sum - tools.delta(i, j)) < 1e-15

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
            for kgii_index, kgii in enumerate(self.KWeights.HalfTheKGridIndexIncrements):
                #added_tuple = self.Crystal.KGrid.reducePeriodically(
                    #tools.addTuples(k_index, kgii))
                added_tuple = tools.addTuples(k_index, kgii)

                mat = num.zeros((n_bands, n_bands), num.Complex)
                for i in range(n_bands):
                    for j in range(n_bands):
                        mat[i,j] = self.ScalarProductCalculator(pbands[i][added_tuple][1], 
                                                                pbands[j][k_index][1])
                scalar_products[k_index, kgii] = mat

                red_tuple = self.Crystal.KGrid.reducePeriodically(added_tuple)
                negated_kgii = tools.negateTuple(kgii)
                scalar_products[red_tuple, negated_kgii] = mat.H

        self.checkScalarProducts(scalar_products)
        return scalar_products

    def checkInitialScalarProducts(self, pbands, scalar_products):
        if not self.DebugMode:
            return
        n_bands = len(pbands)

        violations = []

        for k_index in self.Crystal.KGrid:
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                added_tuple = tools.addTuples(k_index, kgii)

                mat = num.zeros((n_bands, n_bands), num.Complex)
                for i in range(n_bands):
                    for j in range(n_bands):
                        mat[i,j] = self.ScalarProductCalculator(pbands[i][added_tuple][1], 
                                                                pbands[j][k_index][1])
                err = mtools.frobeniusNorm(mat - scalar_products[k_index, kgii]) 
                if err > 1e-13:
                    violations.append((k_index, kgii, err))

        if violations:
            print "WARNING: M^{k,b} = (M^{k+b,-b})^H violated"
            print violations

        return scalar_products

    def updateOffsetScalarProducts(self, scalar_products, mix_matrix):
        new_scalar_products = {}
        for k_index in self.Crystal.KGrid:
            if self.DebugMode:
                assert mtools.unitarietyError(mix_matrix[k_index]) < 1e-8

            for kgii in self.KWeights.HalfTheKGridIndexIncrements:
                added_tuple = self.Crystal.KGrid.reducePeriodically(
                    tools.addTuples(k_index, kgii))

                mat = mix_matrix[added_tuple] * scalar_products[k_index, kgii] * mix_matrix[k_index].H

                new_scalar_products[k_index, kgii] = mat

                red_tuple = self.Crystal.KGrid.reducePeriodically(added_tuple)
                negated_kgii = tools.negateTuple(kgii)
                new_scalar_products[red_tuple, negated_kgii] = mat.H

        self.checkScalarProducts(new_scalar_products)
        return new_scalar_products

    def checkScalarProducts(self, scalar_products):
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
                              * self.KWeights.KGridIncrements[kgii_index] \
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

        gradient = tDictionaryOfMatrices()
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)

                # Omega_OD part
                r = num.multiply(num.hermite(m), m_diagonal)

                if self.DebugMode:
                    r2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r2[i,j] = m[j,i].conjugate() * m[j,j]
                    assert mtools.frobeniusNorm(r-r2) < 1e-15

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (-skewSymmetricPart(r.real.T)
                           +1j*symmetricPart(r.imaginary))

                # Omega_D part
                r_tilde = num.divide(m.H, num.conjugate(m_diagonal))

                if self.DebugMode:
                    r_tilde2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r_tilde2[i,j] = (m[j,i] / m[j,j]).conjugate()
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
                          (-skewSymmetricPart(t.imaginary.T)
                           -1j*symmetricPart(t.real))

            gradient[k_index] = result
        return gradient

    def spreadFunctionalGradientMarzariOmegaOD(self, n_bands, scalar_products, arg, wannier_centers = None):
        ### WRONG!!! DELETE ME!
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        gradient = tDictionaryOfMatrices()
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)

                # Omega_OD part
                r = num.multiply(m.H, m_diagonal)

                if self.DebugMode:
                    r2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r2[i,j] = m[j,i].conjugate() * m[j,j]
                    assert mtools.frobeniusNorm(r-r2) < 1e-15

                result += -2. * self.KWeights.KWeights[kgii_index] * \
                          (r-r.H)
            gradient[k_index] = result
        return gradient
        ### WRONG!!! DELETE ME!

    def spreadFunctionalGradientOmegaOD(self, n_bands, scalar_products, arg, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        gradient = tDictionaryOfMatrices()
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)

                # Omega_OD part
                r = num.multiply(m.H, m_diagonal)

                if self.DebugMode:
                    r2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r2[i,j] = m[j,i].conjugate() * m[j,j]
                    assert mtools.frobeniusNorm(r-r2) < 1e-15

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (-skewSymmetricPart(r.real.T)
                           +1j*symmetricPart(r.imaginary))
            gradient[k_index] = result
        return gradient

    def spreadFunctionalGradientOmegaD(self, n_bands, scalar_products, arg, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        gradient = tDictionaryOfMatrices()
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)

                # Omega_D part
                r_tilde = num.divide(m.H, num.conjugate(m_diagonal))

                if self.DebugMode:
                    r_tilde2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r_tilde2[i,j] = (m[j,i] / m[j,j]).conjugate()
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
                          (-skewSymmetricPart(t.imaginary.T)
                           -1j*symmetricPart(t.real))

            gradient[k_index] = result
        return gradient

    def spreadFunctionalGradientMarzariOmegaD(self, n_bands, scalar_products, arg, wannier_centers = None):
        ### WRONG!!! DELETE ME!
        if wannier_centers is None:
            wannier_centers = self.wannierCenters(n_bands, scalar_products, arg)

        gradient = tDictionaryOfMatrices()
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)

                # Omega_D part
                r_tilde = num.divide(m.H, num.conjugate(m_diagonal))

                if self.DebugMode:
                    r_tilde2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r_tilde2[i,j] = (m[j,i] / m[j,j]).conjugate()
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

                result += -2. * self.KWeights.KWeights[kgii_index] * \
                          (t+t.H)
            gradient[k_index] = result
        return gradient
        ### WRONG!!! DELETE ME!

    def getMixMatrix(self, prev_mix_matrix, factor, gradient):
        temp_mix_matrix = {}
        for k_index in self.Crystal.KGrid:
            dW = factor * gradient[k_index]
            if self.DebugMode:
                assert mtools.skewHermiticityError(dW) < 1e-13

            exp_dW = mtools.matrixExpByDiagonalization(dW)
            if self.DebugMode:
                assert mtools.unitarietyError(exp_dW) < 1e-13

            temp_mix_matrix[k_index] = exp_dW * prev_mix_matrix[k_index]

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
                assert mtools.frobeniusNorm(sps_direct[k_index, kgii]
                                           - sps_updated[k_index, kgii]) < 1e-13

        arg = tContinuityAwareArg()
        sf1 = self.spreadFunctional(len(pbands), sps_updated, arg.copy())
        sf2 = self.spreadFunctional(len(pbands), sps_direct, arg.copy())
        assert abs(sf1-sf2) < 1e-10

        job.done()

    def minimizeOmegaODByCodiagonalization(self, raw_scalar_products, mix_matrix):
        """scalar_products are understood to be before application of the
        mix_matrix specified.
        """
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
            new_mix_matrix[k_index] = q.H * mix_matrix[k_index]

        if self.DebugMode:
            sps_post = self.updateOffsetScalarProducts(raw_scalar_products, new_mix_matrix)
            print "od after pre", self.omegaOD(sps_post)
        return new_mix_matrix

    def minimizeSpread(self, pbands, mix_matrix):
        if self.DebugMode:
            for ii in self.Crystal.KGrid:
                assert mtools.unitarietyError(mix_matrix[ii]) < 5e-3

        self.testSPUpdater(pbands, mix_matrix)

        job = fempy.stopwatch.tJob("computing scalar products")
        orig_sps = self.computeOffsetScalarProducts(pbands)
        self.checkInitialScalarProducts(pbands, orig_sps)
        job.done()

        oi = self.omegaI(len(pbands), orig_sps)
        arg = tSimpleArg()

        observer = iteration.makeObserver(min_change = 1e-3, max_unchanged = 3)
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

                    for k_index in self.Crystal.KGrid:
                        for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                            added_tup = self.Crystal.KGrid.reducePeriodically(tools.addTuples(k_index, kgii))

                            w_b = self.KWeights.KWeights[kgii_index]

                            new_m = temp_sps[k_index,kgii]
                            new_m_diagonal = num.diagonal(temp_sps[k_index,kgii])
                            m = sps[k_index,kgii]
                            m_diagonal = num.diagonal(m)
                            dw = x * gradient[k_index]

                            dw_plusb = x * gradient[added_tup]
                            m_plusb = m.H
                            dm1 = new_m - m
                            dm2 = dw_plusb*m + m*dw.H

                            before_oiod_here = w_b * (len(pbands)-tools.norm2squared(m_diagonal))
                            after_oiod_here = w_b * (len(pbands)-tools.norm2squared(new_m_diagonal))

                            before_oiod2 += before_oiod_here
                            after_oiod2 += after_oiod_here
                            doiod_here2  = after_oiod_here-before_oiod_here
                            doiod_here2b = w_b * (+tools.norm2squared(m_diagonal)
                                                  -tools.norm2squared(new_m_diagonal))
                            assert abs(doiod_here2 - doiod_here2b) < 1e-11

                            doiod_here2c = 2 * w_b * sum(num.multiply(-new_m_diagonal+m_diagonal,
                                                                      num.conjugate(m_diagonal))).real

                            r = num.multiply(m.H, m_diagonal)
                            doiod_here3 = -4*w_b*num.trace((dw*r).real)
                            doiod3 += doiod_here3

                            half_doiod_here3 = -2*w_b*num.trace(mm(dw, r)).real

                            doiod_here4 = -2*w_b*mtools.sp(
                                num.diagonal(dm2),
                                m_diagonal).real
                            doiod4 += doiod_here4

                            half_a_doiod_here5 = -2*w_b*sum(num.multiply(num.diagonal(dw_plusb*m_plusb.H),
                                                                         num.conjugate(m_diagonal))).real
                            half_b_doiod_here5 = -2*w_b*sum(num.multiply(num.diagonal(num.conjugate(dw* m.H)),
                                                                         num.diagonal(num.conjugate(m)))).real
                            doiod_here5 = half_a_doiod_here5 + half_b_doiod_here5
                            doiod5 += doiod_here5
                            assert abs(doiod_here4-doiod_here5) < 1e-11

                            ssym_re_r_t = skewSymmetricPart(r.real.T)
                            sym_im_r = symmetricPart(r.imaginary)
                            re_grad_od = 4 * w_b * (-ssym_re_r_t )
                            im_grad_od = 4 * w_b * sym_im_r
                            doiod_here7 = gradScalarProduct(dw.real, re_grad_od) \
                                         + gradScalarProduct(dw.imaginary, im_grad_od)
                            doiod7 += doiod_here7

                            if print_count:
                                #print k_index, kgii
                                #print "dw", mtools.frobeniusNorm(dw)
                                #print "dm1", mtools.frobeniusNorm(dm1)
                                #print "dm2", mtools.frobeniusNorm(dm2)
                                #print "dm2-dm1", \
                                      #mtools.frobeniusNorm(dm2-dm1) \
                                      #/ mtools.frobeniusNorm(dm1), \
                                      #" - abs:", mtools.frobeniusNorm(dm2-dm1)
                                #print "doiod_here", doiod_here2, doiod_here4
                                #print "b and c", doiod_here2b, doiod_here2c
                                print "doiod_here", k_index, kgii, half_doiod_here3, half_a_doiod_here5, half_b_doiod_here5
                                print_count -= 1

                    assert abs(before_oiod-before_oiod2) < 1e-9
                    assert abs(after_oiod-after_oiod2) < 1e-9
                    assert abs(doiod4-doiod5) < 1e-11
                    #assert abs(doiod3-doiod5) < 1e-11
                    print "doiod total", after_oiod-before_oiod, doiod3, \
                          doiod5, doiod7

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

                    marz_grad_od = self.spreadFunctionalGradientMarzariOmegaOD(len(pbands), temp_sps, arg)
                    marz_grad_d = self.spreadFunctionalGradientMarzariOmegaD(len(pbands), temp_sps, arg)
                    marz_sp_od = kDependentMatrixGradientScalarProduct(self.Crystal.KGrid, marz_grad_od, gradient)
                    marz_sp_d = kDependentMatrixGradientScalarProduct(self.Crystal.KGrid, marz_grad_d, gradient)
                    marz_sp = marz_sp_od + marz_sp_d

                    new_grad_od = self.spreadFunctionalGradientOmegaOD(len(pbands), temp_sps, arg)
                    new_grad_d = self.spreadFunctionalGradientOmegaD(len(pbands), temp_sps, arg)
                    sp_od = kDependentMatrixGradientScalarProduct(self.Crystal.KGrid, new_grad_od, gradient)
                    sp_d = kDependentMatrixGradientScalarProduct(self.Crystal.KGrid, new_grad_d, gradient)
                    sp = sp_od + sp_d

                    oi_here = self.omegaI(len(pbands), temp_sps)
                    od = self.omegaD(len(pbands), temp_sps, arg)
                    ood = self.omegaOD(temp_sps)
                    return oi_here+od+ood, sp, marz_sp
                           
                step = 0.5/(4*sum(self.KWeights.KWeights))

                if self.InteractivityLevel and (raw_input("see plot? y/n [n]:") == "y"):
                    tools.write1DGnuplotGraphs(plotfunc, -5*step, 5 * step, 
                                               steps = 100, progress = True)
                    raw_input("see plot:")

                xmin = scipy.optimize.brent(minfunc, brack = (0, -step))
                # Marzari's fixed step
                #xmin = step

                mix_matrix = self.getMixMatrix(mix_matrix, xmin, gradient)
        except iteration.tIterationStalled:
            pass
        except iteration.tIterationStopped:
            pass
        return mix_matrix

    def minimizeSpread2(self, pbands, mix_matrix):
        if self.DebugMode:
            for ii in self.Crystal.KGrid:
                assert mtools.unitarietyError(mix_matrix[ii]) < 5e-3

        self.testSPUpdater(pbands, mix_matrix)

        job = fempy.stopwatch.tJob("computing scalar products")
        orig_sps = self.computeOffsetScalarProducts(pbands)
        self.checkInitialScalarProducts(pbands, orig_sps)
        job.done()

        arg = tSimpleArg()

        def f(mix_matrix):
            temp_sps = self.updateOffsetScalarProducts(orig_sps, mix_matrix)
            result = self.spreadFunctional(len(pbands), temp_sps, arg)
            return result

        def grad(mix_matrix):
            temp_sps = self.updateOffsetScalarProducts(orig_sps, mix_matrix)
            return self.spreadFunctionalGradient(len(pbands), temp_sps, arg)

        def sp(m1, m2):
            return kDependentMatrixGradientScalarProduct(self.Crystal.KGrid, m1, m2)

        return minimizeByCG(tDictionaryOfMatrices(mix_matrix), 
                            f, grad, self.getMixMatrix,
                            step = 0.5/(4*sum(self.KWeights.KWeights)),
                            sp = sp,
                            log_filenames = (",,cg_target_log.data", ",,cg_step_log.data"))

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
    n_total = 0
    for gi in multicell_grid:
        fvec = func_on_multicell_grid[gi].vector() / avg_phase_term
            
        for z in fvec:
            if abs(z) >= 1e-2:
                my_phase_diff_sum += abs(cmath.log(z).imag)
                n += 1
            n_total += 1
    return my_phase_diff_sum / (n * math.pi), n, n_total

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
            arg = sigma_inv*(point - center)
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
    mix_matrix = tDictionaryOfMatrices()
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
    debug_mode = raw_input("enable debug mode? [n]") == "y"
    ilevel_str = raw_input("interactivity level? [0]")
    interactivity_level = (ilevel_str) and int(ilevel_str) or 0
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
                                                      
    gaps, clusters = pc.analyzeBandStructure(crystal.Bands)
    print "Gaps:", gaps
    print "Clusters:", clusters

    bands = crystal.Bands[1:4]
    pbands = crystal.PeriodicBands[1:4]

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
                                     limits = [(-1,2)] * 2)

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
