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




class DictionaryOfMatrices(pytools.DictionaryOfArithmeticTypes):
    def _get_empty_self(self):
        return DictionaryOfMatrices()

    def conjugate():
        result = self._get_empty_self()
        for key in self:
            result[key] = num.conjugate(self[key])
        return result

    def _hermite(self):
        result = self._get_empty_self()
        for key in self:
            result[key] = self[key].H
        return result

    def _transpose(self):
        result = self._get_empty_self()
        for key in self:
            result[key] = self[key].T
        return result

    def _getreal():
        result = self._get_empty_self()
        for key in self:
            result[key] = self[key].real
        return result

    def _getimaginary():
        result = self._get_empty_self()
        for key in self:
            result[key] = self[key].real
        return result

    def matrix_multiply(self, other):
        return self.binary_operator(other, num.matrixmultiply)

    real = property(_getreal)
    imaginary = property(_getimaginary)
    T = property(_transpose)
    H = property(_hermite)




# tools -----------------------------------------------------------------------
def frobenius_norm_off_diagonal_squared(a):
    result = 0
    for i,j in a.indices():
        if i != j:
            result += abs(a[i,j])**2
    return result

def matrix_to_list(num_mat):
    if len(num_mat.shape) == 1:
        return [x for x in num_mat]
    else:
        return [matrix_to_list(x) for x in num_mat]

def make_random_k_dependent_skew_hermitian_matrix(crystal, size, tc):
    matrix = DicitionaryOfMatrices
    for k_index in crystal.KGrid:
        matrix[k_index] = randomized.make_random_skewhermitian_matrix(size, tc)
    return matrix

def grad_scalar_product(a1, a2):
    rp = num.multiply(a1.real, a2.real)
    ip = num.multiply(a1.imaginary, a2.imaginary)
    return num.sum(rp) + num.sum(ip)

def k_dependent_matrix_gradient_scalar_product(k_grid, a1, a2):
    # we can't use complex multiplication since our "complex" number
    # here is just a convenient notation for a gradient, so the real
    # and imaginary parts have to stay separate.
    sp = 0.
    for k_index in k_grid:
        rp = num.multiply(a1[k_index].real, a2[k_index].real)
        ip = num.multiply(a1[k_index].imaginary, a2[k_index].imaginary)
        sp += num.sum(rp) + num.sum(ip)
    return sp / k_grid.grid_point_count()

def operate_on_k_dependent_matrix(k_grid, a, m_op):
    result = {}
    for k_index in k_grid:
        result[k_index] = m_op(a[k_index])
    return result

def operate_on_k_dependent_matrices(k_grid, a1, a2, m_op):
    result = {}
    for k_index in k_grid:
        result[k_index] = m_op(a1[k_index], a2[k_index])
    return result

def frobenius_norm_on_k_dependent_matrices(k_grid, mat):
    result = 0
    for k_index in k_grid:
        result += comp.norm_frobenius(mat[k_index])
    return result

def symmetric_part(matrix):
    return 0.5*(matrix+matrix.T)

def skew_symmetric_part(matrix):
    return 0.5*(matrix-matrix.T)

def arg(z):
    return cmath.log(z).imag

def minimize_by_gradient_descent(x, f, grad, x_plus_alpha_grad, step):
    observer = iteration.make_observer(min_change = 1e-3, max_unchanged = 3)
    observer.reset()

    last_fval = f(x)
    try:
        while True:
            grad_here = grad(start)
            def minfunc(alpha):
                return f(x_plus_alpha_grad(x, alpha, grad_here))
            alpha, fval, iter, funcalls  = scipy.optimize.brent(
                minfunc, brack = (0, -step), full_output = True)
            observer.add_data_point(fval)
            print "Target value: %f (D:%f) - %d calls in last step - step size %f" % (
                fval, fval - last_fval, funcalls, alpha)
            last_fval = fval

            x = x_plus_alpha_grad(x, alpha, grad_here)
    except iteration.IterationStalled:
        pass
    except iteration.IterationStopped:
        pass
    return start

def minimize_by_cg(x, f, grad, x_plus_alpha_grad, step, sp, log_filenames = None):
    # from Shewchuk's paper, p. 48
    # Polak-Ribi`ere with implicit restart

    d = last_r = -grad(x)
    observer = iteration.make_observer(min_change = 1e-4, max_unchanged = 3)
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
            observer.add_data_point(fval)
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

    except iteration.IterationStalled:
        pass
    except iteration.IterationStopped:
        pass
    return x

def minimize_by_fixed_step(x, f, grad, x_plus_alpha_grad, step, sp):
    d = -grad(x)
    observer = iteration.make_observer(min_change = 1e-3, max_unchanged = 3)
    observer.reset()

    last_fval = f(x)
    try:
        while True:
            alpha = step
            fval = f(x_plus_alpha_grad(x, alpha, d))
            print "Target value: %f (D:%f)" % (fval, fval - last_fval)
            observer.add_data_point(fval)
            last_fval = fval

            x = x_plus_alpha_grad(x, alpha, d)
            d = -grad(x)
    except iteration.IterationStalled:
        print "Continuing with fine-grained CG"
        return minimize_by_cg(x, f, grad, x_plus_alpha_grad, step, sp, 
                            (",,cg_target_log.data", ",,cg_step_log.data"))
    except iteration.IterationStopped:
        pass
    return x




# K space weights -------------------------------------------------------------
class KSpaceDirectionalWeights:
    def __init__(self, crystal):
        self.HalfTheKGridIndexIncrements = []
        dimensions = len(crystal.Lattice.DirectLatticeBasis)
        for i in range(dimensions):
            direction = [0] * dimensions
            direction[i] = 1
            self.HalfTheKGridIndexIncrements.append(tuple(direction))

        self.KGridIndexIncrements = self.HalfTheKGridIndexIncrements + \
                                    [pytools.negate_tuple(kgii) 
                                     for kgii in self.HalfTheKGridIndexIncrements]

        self.KGridIncrements = [crystal.KGrid[kgii] - crystal.KGrid[0,0]
                                for kgii in self.KGridIndexIncrements]

        self.KWeights = [0.5 / comp.norm_2_squared(kgi)
                         for kgi in self.KGridIncrements]

        # verify...
        for i in range(dimensions):
            for j in range(dimensions):
                my_sum = 0
                for kgi_index, kgi in enumerate(self.KGridIncrements):
                    my_sum += self.KWeights[kgi_index]*kgi[i]*kgi[j]
                assert abs(my_sum - pytools.delta(i, j)) < 1e-15




# Marzari minimization --------------------------------------------------------
class MarzariSpreadMinimizer:
    def __init__(self, crystal, spc, debug_mode = True, interactivity_level = 0):
        self.Crystal = crystal
        self.KWeights = KSpaceDirectionalWeights(crystal)
        self.ScalarProductCalculator = spc
        self.DebugMode = debug_mode
        self.InteractivityLevel = interactivity_level

    def compute_offset_scalar_products(self, pbands):
        n_bands = len(pbands)
        scalar_products = {}

        for k_index in self.Crystal.KGrid:
            for kgii_index, kgii in enumerate(self.KWeights.HalfTheKGridIndexIncrements):
                #added_tuple = self.Crystal.KGrid.reduce_periodically(
                    #pytools.add_tuples(k_index, kgii))
                added_tuple = pytools.add_tuples(k_index, kgii)

                mat = num.zeros((n_bands, n_bands), num.Complex)
                for i in range(n_bands):
                    for j in range(n_bands):
                        mat[i,j] = self.ScalarProductCalculator(pbands[i][added_tuple][1], 
                                                                pbands[j][k_index][1])
                scalar_products[k_index, kgii] = mat

                red_tuple = self.Crystal.KGrid.reduce_periodically(added_tuple)
                negated_kgii = pytools.negate_tuple(kgii)
                scalar_products[red_tuple, negated_kgii] = mat.H

        self.check_scalar_products(scalar_products)
        return scalar_products

    def check_initial_scalar_products(self, pbands, scalar_products):
        if not self.DebugMode:
            return
        n_bands = len(pbands)

        violations = []

        for k_index in self.Crystal.KGrid:
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                added_tuple = pytools.add_tuples(k_index, kgii)

                mat = num.zeros((n_bands, n_bands), num.Complex)
                for i in range(n_bands):
                    for j in range(n_bands):
                        mat[i,j] = self.ScalarProductCalculator(pbands[i][added_tuple][1], 
                                                                pbands[j][k_index][1])
                err = comp.norm_frobenius(mat - scalar_products[k_index, kgii]) 
                if err > 1e-13:
                    violations.append((k_index, kgii, err))

        if violations:
            print "WARNING: M^{k,b} = (M^{k+b,-b})^H violated"
            print violations

        return scalar_products

    def update_offset_scalar_products(self, scalar_products, mix_matrix):
        new_scalar_products = {}
        for k_index in self.Crystal.KGrid:
            if self.DebugMode:
                assert toybox.unitariety_error(mix_matrix[k_index]) < 1e-8

            for kgii in self.KWeights.HalfTheKGridIndexIncrements:
                added_tuple = self.Crystal.KGrid.reduce_periodically(
                    pytools.add_tuples(k_index, kgii))

                mat = mix_matrix[added_tuple] * scalar_products[k_index, kgii] * mix_matrix[k_index].H

                new_scalar_products[k_index, kgii] = mat

                red_tuple = self.Crystal.KGrid.reduce_periodically(added_tuple)
                negated_kgii = pytools.negate_tuple(kgii)
                new_scalar_products[red_tuple, negated_kgii] = mat.H

        self.check_scalar_products(new_scalar_products)
        return new_scalar_products

    def check_scalar_products(self, scalar_products):
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
                        argfiles[i].write("%f\t%f\t%f\n" %(where[0], where[1]))
            for af in argfiles:
                af.close()
            for mf in magfiles:
                mf.close()
            raw_input("[magnitude/argument plot ready]")

    def wannier_centers(self, n_bands, scalar_products):
        wannier_centers = []
        for n in range(n_bands):
            result = num.zeros((2,), num.Float)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                for k_index in self.Crystal.KGrid:
                    if scalar_products[k_index, kgii] is None:
                        continue

                    result -= self.KWeights.KWeights[kgii_index] \
                              * self.KWeights.KGridIncrements[kgii_index] \
                              * arg(scalar_products[k_index, kgii][n,n])
            result /= self.Crystal.KGrid.grid_point_count()
            wannier_centers.append(result)
        return wannier_centers

    def spread_functional(self, n_bands, scalar_products):
        wannier_centers = self.wannier_centers(n_bands, scalar_products)

        total_spread_f = 0
        for n in range(n_bands):
            mean_r_squared = 0
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                for k_index in self.Crystal.KGrid:
                    if scalar_products[k_index, kgii] is None:
                        continue

                    mean_r_squared += self.KWeights.KWeights[kgii_index] \
                                      * (1 - abs(scalar_products[k_index, kgii][n,n])**2 
                                         + arg(scalar_products[k_index, kgii][n,n])**2)
            mean_r_squared /= self.Crystal.KGrid.grid_point_count()
            total_spread_f += mean_r_squared - comp.norm_2_squared(wannier_centers[n])
        return total_spread_f

    def omega_i(self, n_bands, scalar_products):
        omega_i = 0
        for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
            for k_index in self.Crystal.KGrid:
                if scalar_products[k_index, kgii] is None:
                    continue

                omega_i += self.KWeights.KWeights[kgii_index] \
                           * (n_bands - comp.norm_frobenius_squared(scalar_products[k_index, kgii]))
        return omega_i / self.Crystal.KGrid.grid_point_count()

    def omega_od(self, scalar_products):
        omega_od = 0
        for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
            for k_index in self.Crystal.KGrid:
                if scalar_products[k_index, kgii] is None:
                    continue

                omega_od += self.KWeights.KWeights[kgii_index] \
                           * (frobenius_norm_off_diagonal_squared(scalar_products[k_index, kgii]))
        return omega_od / self.Crystal.KGrid.grid_point_count()

    def omega_d(self, n_bands, scalar_products, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannier_centers(n_bands, scalar_products)

        omega_d = 0.
        for n in range(n_bands):
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                for k_index in self.Crystal.KGrid:
                    if scalar_products[k_index, kgii] is None:
                        continue

                    b = self.KWeights.KGridIncrements[kgii_index]

                    omega_d += self.KWeights.KWeights[kgii_index] \
                               * (arg(scalar_products[k_index,kgii][n,n]) \
                                  + (wannier_centers[n]*b))**2
        return omega_d / self.Crystal.KGrid.grid_point_count()

    def spread_functional_via_omegas(self, n_bands, scalar_products, wannier_centers = None):
        return self.omega_i(n_bands, scalar_products) + \
               self.omega_od(scalar_products) + \
               self.omega_d(n_bands, scalar_products, wannier_centers)

    def spread_functional_gradient(self, n_bands, scalar_products, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannier_centers(n_bands, scalar_products)

        gradient = DictionaryOfMatrices()
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
                    assert comp.norm_frobenius(r-r2) < 1e-15

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (-skew_symmetric_part(r.real.T)
                           +1j*symmetric_part(r.imaginary))

                # Omega_D part
                r_tilde = num.divide(m.H, num.conjugate(m_diagonal))

                if self.DebugMode:
                    r_tilde2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            r_tilde2[i,j] = (m[j,i] / m[j,j]).conjugate()
                    assert comp.norm_frobenius(r_tilde-r_tilde2) < 1e-13

                q = num.zeros((n_bands,), num.Complex)
                for n in range(n_bands):
                    q[n] = arg(m_diagonal[n])

                for n in range(n_bands):
                    q[n] += self.KWeights.KGridIncrements[kgii_index] \
                            * wannier_centers[n]
                t = num.multiply(r_tilde, q)
                if self.DebugMode:
                    t2 = num.zeros((n_bands, n_bands), num.Complex)
                    for i in range(n_bands):
                        for j in range(n_bands):
                            t2[i,j] = r_tilde[i,j] * q[j]
                    assert comp.norm_frobenius(t-t2) < 1e-15

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (-skew_symmetric_part(t.imaginary.T)
                           -1j*symmetric_part(t.real))

            gradient[k_index] = result
        return gradient

    def spread_functional_gradient_omega_od(self, n_bands, scalar_products, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannier_centers(n_bands, scalar_products)

        gradient = DictionaryOfMatrices()
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)

                # Omega_OD part
                r = num.multiply(m.H, m_diagonal)

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (-skew_symmetric_part(r.real.T)
                           +1j*symmetric_part(r.imaginary))
            gradient[k_index] = result
        return gradient

    def spread_functional_gradient_omega_d(self, n_bands, scalar_products, wannier_centers = None):
        if wannier_centers is None:
            wannier_centers = self.wannier_centers(n_bands, scalar_products)

        gradient = DictionaryOfMatrices()
        for k_index in self.Crystal.KGrid:
            result = num.zeros((n_bands, n_bands), num.Complex)
            for kgii_index, kgii in enumerate(self.KWeights.KGridIndexIncrements):
                if scalar_products[k_index, kgii] is None:
                    continue

                m = scalar_products[k_index, kgii]
                m_diagonal = num.diagonal(m)

                # Omega_D part
                r_tilde = num.divide(m.H, num.conjugate(m_diagonal))

                q = num.zeros((n_bands,), num.Complex)
                for n in range(n_bands):
                    q[n] = arg(m_diagonal[n])

                for n in range(n_bands):
                    q[n] += self.KWeights.KGridIncrements[kgii_index] \
                            * wannier_centers[n]
                t = num.multiply(r_tilde, q)

                result += 4. * self.KWeights.KWeights[kgii_index] * \
                          (-skew_symmetric_part(t.imaginary.T)
                           -1j*symmetric_part(t.real))

            gradient[k_index] = result
        return gradient

    def get_mix_matrix(self, prev_mix_matrix, factor, gradient):
        temp_mix_matrix = {}
        for k_index in self.Crystal.KGrid:
            dW = factor * gradient[k_index]
            if self.DebugMode:
                assert toybox.skewhermiticity_error(dW) < 1e-13

            exp_dW = toybox.matrix_exp_by_diagonalization(dW)
            if self.DebugMode:
                assert toybox.unitariety_error(exp_dW) < 1e-10

            temp_mix_matrix[k_index] = exp_dW * prev_mix_matrix[k_index]

        return temp_mix_matrix

    def test_sp_updater(self, pbands, mix_matrix):
        if not self.DebugMode:
            return

        job = stopwatch.Job("self-test")
        sps_original = self.compute_offset_scalar_products(pbands)
        sps_updated = self.update_offset_scalar_products(sps_original, mix_matrix)
        mixed_bands = compute_mixed_periodic_bands(self.Crystal, pbands, mix_matrix)
        sps_direct = self.compute_offset_scalar_products(mixed_bands)

        for k_index in self.Crystal.KGrid:
            for kgii in self.KWeights.KGridIndexIncrements:
                assert comp.norm_frobenius(sps_direct[k_index, kgii]
                                         - sps_updated[k_index, kgii]) < 1e-13

        sf1 = self.spread_functional(len(pbands), sps_updated)
        sf2 = self.spread_functional(len(pbands), sps_direct)
        assert abs(sf1-sf2) < 1e-10

        job.done()

    def minimize_omega_od_by_codiagonalization(self, raw_scalar_products, mix_matrix):
        """scalar_products are understood to be before application of the
        mix_matrix specified.
        """
        sps = self.update_offset_scalar_products(raw_scalar_products, mix_matrix)

        if self.DebugMode:
            print "od before pre", self.omega_od(sps)

        new_mix_matrix = {}
        omega_od_matrices = []
        for k_index in self.Crystal.KGrid:
            for kgii in self.KWeights.KGridIndexIncrements:
                if sps[k_index, kgii] is not None:
                    omega_od_matrices.append(sps[k_index, kgii].copy())

        job = stopwatch.Job("pre-minimization")
        q, diag_mats, tol = toybox.codiagonalize(omega_od_matrices)
        job.done()

        for k_index in self.Crystal.KGrid:
            new_mix_matrix[k_index] = q.H * mix_matrix[k_index]

        if self.DebugMode:
            sps_post = self.update_offset_scalar_products(raw_scalar_products, new_mix_matrix)
            print "od after pre", self.omega_od(sps_post)
        return new_mix_matrix

    def minimize_spread(self, bands, pbands, mix_matrix):
        if self.DebugMode:
            for ii in self.Crystal.KGrid:
                assert toybox.unitariety_error(mix_matrix[ii]) < 5e-3

        self.test_sp_updater(pbands, mix_matrix)

        job = stopwatch.Job("computing scalar products")
        orig_sps = self.compute_offset_scalar_products(pbands)
        self.check_initial_scalar_products(pbands, orig_sps)
        job.done()

        oi = self.omega_i(len(pbands), orig_sps)

        observer = iteration.make_observer(min_change = 1e-3, max_unchanged = 3)
        observer.reset()
        try:
            while True:
                sps = self.update_offset_scalar_products(orig_sps, mix_matrix)
                if self.DebugMode:
                    assert abs(oi - self.omega_i(len(pbands), sps)) < 1e-5
                od, ood = self.omega_d(len(pbands), sps), \
                          self.omega_od(sps)
                sf = oi+od+ood
                print "spread_func:", sf, oi, od, ood
                observer.add_data_point(sf)

                gradient = self.spread_functional_gradient(len(pbands), sps)
                #gradient = makeRandomKDependentSkewHermitianMatrix(crystal, len(pbands), num.Complex)

                if self.DebugMode:
                    assert abs(self.spread_functional(len(pbands), sps) - sf) < 1e-5

                def testDerivs(x):
                    print_count = 4
                    print "--------------------------"
                    print x
                    print "--------------------------"
                    temp_mix_matrix = self.get_mix_matrix(mix_matrix, x, gradient)
                    temp_sps = self.update_offset_scalar_products(orig_sps, temp_mix_matrix)

                    gpc = self.Crystal.KGrid.grid_point_count()
                    before_oiod = (self.omega_i(len(pbands), sps) \
                                  + self.omega_o_d(sps)) * gpc
                    after_oiod = (self.omega_i(len(pbands), temp_sps) \
                                  + self.omega_od(temp_sps)) * gpc

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
                            added_tup = self.Crystal.KGrid.reduce_periodically(pytools.add_tuples(k_index, kgii))

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

                            before_oiod_here = w_b * (len(pbands)-comp.norm_2_squared(m_diagonal))
                            after_oiod_here = w_b * (len(pbands)-comp.norm_2_squared(new_m_diagonal))

                            before_oiod2 += before_oiod_here
                            after_oiod2 += after_oiod_here
                            doiod_here2  = after_oiod_here-before_oiod_here
                            doiod_here2b = w_b * (+comp.norm_2_squared(m_diagonal)
                                                  -comp.norm_2_squared(new_m_diagonal))
                            assert abs(doiod_here2 - doiod_here2b) < 1e-11

                            doiod_here2c = 2 * w_b * sum(num.multiply(-new_m_diagonal+m_diagonal,
                                                                      num.conjugate(m_diagonal))).real

                            r = num.multiply(m.H, m_diagonal)
                            doiod_here3 = -4*w_b*num.trace((dw*r).real)
                            doiod3 += doiod_here3

                            half_doiod_here3 = -2*w_b*num.trace(dw*r).real

                            doiod_here4 = -2*w_b*(num.diagonal(dm2)*m_diagonal.H).real
                            doiod4 += doiod_here4

                            half_a_doiod_here5 = -2*w_b*sum(num.multiply(num.diagonal(dw_plusb*m_plusb.H),
                                                                         num.conjugate(m_diagonal))).real
                            half_b_doiod_here5 = -2*w_b*sum(num.multiply(num.diagonal(num.conjugate(dw* m.H)),
                                                                         num.diagonal(num.conjugate(m)))).real
                            doiod_here5 = half_a_doiod_here5 + half_b_doiod_here5
                            doiod5 += doiod_here5
                            assert abs(doiod_here4-doiod_here5) < 1e-11

                            ssym_re_r_t = skew_symmetric_part(r.real.T)
                            sym_im_r = symmetric_part(r.imaginary)
                            re_grad_od = 4 * w_b * (-ssym_re_r_t )
                            im_grad_od = 4 * w_b * sym_im_r
                            doiod_here7 = grad_scalar_product(dw.real, re_grad_od) \
                                         + grad_scalar_product(dw.imaginary, im_grad_od)
                            doiod7 += doiod_here7

                            if print_count:
                                #print k_index, kgii
                                #print "dw", comp.norm_frobenius(dw)
                                #print "dm1", comp.norm_frobenius(dm1)
                                #print "dm2", comp.norm_frobenius(dm2)
                                #print "dm2-dm1", \
                                      #comp.norm_frobenius(dm2-dm1) \
                                      #/ comp.norm_frobenius(dm1), \
                                      #" - abs:", comp.norm_frobenius(dm2-dm1)
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
                    temp_mix_matrix = self.get_mix_matrix(mix_matrix, x, gradient)
                    temp_sps = self.update_offset_scalar_products(orig_sps, temp_mix_matrix)

                    result = self.spread_functional(len(pbands), temp_sps)
                    if self.DebugMode:
                        print x, result
                    return result

                def plotfunc(x):
                    temp_mix_matrix = self.get_mix_matrix(mix_matrix, x, gradient)
                    temp_sps = self.update_offset_scalar_products(orig_sps, temp_mix_matrix)

                    new_grad_od = self.spread_functional_gradient_omega_od(len(pbands), temp_sps)
                    new_grad_d = self.spread_functional_gradient_omega_d(len(pbands), temp_sps)
                    sp_od = k_dependent_matrix_gradient_scalar_product(self.Crystal.KGrid, new_grad_od, gradient)
                    sp_d = k_dependent_matrix_gradient_scalar_product(self.Crystal.KGrid, new_grad_d, gradient)
                    sp = sp_od + sp_d

                    oi_here = self.omega_i(len(pbands), temp_sps)
                    od = self.omega_d(len(pbands), temp_sps)
                    ood = self.omega_od(temp_sps)
                    return od, ood, sp_d, sp_od
                           
                step = 0.5/(4*sum(self.KWeights.KWeights))

                if self.InteractivityLevel and (raw_input("see plot? y/n [n]:") == "y"):
                    pytools.write_1d_gnuplot_graphs(plotfunc, -5*step, 5 * step, 
                                               steps = 400, progress = True)
                    raw_input("see plot:")

                xmin = scipy.optimize.brent(minfunc, brack = (0, -step))
                # Marzari's fixed step
                #xmin = step

                mix_matrix = self.get_mix_matrix(mix_matrix, xmin, gradient)
        except iteration.IterationStalled:
            pass
        except iteration.IterationStopped:
            pass
        return mix_matrix

    def minimize_spread_2(self, bands, pbands, mix_matrix):
        if self.DebugMode:
            for ii in self.Crystal.KGrid:
                assert toybox.unitariety_error(mix_matrix[ii]) < 5e-3

        self.test_sp_updater(pbands, mix_matrix)

        job = stopwatch.Job("computing scalar products")
        orig_sps = self.compute_offset_scalar_products(pbands)
        self.check_initial_scalar_products(pbands, orig_sps)
        job.done()

        def f(mix_matrix):
            temp_sps = self.update_offset_scalar_products(orig_sps, mix_matrix)
            result = self.spread_functional(len(pbands), temp_sps)
            return result

        def grad(mix_matrix):
            temp_sps = self.update_offset_scalar_products(orig_sps, mix_matrix)
            return self.spread_functional_gradient(len(pbands), temp_sps)

        def sp(m1, m2):
            return k_dependent_matrix_gradient_scalar_product(self.Crystal.KGrid, m1, m2)

        return minimize_by_cg(DictionaryOfMatrices(mix_matrix), 
                            f, grad, self.get_mix_matrix,
                            step = 0.5/(4*sum(self.KWeights.KWeights)),
                            sp = sp,
                            log_filenames = (",,cg_target_log.data", ",,cg_step_log.data"))




def compute_mixed_bands(crystal, bands, mix_matrix):
    # WARNING! Don't be tempted to insert symmetry code in here, since
    # mix_matrix is of potentially unknown symmetry.

    result = []
    for n in range(len(bands)):
        band = {}

        for k_index in crystal.KGrid:
            # set eigenvalue to 0 since there is no meaning attached to it
            band[k_index] = 0, pytools.linear_combination(mix_matrix[k_index][n],
                                                        [bands[i][k_index][1] 
                                                         for i in range(len(bands))])
        result.append(band)
    return result

def compute_mixed_periodic_bands(crystal, pbands, mix_matrix):
    # WARNING! Don't be tempted to insert symmetry code in here, since
    # mix_matrix is of potentially unknown symmetry.

    result = []
    for n in range(len(pbands)):
        pband = {}

        for k_index in crystal.KGrid.enlarge_at_both_boundaries():
            reduced_k_index = crystal.KGrid.reduce_periodically(k_index)

            # set eigenvalue to 0 since there is no meaning attached to it
            pband[k_index] = 0.j, pytools.linear_combination(mix_matrix[reduced_k_index][n],
                                                           [pbands[i][k_index][1] 
                                                            for i in range(len(pbands))])
        result.append(pband)
    return result

def integrate_over_k_grid(k_grid, f):
    return (1./ k_grid.grid_point_count()) \
           * pytools.general_sum([f(k_index, k_grid[k_index])
                               for k_index in k_grid])

def compute_wanniers(crystal, bands, wannier_grid):
    job = stopwatch.Job("computing wannier functions")
    wannier_functions = []

    for n, band in enumerate(bands):
        this_wf = {}
        for wannier_index in wannier_grid:
            R = wannier_grid[wannier_index]
            def function_in_integral(k_index, k):
                k = crystal.KGrid[k_index]
                return cmath.exp(1.j * (k*R)) * band[k_index][1]

            this_wf[wannier_index] = integrate_over_k_grid(crystal.KGrid, 
                                                           function_in_integral)
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

def generate_random_gaussians(crystal, typecode):
    dlb = crystal.Lattice.DirectLatticeBasis
    while True:
        i = random.randint(0,len(dlb)-1)
        center_coords = num.zeros((len(dlb),), num.Float)
        center_coords[i] = random.uniform(-0.4, 0.4)
        center = dlb[i] * pytools.linear_combination(center_coords, dlb)

        sigma = num.zeros((len(dlb), len(dlb)), num.Float)
        for i in range(len(dlb)):
            max_width = min(1-center_coords[i], center_coords[i])
            sigma[i,i] = random.uniform(0.1, max_width)
        sigma_inv = la.inverse(sigma)
            
        # FIXME this is dependent on dlb actually being unit vectors
        def gaussian(point):
            arg = sigma_inv*(point - center)
            return math.exp(-comp.norm_2_squared(arg))

        yield fempy.mesh_function.discretize_function(crystal.Mesh, 
                                                      gaussian, 
                                                      typecode,
                                                      crystal.NodeNumberAssignment)
    
def guess_initial_mix_matrix(crystal, bands, sp):
    # generate the gaussians
    gaussians = []
    gaussian_it = generate_random_gaussians(crystal, num.Complex)
    for n in range(len(bands)):
        gaussians.append(gaussian_it.next())

    # project the gaussians
    projected_bands = []
    projected_bands_co = []
    for n in range(len(bands)):
        projected_band = {}
        projected_band_co = {}

        for k_index in crystal.KGrid:
            mf = fempy.mesh_function.discretize_function(
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
    mix_matrix = DictionaryOfMatrices()
    for k_index in crystal.KGrid:
        # calculate scalar products
        my_sps = num.zeros((len(bands), len(bands)), num.Complex)
        for n in range(len(bands)):
            for m in range(m+1):
                my_sp = sp(projected_bands[n][k_index], projected_bands[m][k_index])
                my_sps[n,m] = my_sp
                my_sps[m,n] = my_sp.conjugate()

        inv_sqrt_my_sps = 1/comp.cholesky(my_sps)

        mix_matrix[k_index] = num.zeros((len(bands), len(bands)), num.Complex)
        for n in range(len(bands)):
            # determine and compute correct mixture of projected bands
            mix_matrix[k_index][n] = pytools.linear_combination(
                inv_sqrt_my_sps[n], 
                [projected_bands_co[i][k_index] 
                 for i in range(len(bands))])
                
    return mix_matrix

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
        lambda k_index, k: cmath.exp(1j*(k * num.array([5.,17.]))))) < 1e-10
    assert abs(1-integrate_over_k_grid(
        crystal.KGrid, 
        lambda k_index, k: cmath.exp(1j*(k * num.array([0.,0.]))))) < 1e-10

    sp = fempy.mesh_function.ScalarProductCalculator(crystal.NodeNumberAssignment,
                                                     crystal.MassMatrix)
                                                      
    gaps, clusters = pc.analyze_band_structure(crystal.Bands)
    print "Gaps:", gaps
    print "Clusters:", clusters

    bands = crystal.Bands[1:4]
    pbands = crystal.PeriodicBands[1:4]

    job = stopwatch.Job("guessing initial mix")
    mix_matrix = guess_initial_mix_matrix(crystal, bands, sp)
    job.done()

    minimizer_class = MarzariSpreadMinimizer
    minimizer = minimizer_class(crystal, sp, debug_mode, interactivity_level)
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
