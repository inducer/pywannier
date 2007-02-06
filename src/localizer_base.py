# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.linear_algebra as la
import pylinear.computation as comp
import pylinear.toybox as toybox
import pylinear.iteration as iteration

from pytools import stopwatch

# Local imports ---------------------------------------------------------------
from localizer_tools import *
from localizer_preproc import compute_mixed_bands, compute_mixed_periodic_bands
from localizer_postproc import visualize_wannier




class SpreadMinimizer:
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
                        #arg = cmath.log(m[i,i]).imag
                        magfiles[i].write("%f\t%f\t%f\n" %(where[0], where[1], abs(m[i,i])))
                        argfiles[i].write("%f\t%f\n" %(where[0], where[1]))
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

        observer = iteration.make_observer(min_change = 1e-7, max_unchanged = 17)
        observer.reset()

        stepcount = 0

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
                                  + self.omega_od(sps)) * gpc
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
                            doiod_here2b = w_b * ( comp.norm_2_squared(m_diagonal)
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
                            doiod_here7 = self.grad_scalar_product(dw.real, re_grad_od) \
                                         + self.grad_scalar_product(dw.imaginary, im_grad_od)
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
                    sp_od = complex2float(
                        self.k_dependent_matrix_gradient_scalar_product(self.Crystal.KGrid, new_grad_od, gradient))
                    sp_d = complex2float(
                        self.k_dependent_matrix_gradient_scalar_product(self.Crystal.KGrid, new_grad_d, gradient))
                    #sp = sp_od + sp_d

                    #oi_here = self.omega_i(len(pbands), temp_sps)
                    od = self.omega_d(len(pbands), temp_sps)
                    ood = self.omega_od(temp_sps)
                    return od, ood, sp_d, sp_od
                           
                # negative because we're pointing downwards!
                step = 0.5/(4*sum(self.KWeights.KWeights))

                if self.InteractivityLevel and (raw_input("see plot? y/n [n]:") == "y"):
                    pytools.write_1d_gnuplot_graphs(plotfunc, -5*step, 5 * step, 
                                               steps = 400, progress = True)
                    raw_input("see plot:")

                #xmin = scipy.optimize.brent(minfunc, brack = (0, -step))
                # Marzari's fixed step
                xmin = -step

                mix_matrix = self.get_mix_matrix(mix_matrix, xmin, gradient)

                #visualize_wannier(self.Crystal, bands, mix_matrix,
                        #basename="wannier_%05d" % stepcount)
                stepcount += 1
        except iteration.IterationStalled:
            pass
        except iteration.IterationStopped:
            pass
        return mix_matrix

    def minimize_spread_cg(self, bands, pbands, mix_matrix):
        if self.DebugMode:
            for ii in self.Crystal.KGrid:
                assert toybox.unitariety_error(mix_matrix[ii]) < 5e-3

        self.test_sp_updater(pbands, mix_matrix)

        job = stopwatch.Job("computing scalar products")
        orig_sps = self.compute_offset_scalar_products(pbands)
        self.check_initial_scalar_products(pbands, orig_sps)
        job.done()

        stepcount = [0]

        def trace(mix_matrix):
            visualize_wannier(self.Crystal, bands, mix_matrix,
                    basename="wannier_%05d" % stepcount[0])
            stepcount[0] += 1

        def f(mix_matrix):
            temp_sps = self.update_offset_scalar_products(orig_sps, mix_matrix)
            result = self.spread_functional(len(pbands), temp_sps)
            return result

        def grad(mix_matrix):
            temp_sps = self.update_offset_scalar_products(orig_sps, mix_matrix)
            return self.spread_functional_gradient(len(pbands), temp_sps)

        def sp(m1, m2):
            return self.k_dependent_matrix_gradient_scalar_product(self.Crystal.KGrid, m1, m2)

        return minimize_by_cg(DictionaryOfMatrices(mix_matrix), 
                            f, grad, self.get_mix_matrix,
                            step = 0.5/(4*sum(self.KWeights.KWeights)),
                            sp=sp,
                            log_filenames = (",,cg_target_log.data", ",,cg_step_log.data"),
                            trace_func=trace
                            )

