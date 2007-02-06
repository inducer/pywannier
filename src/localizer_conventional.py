import cmath

import pytools
import pytools.grid
import pytools.stopwatch as stopwatch

# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.computation as comp
import pylinear.toybox as toybox
import pylinear.randomized as randomized
import pylinear.iteration as iteration

import scipy.optimize

# local imports ---------------------------------------------------------------
from localizer_tools import *
from localizer_base import SpreadMinimizer
from localizer_preproc import compute_mixed_bands, compute_mixed_periodic_bands




class ConventionalSpreadMinimizer(SpreadMinimizer):
    @staticmethod
    def grad_scalar_product(a1, a2):
        rp = num.multiply(a1.real, a2.real)
        ip = num.multiply(a1.imaginary, a2.imaginary)
        return num.sum(rp) + num.sum(ip)

    @staticmethod
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
