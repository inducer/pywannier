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




class MarzariSpreadMinimizer(SpreadMinimizer):
    @staticmethod
    def grad_scalar_product(a1, a2):
        return num.trace[a1*a2.H]

    @staticmethod
    def k_dependent_matrix_gradient_scalar_product(k_grid, a1, a2):
        return sum(num.trace(a1[k_index]*a2[k_index].H) for k_index in k_grid) \
                / k_grid.grid_point_count()

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

                # Omega_D part
                r_tilde = num.divide(m.H, num.conjugate(m_diagonal))

                q = num.zeros((n_bands,), num.Complex)
                for n in range(n_bands):
                    q[n] = arg(m_diagonal[n])

                for n in range(n_bands):
                    q[n] += self.KWeights.KGridIncrements[kgii_index] \
                            * wannier_centers[n]
                t = num.multiply(r_tilde, q)

                result += 2. * self.KWeights.KWeights[kgii_index] * \
                          (r-r.H + (t+t.H)*-1j)

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

                result += 2. * self.KWeights.KWeights[kgii_index] * (r-r.H)
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

                result += -2.j * self.KWeights.KWeights[kgii_index] * (t+t.H)

            gradient[k_index] = result
        return gradient
