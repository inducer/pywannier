import cmath

import pytools
import pytools.grid

# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.computation as comp
import pylinear.randomized as randomized
import pylinear.iteration as iteration

import scipy.optimize




# tools -----------------------------------------------------------------------
class DictionaryOfMatrices(pytools.DictionaryOfArithmeticTypes):
    def _get_empty_self(self):
        return DictionaryOfMatrices()

    def conjugate(self):
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

    def _getreal(self):
        result = self._get_empty_self()
        for key in self:
            result[key] = self[key].real
        return result

    def _getimaginary(self):
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
    matrix = DictionaryOfMatrices
    for k_index in crystal.KGrid:
        matrix[k_index] = randomized.make_random_skewhermitian_matrix(size, tc)
    return matrix

def complex2float(x, bound=1e-10):
    assert abs(x.imag) < bound
    return x.real

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

def integrate_over_k_grid(k_grid, f):
    return (1./ k_grid.grid_point_count()) \
           * pytools.general_sum([f(k_index, k_grid[k_index])
                               for k_index in k_grid])





# minimization algorithms -----------------------------------------------------
def minimize_by_gradient_descent(x, f, grad, x_plus_alpha_grad, step):
    observer = iteration.make_observer(min_change = 1e-3, max_unchanged = 3)
    observer.reset()

    last_fval = f(x)
    try:
        while True:
            grad_here = grad(x)
            def minfunc(alpha):
                return f(x_plus_alpha_grad(x, alpha, grad_here))
            alpha, fval, iterations, funcalls  = scipy.optimize.brent(
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
    return x

def minimize_by_cg(x, f, grad, x_plus_alpha_grad, step, sp, log_filenames = None,
        trace_func=None):
    # from Shewchuk's paper, p. 48
    # Polak-Ribi`ere with implicit restart

    d = last_r = -grad(x)
    observer = iteration.make_observer(min_change = 1e-14, max_unchanged = 300)
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
                return f(x_plus_alpha_grad(x, float(alpha), d))

            if trace_func is not None:
                trace_func(x)

            alpha, fval, iterations, funcalls = scipy.optimize.brent(
                minfunc, brack = (0, step), full_output = True)
            alpha = float(alpha)
            
            observer.add_data_point(fval)
            print "Target value: %f (D:%.9f) - %d calls in last step - step size %f" % (
                fval, fval - last_fval, funcalls, alpha)
            last_fval = fval

            x = x_plus_alpha_grad(x, alpha, d)
            r = -grad(x)
            beta = max(0, complex2float(
                sp(r, r - last_r)/sp(last_r, last_r)))
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





