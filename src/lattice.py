import math

import fempy.tools as tools

# Numeric imports -------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la





class tLattice:
  def __init__(self, direct_lattice_basis):
    self.DirectLatticeBasis = direct_lattice_basis
                         
    # compute reciprocal lattice
    d = len(self.DirectLatticeBasis[0])
    mat = num.zeros((d*d, d*d), num.Float)
    rhs = num.zeros((d*d,), num.Float)


    equation_no = 0
    for direct_vector_number, direct_vec in tools.indexAnd(self.DirectLatticeBasis):
      for indirect_vector_number in range(d):
        for indirect_vector_coordinate in range(d):
          mat[equation_no, indirect_vector_number * d + indirect_vector_coordinate] = \
                           direct_vec[indirect_vector_coordinate]
          rhs[equation_no] = math.pi * 2 * tools.delta(direct_vector_number,
                                                       indirect_vector_number)
        equation_no += 1

    sol = la.solve_linear_equations(mat, rhs)
    self.ReciprocalLattice = []
    for indirect_vector_number in range(d):
      rec_vec = num.zeros((d,), num.Float)
      for indirect_vector_coordinate in range(d):
        rec_vec[indirect_vector_coordinate] = sol[indirect_vector_number * d + indirect_vector_coordinate]
      self.ReciprocalLattice.append(rec_vec)

