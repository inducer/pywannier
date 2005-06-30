import math, cmath, sets, sys
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.linear_algebra as la
import pylinear.operation as op

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.geometry
import fempy.stopwatch
import fempy.visualization
import fempy.solver
import fempy.tools as tools

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc
 



def compute_eigenmodes(crystal, sigma):
    periodicity_nodes = pc.find_periodicity_nodes(crystal.Mesh, 
                                                  crystal.BoundaryShapeSection,
                                                  crystal.Lattice.DirectLatticeBasis)

    eigensolver = fempy.solver.tLaplacianEigenproblemSolver(crystal.Mesh, 
                                                            constrained_nodes = periodicity_nodes,
                                                            g = crystal.Epsilon, 
                                                            typecode = num.Complex)

    crystal.MassMatrix = eigensolver.massMatrix()
    crystal.NodeNumberAssignment = eigensolver.nodeNumberAssignment()

    for k_index in crystal.KGrid:
        k = crystal.KGrid[k_index]

        if crystal.HasInversionSymmetry and k[0] > 0:
            # use inversion symmetry, only compute one half.
            continue

        print "computing for k =", k
        eigensolver.setupConstraints(pc.get_floquet_constraints(periodicity_nodes, k))
        crystal.Modes[k_index] = eigensolver.solve(sigma,
                                                   tolerance = 1e-10,
                                                   number_of_eigenvalues = 10,
                                                   allow_direct=min(num.absolute(k)) > 1e-5)
        for ev, em in crystal.Modes[k_index]:
            assert em.numberAssignment() is crystal.NodeNumberAssignment

def compute_eigenmodes_for_standard_unit_cell(lattice, epsilon, inner_radius,
                                              refine_steps = 1,
                                              k_grid_points = 16,
                                              coarsening_factor = 1):

    job = fempy.stopwatch.Job("geometry")
    mesh, boundary = pc.generate_square_mesh_with_rod_center(
        lattice, 
        inner_radius = inner_radius,
        coarsening_factor = coarsening_factor)
    job.done()

    fempy.visualization.writeGnuplotMesh(mesh, ",,mesh.data")

    sigma = 0.
  
    rl = lattice.ReciprocalLattice

    has_inversion_symmetry = True

    # make sure the grid has an even number of grid points,
    # otherwise k=0 ends up on the grid. k=0 means zero frequency,
    # so no eigenvalues. Arpack will still compute some, but they
    # are garbage.

    # This ends up being a genuine Monkhorst-Pack mesh.
    k_grid  = tools.make_cell_centered_grid(-0.5*(rl[0]+rl[1]), lattice.ReciprocalLattice,
                                            [(0, k_grid_points)] * 2)

    
    crystals = []

    max_area = 2e-4 * coarsening_factor
    while len(crystals) <= refine_steps:
        print "have %d elements" % len(mesh.elements())
        fempy.visualization.writeGnuplotMesh(mesh, ",,mesh.data")
        
        if has_inversion_symmetry:
            mode_dict = tools.DependentDictionary(
                pc.ReducedBrillouinModeListLookerUpper(k_grid))
        else:
            mode_dict = pc.make_k_periodic_lookup_structure(k_grid)

        crystal = pc.PhotonicCrystal(lattice, 
                                     mesh, boundary,
                                     k_grid, 
                                     has_inversion_symmetry=has_inversion_symmetry, 
                                     epsilon=epsilon)

        crystal.Modes = mode_dict
        
        compute_eigenmodes(crystal, sigma)
        crystals.append(crystal)

        if len(crystals) <= refine_steps:
            job = fempy.stopwatch.Job("refining")
            mesh_change = mesh.getRefinement(lambda el: el.area() > max_area)
            job.done()
            mesh = mesh_change.meshAfter()

        max_area *= 0.8

    return crystals



def run():
    fempy.stopwatch.HIDDEN_JOBS.append("arpack rci")
    fempy.stopwatch.HIDDEN_JOBS.append("constrained matrices")

    a = 1.
    inner_radius = 0.18 

    my_lattice = pc.BravaisLattice([num.array([a,0], num.Float), num.array([0,a], num.Float)])

    #epsilon = pc.CircularFunctionRemapper(pc.StepFunction(1, a*inner_radius, 1.))
    epsilon = pc.CircularFunctionRemapper(pc.StepFunction(11.56, a*inner_radius, 1.))
    
    crystals = compute_eigenmodes_for_standard_unit_cell(my_lattice, 
                                                         epsilon,
                                                         a*inner_radius,
                                                         refine_steps=0, # 4
                                                         coarsening_factor=1,
                                                         k_grid_points=7)

    job = fempy.stopwatch.Job("saving")
    pickle.dump(crystals, file(",,crystal.pickle", "wb"), pickle.HIGHEST_PROTOCOL)
    job.done()

if __name__ == "__main__":
    run()
