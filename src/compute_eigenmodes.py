import math
import cmath
import cPickle
import sets

# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.geometry
import fempy.stopwatch
import fempy.solver
import fempy.tools as tools

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc
 



def computeEigenmodes(crystal, epsilon, sigma, k_grid, lattice):
    periodicity_nodes = pc.findPeriodicityNodes(crystal.Mesh, lattice.DirectLatticeBasis)

    constrained_nodes = sets.Set([node for gv, node, rest in periodicity_nodes])
    eigensolver = fempy.solver.tLaplacianEigenproblemSolver(crystal.Mesh, 
                                                            constrained_nodes = constrained_nodes,
                                                            g = epsilon, 
                                                            typecode = num.Complex)

    crystal.ScalarProduct = eigensolver.massMatrix()

    for key in k_grid:
        k = k_grid[key]

        if crystal.HasInversionSymmetry and k[0] < 0:
            # use inversion symmetry, only compute one half.
            continue

        print "computing for k =", k
        crystal.Modes[key] = eigensolver.solve(
            sigma,
            pc.getFloquetConstraints(periodicity_nodes, k))



      
def computeEigenmodesForStandardUnitCell(lattice, epsilon, inner_radius,
                                         refine_steps = 1,
                                         k_grid_points = 16):
    def needsRefinement( vert_origin, vert_destination, vert_apex, area ):
        bary_x = ( vert_origin.x() + vert_destination.x() + vert_apex.x() ) / 3
        bary_y = ( vert_origin.y() + vert_destination.y() + vert_apex.y() ) / 3
    
        dist_center = math.sqrt( bary_x**2 + bary_y**2 )
        if dist_center < inner_radius * 1.2:
            return area >= 2e-3
        else:
            return area >= 1e-2

    job = fempy.stopwatch.tJob("geometry")
    mesh = fempy.mesh.tTwoDimensionalMesh(
        fempy.geometry.getUnitCellGeometry(lattice.DirectLatticeBasis, 
                                           inner_radius = inner_radius,
                                           use_exact = True,
                                           constraint_id = "floquet"),
        refinement_func = needsRefinement)
    job.done()

    sigma = 0.
  
    rl = lattice.ReciprocalLattice

    # make sure the grid has an even number of grid points,
    # otherwise k=0 ends up on the grid. k=0 means zero frequency,
    # so no eigenvalues. Arpack will still compute some, but they
    # are garbage.
    k_grid  = tools.makeSubdivisionGrid(-0.5*(rl[0]+rl[1]), lattice.ReciprocalLattice,
                                        [(0, k_grid_points)] * 2)

    crystals = []
    while len(crystals) <= refine_steps:
        print "have %d elements" % len(mesh.elements())
        mode_dict = tools.tDependentDictionary(
            pc.tInvertedModeListLookerUpper(k_grid.gridIntervalCounts()))
        
        crystal = pc.tPhotonicCrystal(lattice, 
                                      mesh, 
                                      k_grid, 
                                      has_inversion_symmetry = True, 
                                      epsilon = epsilon,
                                      modes_start = mode_dict,
                                      )
        
        computeEigenmodes(crystal, epsilon, sigma, k_grid = k_grid, 
                          lattice = lattice)
        crystals.append(crystal)

        if len(crystals) <= refine_steps:
            job = fempy.stopwatch.tJob("refining")
            mesh_change = mesh.getRefinement(lambda x: True)
            job.done()
            mesh = mesh_change.meshAfter()

    return crystals



def main():
    fempy.stopwatch.HIDDEN_JOBS.append("arpack rci")
    fempy.stopwatch.HIDDEN_JOBS.append("linear combination constraints")
    fempy.stopwatch.HIDDEN_JOBS.append("shift matrix")

    a = 1.
    inner_radius = 0.18 

    my_lattice = pc.tBravaisLattice([num.array([a,0], num.Float), num.array([0,a], num.Float)])

    epsilon = pc.tCircularFunctionRemapper(pc.tStepFunction(11.56, a*inner_radius, 1.))
    
    crystals = computeEigenmodesForStandardUnitCell(my_lattice, 
                                                    epsilon,
                                                    a*inner_radius,
                                                    refine_steps = 0, # 4
                                                    k_grid_points = 8)
    job = fempy.stopwatch.tJob("saving")
    cPickle.dump(crystals, file(",,crystal.pickle", "wb"), cPickle.HIGHEST_PROTOCOL)
    job.done()

main()
