import math
import cmath
import cPickle

# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.geometry
import fempy.stopwatch
import fempy.solver
import fempy.tools as tools

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc
 



def computeEigenmodes(crystal, mesh, epsilon, sigma, keys, k_grid, lattice):
    periodicity_nodes = pc.findPeriodicityNodes(mesh, lattice.DirectLatticeBasis)

    eigensolver = fempy.solver.tLaplacianEigenproblemSolver(mesh, 
                                                            g = epsilon, 
                                                            typecode = num.Complex)

    for key in keys:
        k = k_grid[key]
        
        if crystal.HasInversionSymmetry and k[0] < 0:
            # use inversion symmetry, only compute one half.
            continue

        print "computing for k =", k
        my_periodicity_nodes = pc.computeFloquetBCs(periodicity_nodes, k)
        eigensolver.addPeriodicBoundaryConditions(my_periodicity_nodes)

        modes = eigensolver.solve(sigma)
        crystal.Modes[key] = modes



      
def computeEigenmodesForStandardUnitCell(lattice, inner_radius):
    def needsRefinement( vert_origin, vert_destination, vert_apex, area ):
        return area >= 1e-2
        bary_x = ( vert_origin.x() + vert_destination.x() + vert_apex.x() ) / 3
        bary_y = ( vert_origin.y() + vert_destination.y() + vert_apex.y() ) / 3
    
        dist_center = math.sqrt( bary_x**2 + bary_y**2 )
        if dist_center < 0.4:
            return area >= 1e-2
        else:
            return False

    job = fempy.stopwatch.tJob("geometry")
    mesh = fempy.mesh.tTwoDimensionalMesh(
        fempy.geometry.getUnitCellGeometry(lattice.DirectLatticeBasis, 
                                           inner_radius = inner_radius,
                                           use_exact = False,
                                           constraint_id = "floquet"),
        refinement_func = needsRefinement)
    job.done()

    def epsilon(x):
        if tools.norm2(x) < inner_radius:
            return 11.56
        else:
            return 1

    sigma = 0.
  
    rl = lattice.ReciprocalLattice

    # make sure the grid has an uneven number of subdivisions,
    # otherwise k=0 ends up on the grid. k=0 means zero frequency,
    # so no eigenvalues ... so pure mess.
    k_grid  = tools.makeSubdivisionGrid(-0.5*(rl[0]+rl[1]), lattice.ReciprocalLattice,
                                        [(0, 16)] * 2)
    keys = k_grid.asSequence().getAllIndices()

    mode_dict = tools.tDependentDictionary(
        pc.tInvertedModeListLookerUpper(k_grid.gridIntervalCounts()))

    crystal = pc.tPhotonicCrystal(lattice, mesh, k_grid, True, mode_dict)
                             
    computeEigenmodes(crystal, mesh, epsilon, sigma, keys = keys, k_grid = k_grid, 
                      lattice = lattice)

    return crystal



def main():
    fempy.stopwatch.HIDDEN_JOBS.append("arpack rci")
    fempy.stopwatch.HIDDEN_JOBS.append("bcs, periodic")
    fempy.stopwatch.HIDDEN_JOBS.append("shift matrix")

    a = 1.
    my_lattice = pc.tBravaisLattice([a*num.array([1,0], num.Float), a*num.array([0,1], num.Float)])
    crystal = computeEigenmodesForStandardUnitCell(my_lattice, a*0.18)
    job = fempy.stopwatch.tJob("saving")
    cPickle.dump(crystal, file(",,crystal.pickle", "wb"), cPickle.HIGHEST_PROTOCOL)
    job.done()

main()
