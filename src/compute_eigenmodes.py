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
import eigenmodes
import lattice
 



def findPeriodicityNodes(mesh, grid_vectors):
  bnodes = filter(lambda node: node.ConstraintId == "floquet",
                  mesh.dofManager().constrainedNodes())

  job = fempy.stopwatch.tJob("periodicity")

  periodicity_nodes = []
  for node in bnodes:
    matches_made = 0
    for gv in grid_vectors:
      ideal_point = node.Coordinates + gv
      dist_threshold = tools.norm2(gv) * 0.3

      close_nodes_with_dists = filter(
        lambda (n, dist): dist <= dist_threshold, 
        tools.decorate(
        lambda other_node: tools.norm2(ideal_point - other_node.Coordinates), bnodes))
      close_nodes_with_dists.sort(lambda (n1,d1), (n2,d2): cmp(d1,d2))

      if not close_nodes_with_dists:
        continue

      if close_nodes_with_dists[0][1] < 1e-5 * dist_threshold:
        # Found one node that is close enough, bet our
        # butt on it.
        other_node, dist = close_nodes_with_dists[0]
        periodicity_nodes.append((gv, node, [(other_node, 1)]))
      else:
        if len(close_nodes_with_dists) == 1:
          print "WARNING: Found only one node that's near enough."
          continue

        other_node_a, dist_a = close_nodes_with_dists[0]
        other_node_b = None

        for candidate_b, dist in close_nodes_with_dists[1:]:
          if tools.angleCosineBetweenVectors(other_node_a.Coordinates - ideal_point,
                                             candidate_b.Coordinates - ideal_point) < -0.1:
            other_node_b = candidate_b
            dist_b = dist

        if other_node_b is None:
          continue

        for candidate_b, dist in close_nodes_with_dists[1:]:
          if tools.angleCosineBetweenVectors(other_node_a.Coordinates - ideal_point,
                                             candidate_b.Coordinates - ideal_point) < -0.1:
            other_node_b = candidate_b
            dist_b = dist
            break
        total_dist = dist_a + dist_b
        periodicity_nodes.append(
          (gv, node,
           [(other_node_a, dist_a/total_dist),
            (other_node_b, dist_b/total_dist)]))
  job.done()

  return periodicity_nodes




def computeFloquetBCs(periodicity_nodes, k):
  result = []
  for gv, node, other_nodes in periodicity_nodes:
    floquet_factor = cmath.exp(1j * num.innerproduct(gv, k))
    my_condition = [(node,1)]
    for other_node, factor in other_nodes:
      my_condition.append((other_node, -factor*floquet_factor))
    result.append(my_condition)
  return result




def unitCellDemo(mesh, epsilon, sigma, list_of_ks, lattice):
  periodicity_nodes = findPeriodicityNodes(mesh, lattice.DirectLatticeBasis)

  eigensolver = fempy.solver.tLaplacianEigenproblemSolver(mesh, 
                                                          g = epsilon, 
                                                          typecode = num.Complex)

  results = eigenmodes.tEigenmodes(mesh)
  for k in list_of_ks:
    print "computing for k =", k
    my_periodicity_nodes = computeFloquetBCs(periodicity_nodes, k)
    eigensolver.addPeriodicBoundaryConditions(my_periodicity_nodes)

    solution = eigensolver.solve(sigma)
    results.addKValue(k, list(solution.RitzValues), list(solution.RitzVectors))

  return results

  #band_diagram_file = file(",,band_diagram.data", "w")
  #for index, (k, result) in tools.indexAnd(results):
    #for val in result.RitzValues:
      #band_diagram_file.write("%d\t%f\n" % (index, math.sqrt(val.real)))




      
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

  sigma = 0.9
  
  rl = lattice.ReciprocalLattice
  #raw_ks = [0 * rl[0], 0.5 * rl[0], 0.5 * (rl[0]+rl[1]), 0 * rl[0]]
  #all_ks = tools.interpolateVectorList(raw_ks, 20)

  list_of_ks = tools.getGrid(-0.5*(rl[0]+rl[1]), lattice.ReciprocalLattice,
                             [4] * len (lattice.ReciprocalLattice))
  return unitCellDemo(mesh, epsilon, sigma, list_of_ks, lattice)



def main():
  a = 1.
  my_lattice = lattice.tLattice([a*num.array([1,0], num.Float), a*num.array([0,1], num.Float)])
  results = computeEigenmodesForStandardUnitCell(my_lattice, a*0.18)
  job = fempy.stopwatch.tJob("saving")
  cPickle.dump((my_lattice, results), file(",,eigenmodes.pickle", "w"), cPickle.HIGHEST_PROTOCOL)
  job.done()

main()
