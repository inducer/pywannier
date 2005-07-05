import pytools
import pytools.stopwatch as stopwatch

import fempy.solver
import fempy.mesh
import fempy.eoc as eoc
import fempy.element_norm as element_norm
import fempy.mesh_function as mesh_function
import fempy.visualization
import fempy.geometry
import fempy.visualization as visualization
import photonic_crystal as pc
import pylinear.array as num
import pylinear.operation as op
import math, cmath

CONSIDERED_EVALUES = 5
origin = num.zeros((2,), num.Float)

stopwatch.HIDDEN_JOBS.append("arpack rci")

def needsRefinement( vert_origin, vert_destination, vert_apex, area ):
    return area >= max_tri_area

lattice = pc.BravaisLattice([num.array([1,0]), num.array([0, 1])])
rl = lattice.ReciprocalLattice
k_track = [0.1*rl[0],
           0.4*rl[0],
           0.4*(rl[0]+rl[1]),
           0.1*(rl[0]+rl[1])]
k_track = pytools.interpolate_vector_list(k_track, 3)

eigenvalue_eoc = eoc.EOCRecorder()
eigenfunc_l2_eoc = eoc.EOCRecorder()
eigenfunc_energy_eoc = eoc.EOCRecorder()

do_visualization = raw_input("visualize? [n]") == "y"

max_tri_area = 1e-1
for step in range(7):
    boundary = fempy.mesh.ShapeSection(
        fempy.geometry.get_parallelogram(lattice.DirectLatticeBasis), 
        "floquet")

    mesh = fempy.mesh.TwoDimensionalMesh(
        [fempy.mesh.ShapeSection(
        fempy.geometry.get_parallelogram(lattice.DirectLatticeBasis), 
        "floquet")],
        refinement_func = needsRefinement)
    
    print "ITERATION %d: %d elements, %d nodes" % (step, len(mesh.elements()),
                                                   len(mesh.dof_manager()))
                                                   
    periodicity_nodes = pc.find_periodicity_nodes(mesh, boundary,
                                                lattice.DirectLatticeBasis)

    eigensolver = fempy.solver.LaplacianEigenproblemSolver(
        mesh, constrained_nodes = periodicity_nodes,
        typecode = num.Complex)

    eigenvalue_errors = 0.
    eigenfunc_energy_errors = 0.
    eigenfunc_l2_errors = 0.

    evalue_count = 0
    efunc_count = 0
    for k in k_track:
        eigensolver.setup_constraints(
            pc.get_floquet_constraints(periodicity_nodes, k))

        computed_pairs = eigensolver.solve(
            0, tolerance = 1e-10, number_of_eigenvalues = CONSIDERED_EVALUES + 3)

        analytic_evalues = []
        for i,j in pytools.generate_all_integer_tuples_below(5, 2):
            lambd = (2*math.pi*i+k[0])**2 + \
                    (2*math.pi*j+k[1])**2 + 0.j
            analytic_evalues.append((lambd, (i,j)))

        analytic_evalues.sort(lambda (e1, m1), (e2, m2): cmp(abs(e1), abs(e2)))

        computed_pairs.sort(lambda (e1, m1), (e2, m2): cmp(abs(e1), abs(e2)))

        evalue_index = 0
        while evalue_index < CONSIDERED_EVALUES:
            ana_evalue, (i,j) = analytic_evalues[evalue_index]
            comp_evalue, comp_emode = computed_pairs[evalue_index]

            eigenvalue_errors += abs(ana_evalue-comp_evalue)**2
            evalue_count += 1

            if abs(ana_evalue - analytic_evalues[evalue_index+1][0]) < 1e-10:
                comp_evalue_dist = abs(comp_evalue - computed_pairs[evalue_index+1][0])**2
                eigenvalue_errors += comp_evalue_dist
                evalue_count += 1
                if comp_evalue_dist > 1e-8:
                    print "*** DISCREPANCY!", comp_evalue_dist
                evalue_index += 2
            else:
                evalue_index +=1
                def eigenfunc(x):
                    return cmath.exp(1j*((k[0]+2*math.pi*i)*x[0]+
                                         (k[1]+2*math.pi*j)*x[1]))

                def grad_eigenfunc(x):
                    return 1j* num.array([k[0]+2*math.pi*i,
                                          k[1]+2*math.pi*j], num.Complex) * \
                           cmath.exp(1j*((k[0]+2*math.pi*i)*x[0]+
                                         (k[1]+2*math.pi*j)*x[1]))

                comp_emode *=  eigenfunc(origin) / comp_emode(origin)

                energy_estimator = element_norm.make_energy_error_norm_squared(grad_eigenfunc, comp_emode)
                energy_error = pytools.sum_over(energy_estimator, mesh.elements())
                eigenfunc_energy_errors += energy_error

                l2_estimator = element_norm.make_l2_error_norm_squared(eigenfunc, comp_emode)
                l2_error = pytools.sum_over(l2_estimator, mesh.elements())
                eigenfunc_l2_errors += l2_error

                efunc_count += 1

                if do_visualization:
                    print "computed evalue", comp_evalue
                    visualization.visualize("vtk", (",,result.vtk", ",,result_grid.vtk"), comp_emode.real)
                    raw_input("[showing computed, enter]")

                    print "analytic evalue", ana_evalue
                    ana = mesh_function.discretizeFunction(mesh, eigenfunc, typecode = num.Complex)
                    visualization.visualize("vtk", (",,result.vtk", ",,result_grid.vtk"), ana.real)
                    raw_input("[showing analytic, enter]")

                    print "this error:", total_error
        print "erors:", eigenvalue_errors, eigenfunc_l2_errors, eigenfunc_energy_errors

    eigenfunc_l2_eoc.add_data_point(math.sqrt(len(mesh.elements())),
                                    math.sqrt(abs(eigenfunc_l2_errors))/efunc_count)

    eigenfunc_energy_eoc.add_data_point(math.sqrt(len(mesh.elements())),
                                        math.sqrt(abs(eigenfunc_energy_errors))/efunc_count)

    eigenvalue_eoc.add_data_point(math.sqrt(len(mesh.elements())),
                                  math.sqrt(eigenvalue_errors)/evalue_count)

    max_tri_area *= 0.5

print "-------------------------------------------------------"
print "Eigenvalue EOC overall:", eigenvalue_eoc.estimateOrderOfConvergence()[0,1]
print "EOC Gliding means:"
gliding_means = eigenvalue_eoc.estimateOrderOfConvergence(3)
gliding_means_iterations,dummy = gliding_means.shape
for i in range(gliding_means_iterations):
    print "Iteration %d: %f" % (i, gliding_means[i,1])
print "-------------------------------------------------------"
eigenvalue_eoc.writeGnuplotFile(",,eigenvalue_conv.data")
print "-------------------------------------------------------"
print "Eigenfunction L2 EOC overall:", eigenfunc_l2_eoc.estimateOrderOfConvergence()[0,1]
print "EOC Gliding means:"
gliding_means = eigenfunc_l2_eoc.estimateOrderOfConvergence(3)
gliding_means_iterations,dummy = gliding_means.shape
for i in range(gliding_means_iterations):
    print "Iteration %d: %f" % (i, gliding_means[i,1])
print "-------------------------------------------------------"
eigenfunc_l2_eoc.writeGnuplotFile(",,eigenfunc_l2_conv.data")
print "-------------------------------------------------------"
print "Eigenfunction energy-norm EOC overall:", eigenfunc_energy_eoc.estimateOrderOfConvergence()[0,1]
print "EOC Gliding means:"
gliding_means = eigenfunc_energy_eoc.estimateOrderOfConvergence(3)
gliding_means_iterations,dummy = gliding_means.shape
for i in range(gliding_means_iterations):
    print "Iteration %d: %f" % (i, gliding_means[i,1])
print "-------------------------------------------------------"
eigenfunc_energy_eoc.writeGnuplotFile(",,eigenfunc_energy_conv.data")


