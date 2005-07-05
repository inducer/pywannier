import pytools 
import pytools.grid
import pytools.stopwatch as stopwatch

import pylinear.array as num

import math, cmath
import fempy.element
import fempy.solver
import photonic_crystal as pc
import cPickle as pickle

grid_vectors = [num.array([1,0]), num.array([0, 1])]
lattice = pc.BravaisLattice(grid_vectors)

rl = lattice.ReciprocalLattice
k_grid  = pytools.grid.make_cell_centered_grid(
    -0.5*(rl[0]+rl[1]), rl,
    [(0, 20)] * 2)

job = stopwatch.Job("geometry")
mesh, boundary = pc.generate_square_mesh_with_rod_center(lattice, 0.18)
job.done()

periodicity_nodes = pc.find_periodicity_nodes(mesh, boundary, 
                                              lattice.DirectLatticeBasis)

unconstrained_nodes = [node for node in mesh.dof_manager() if node not in periodicity_nodes]
number_assignment = fempy.element.assign_node_numbers(unconstrained_nodes)
complete_number_assignment = fempy.element.assign_node_numbers(periodicity_nodes, 
                                                               number_assignment)
crystal = pc.PhotonicCrystal(lattice,
                             mesh,
                             boundary,
                             k_grid,
                             has_inversion_symmetry=True,
                             epsilon=pc.ConstantFunction(1)) 

crystal.Modes = pytools.DependentDictionary(pc.ReducedBrillouinModeListLookerUpper(k_grid))
crystal.MassMatrix = fempy.solver.build_mass_matrix(
    mesh, complete_number_assignment,
    crystal.Epsilon, num.Complex)
crystal.NodeNumberAssignment = complete_number_assignment

n_bands = 10

crystal.Modes = pytools.DependentDictionary(pc.ReducedBrillouinModeListLookerUpper(k_grid))

for k_index in k_grid:
    k = k_grid[k_index]
    if k[0] > 0:
        continue
    print "computing for k =",k

    eigenvalues_here = []
    for i,j in pytools.generate_all_integer_tuples_below(5, 2):
        lambda_ = (2*math.pi*i+k_grid[k_index][0])**2 + \
                  (2*math.pi*j+k_grid[k_index][1])**2 + 0.j
        eigenvalues_here.append((lambda_, (i,j)))

    eigenvalues_here.sort(lambda (ev1, t1), (ev2, t2): cmp(abs(ev1), abs(ev2)))

    modes_here = []
    for band_index, (ev, (i,j)) in enumerate(eigenvalues_here[:n_bands]):
        mode = fempy.mesh_function.discretize_function(
            mesh, 
            lambda x: cmath.exp(1j*((k[0]+2*math.pi*i)*x[0]+
                                    (k[1]+2*math.pi*j)*x[1])),
            num.Complex,
            crystal.NodeNumberAssignment)

        modes_here.append((ev, mode))
    crystal.Modes[k_index] = modes_here

job = stopwatch.Job("saving")
pickle.dump([crystal], file(",,crystal.pickle", "wb"), pickle.HIGHEST_PROTOCOL)
job.done()
