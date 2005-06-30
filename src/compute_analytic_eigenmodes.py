import pylinear.array as num
import math, cmath
import fempy.tools as tools
import fempy.stopwatch
import fempy.element
import fempy.solver
import photonic_crystal as pc
import cPickle as pickle

grid_vectors = [num.array([1,0]), num.array([0, 1])]
lattice = pc.BravaisLattice(grid_vectors)

rl = lattice.ReciprocalLattice
k_grid  = tools.make_cell_centered_grid(-0.5*(rl[0]+rl[1]), rl,
                                        [(0, 20)] * 2)

job = fempy.stopwatch.Job("geometry")
mesh, boundary = pc.generate_square_mesh_with_rod_center(lattice, 0.18)
job.done()

periodicity_nodes = pc.find_periodicity_nodes(mesh, boundary, 
                                              lattice.DirectLatticeBasis)

unconstrained_nodes = [node for node in mesh.dofManager() if node not in periodicity_nodes]
number_assignment = fempy.element.assignNodeNumbers(unconstrained_nodes)
complete_number_assignment = fempy.element.assignNodeNumbers(periodicity_nodes, 
                                                             number_assignment)
crystal = pc.PhotonicCrystal(lattice,
                             mesh,
                             boundary,
                             k_grid,
                             has_inversion_symmetry=True,
                             epsilon=pc.ConstantFunction(1)) 

crystal.Modes = tools.DependentDictionary(pc.ReducedBrillouinModeListLookerUpper(k_grid))
crystal.MassMatrix = fempy.solver.buildMassMatrix(mesh, complete_number_assignment,
                                                  crystal.Epsilon, num.Complex)
crystal.NodeNumberAssignment = complete_number_assignment

n_bands = 10

crystal.Modes = tools.DependentDictionary(pc.ReducedBrillouinModeListLookerUpper(k_grid))

for k_index in k_grid:
    k = k_grid[k_index]
    if k[0] > 0:
        continue
    print "computing for k =",k

    eigenvalues_here = []
    for i,j in tools.generate_all_integer_tuples_below(5, 2):
        lambda_ = (2*math.pi*i+k_grid[k_index][0])**2 + \
                  (2*math.pi*j+k_grid[k_index][1])**2 + 0.j
        eigenvalues_here.append((lambda_, (i,j)))

    eigenvalues_here.sort(lambda (ev1, t1), (ev2, t2): cmp(abs(ev1), abs(ev2)))

    modes_here = []
    for band_index, (ev, (i,j)) in enumerate(eigenvalues_here[:n_bands]):
        mode = fempy.mesh_function.discretizeFunction(
            mesh, 
            lambda x: cmath.exp(1j*((k[0]+2*math.pi*i)*x[0]+
                                    (k[1]+2*math.pi*j)*x[1])),
            num.Complex,
            crystal.NodeNumberAssignment)

        modes_here.append((ev, mode))
    crystal.Modes[k_index] = modes_here

job = fempy.stopwatch.Job("saving")
pickle.dump([crystal], file(",,crystal.pickle", "wb"), pickle.HIGHEST_PROTOCOL)
job.done()
