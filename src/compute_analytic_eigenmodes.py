import pylinear.matrices as num
import math, cmath
import fempy.tools as tools
import fempy.stopwatch
import fempy.element
import fempy.solver
import photonic_crystal as pc
import cPickle as pickle

lattice = pc.tBravaisLattice([num.array([1,0]), num.array([0, 1])])

rl = lattice.ReciprocalLattice
k_grid  = tools.makeCellCenteredGrid(-0.5*(rl[0]+rl[1]), rl,
                                     [(0, 8)] * 2)

job = fempy.stopwatch.tJob("geometry")
mesh = pc.generateSquareMeshWithRodCenter(lattice, 0.18)
job.done()

periodicity_nodes = pc.findPeriodicityNodes(mesh, lattice.DirectLatticeBasis)

unconstrained_nodes = [node for node in mesh.dofManager() if node not in periodicity_nodes]
number_assignment = fempy.element.assignNodeNumbers(unconstrained_nodes)
complete_number_assignment = fempy.element.assignNodeNumbers(periodicity_nodes, 
                                                             number_assignment)
crystal = pc.tPhotonicCrystal(lattice,
                              mesh,
                              k_grid,
                              has_inversion_symmetry = True,
                              epsilon = pc.tConstantFunction(1)) 

crystal.Modes = tools.tDependentDictionary(pc.tReducedBrillouinModeListLookerUpper(k_grid))
crystal.MassMatrix = fempy.solver.buildMassMatrix(mesh, complete_number_assignment,
                                                  crystal.Epsilon, num.Complex)
crystal.NodeNumberAssignment = complete_number_assignment

for k_index in k_grid:
    k = k_grid[k_index]
    if k[0] > 0:
        continue
    print "computing for k =",k

    eigenvalues_here = []
    for i,j in tools.generateAllIntegerTuplesBelow(5, 2):
        lambda_ = (2*math.pi*i+k_grid[k_index][0])**2 + \
                  (2*math.pi*j+k_grid[k_index][1])**2 + 0.j
        eigenvalues_here.append((lambda_, (i,j)))

    eigenvalues_here.sort(lambda (ev1, t1), (ev2, t2): cmp(abs(ev1), abs(ev2)))

    modes_here = []
    for ev, (i,j) in eigenvalues_here[:10]:
        modes_here.append((ev, fempy.mesh_function.discretizeFunctionByNodes(
            mesh, 
            lambda x: cmath.exp(1j*((k[0]+2*math.pi*i)*x[0]+
                                    (k[1]+2*math.pi*j)*x[1])),
            num.Complex,
            crystal.NodeNumberAssignment)))
    crystal.Modes[k_index] = modes_here

job = fempy.stopwatch.tJob("saving")
pickle.dump([crystal], file(",,crystal.pickle", "wb"), pickle.HIGHEST_PROTOCOL)
job.done()
