import pytools 
import fempy.solver
import fempy.mesh
import fempy.visualization
import photonic_crystal as pc
import pylinear.array as num
import pylinear.computation as comp
import pylinear.toybox as toybox
import math, cmath

lowest_functions = []

class Problem1D:
    def __init__(self):
        self.Lattice = pc.BravaisLattice([num.array([1])])
        self.Mesh = fempy.mesh.tOneDimensionalMesh(0, 1, 40, "floquet_l", "floquet_r")
        self.LeftNode = fempy.solver.getNodesWithTrackingId(self.Mesh, "floquet_l")[0]
        self.RightNode = fempy.solver.getNodesWithTrackingId(self.Mesh, "floquet_r")[0]

        def eps(x):
            if 0.4 <= x[0] < 0.6:
                return 5
            else:
                return 1

        self.EigenSolver = fempy.solver.LaplacianEigenproblemSolver(
            self.Mesh, constrained_nodes = [self.RightNode], g = eps,
            typecode = num.Complex)
        nodes = [node for node in self.Mesh.dofManager()]
        nodes.sort(lambda n1, n2: cmp(n1.Coordinates[0], n2.Coordinates[0]))
        self.CenterNode = nodes[len(nodes)/2]

    def k_track(self):
        rl = self.Lattice.ReciprocalLattice
        return [0.001*rl[0], 0.999*rl[0]]

    def get_condition_number(self, k):
        mm = num.matrixmultiply
        a = self.EigenSolver.setup_constraints(
            {self.RightNode: (0, [(cmath.exp(1j*k[0]), self.LeftNode)])})
        s = self.EigenSolver.stiffnessMatrix()
        m = self.EigenSolver.massMatrix()
        a = num.asarray(a, a.typecode(), num.DenseMatrix)
        s = num.asarray(s, s.typecode(), num.DenseMatrix)
        m = num.asarray(m, s.typecode(), num.DenseMatrix)
        total_s = mm(a, mm(s, num.hermite(a)))
        total_m = mm(a, mm(m, num.hermite(a)))
        return comp.spectral_condition_number(total_s), op.spectral_condition_number(total_s, 1)

    def get_eigenvalues(self, k):
        rl = self.Lattice.ReciprocalLattice

        print "computing for k = ",k
        self.EigenSolver.setup_constraints(
            {self.RightNode: (0, [(cmath.exp(1j*k[0]), self.LeftNode)])})
        pairs = self.EigenSolver.solve(0,
                                       tolerance = 1e-6,
                                       number_of_eigenvalues = 7)
        
        pairs.sort(lambda (e1, m1), (e2, m2): cmp(abs(e1), abs(e2)))
        if abs(k[0] - 0.5*math.pi) < 0.3:
            for i, (ev, em) in enumerate(pairs):
                em /= em[self.LeftNode]
                fempy.visualization.visualize1DMeshFunction(em.real, ",,f%d.data" % i)

        lowest_func = pairs[0][1]
        uniphase = lowest_func[self.CenterNode]
        lowest_functions.append(lowest_func/uniphase)
        return [evalue for evalue, em in pairs]
    
class Problem2D:
    def __init__(self):
        self.Lattice = pc.BravaisLattice([num.array([1,0]), num.array([0, 1])])
        epsilon = pc.CircularFunctionRemapper(pc.StepFunction(11.56, 0.18, 1.)) # 11.56
        self.Mesh, boundary = pc.generate_square_mesh_with_rod_center(
            self.Lattice, inner_radius = 0.18, coarsening_factor = 4)

        self.PeriodicityNodes = pc.find_periodicity_nodes(self.Mesh, 
                                                          boundary,
                                                          self.Lattice.DirectLatticeBasis)
        self.EigenSolver = fempy.solver.LaplacianEigenproblemSolver(
            self.Mesh, constrained_nodes = self.PeriodicityNodes,
            g = epsilon, typecode = num.Complex)

    def k_track(self):
        rl = self.Lattice.ReciprocalLattice
        return [0.01*rl[0],
                0.49*rl[0],
                0.49*(rl[0]+rl[1]),
                0.01*(rl[0]+rl[1])]

    def get_eigenvalues(self, k):
        self.EigenSolver.setup_constraints(
            pc.get_floquet_constraints(self.PeriodicityNodes, k))

        print "computing for k = ",k

        pairs = self.EigenSolver.solve(
            0, tolerance = 1e-10, number_of_eigenvalues = 10)
        pairs.sort(lambda (e1, m1), (e2, m2): cmp(abs(e1), abs(e2)))
        return [evalue for evalue, em in pairs]

problem = Problem2D()
k_track = toybox.interpolate_vector_list(problem.k_track(), 49)

def scale_eigenvalue(ev):
    return math.sqrt(abs(ev)) / (2 * math.pi)

if raw_input("write condition file? [n]") == "y":
    condfile = file(",,condition_k.data", "w")
    effcondfile = file(",,eff_condition_k.data", "w")
    for j,k in enumerate(k_track):
        cond, effcond = problem.get_condition_number(k)
        condfile.write("%f\t%f\n" % (k_track[j][0], cond))
        effcondfile.write("%f\t%f\n" % (k_track[j][0], effcond))

bdfile = file(",,band_diagram.data", "w")
evlists = [problem.get_eigenvalues(k) for k in k_track]
for i in range(len(evlists[0])):
    for j in range(len(k_track)):
        bdfile.write("%f\t%f\n" % (j, scale_eigenvalue(evlists[j][i])))
    bdfile.write("\n")

if raw_input("plot u development (1D)? [n]") == "y":
    major_grid = [k[0] for k in k_track]
    nodes = [node for node in problem.Mesh.dofManager()]
    nodes.sort(lambda n1, n2: cmp(n1.Coordinates[0], n2.Coordinates[0]))
    minor_grid = [node.Coordinates[0] for node in nodes]
    data = [[(cmath.exp(-1j*k[0]*node.Coordinates[0])*mf[node]).real 
             for node in nodes] 
            for k, mf in zip(k_track, lowest_functions)]
    fempy.visualization.visualize2DGridData("vtk",
                                            major_grid, minor_grid, data, ",,u_development",
                                            scale_minor = 10)
