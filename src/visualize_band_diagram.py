import fempy.tools as tools
import fempy.solver
import fempy.mesh
import photonic_crystal as pc
import pylinear.matrices as num
import math, cmath

class t1DProblem:
    def __init__(self):
        self.Lattice = pc.tBravaisLattice([num.array([1])])
        self.Mesh = fempy.mesh.tOneDimensionalMesh(-1, 1, 100, "floquet_l", "floquet_r")
        self.LeftNode = fempy.solver.getNodesWithTrackingId(self.Mesh, "floquet_l")[0]
        self.RightNode = fempy.solver.getNodesWithTrackingId(self.Mesh, "floquet_r")[0]

        def eps(x):
            if 0.4 <= x[0] < 0.6:
                return 10
            else:
                return 1

        self.EigenSolver = fempy.solver.tLaplacianEigenproblemSolver(
            self.Mesh, constrained_nodes = [self.RightNode], g = eps,
            typecode = num.Complex)

    def kTrack(self):
        rl = self.Lattice.ReciprocalLattice
        return [0.01*rl[0], 0.99*rl[0]]

    def getEigenvalues(self, k):
        print "computing for k = ",k
        self.EigenSolver.setupConstraints(
            {self.RightNode: (0, [(cmath.exp(1j*k[0]), self.LeftNode)])})
        pairs = self.EigenSolver.solve(0,
                                       tolerance = 1e-6,
                                       number_of_eigenvalues = 7)
        
        result = [evalue for evalue, em in pairs]
        result.sort(lambda e1, e2: cmp(abs(e1), abs(e2)))
        return result
    
class t2DProblem:
    def __init__(self):
        self.Lattice = pc.tBravaisLattice([num.array([1,0]), num.array([0, 1])])

    def kTrack(self):
        rl = lattice.ReciprocalLattice
        return [0*rl[0],
                0.5*rl[0],
                0.5*(rl[0]+rl[1]),
                0*rl[0]]

    def getEigenvalues(self, k):
        pass

problem = t1DProblem()
k_track = tools.interpolateVectorList(problem.kTrack(), 20)

def scale_eigenvalue(ev):
    return math.sqrt(abs(ev)) / (2 * math.pi)

bdfile = file(",,band_diagram.data", "w")
evlists = [problem.getEigenvalues(k) for k in k_track]
for i in range(len(evlists[0])):
    for j in range(len(k_track)):
        bdfile.write("%f\t%f\n" % (k_track[j][0], scale_eigenvalue(evlists[j][i])))
    bdfile.write("\n")

