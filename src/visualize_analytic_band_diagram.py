import fempy.tools as tools
import photonic_crystal as pc
import pylinear.matrices as num
import math

dimensions = 1

def getEigenvalues(k):
    eigenvalues_here = []
    for tup in tools.generateAllIntegerTuplesBelow(5, dimensions):
        lambda_ = sum([(2*math.pi*i+k[idx])**2 for idx, i in enumerate(tup)]) + 0.j
        eigenvalues_here.append(lambda_)

    #eigenvalues_here.sort(lambda ev1, ev2: cmp(abs(ev1), abs(ev2)))
    return eigenvalues_here

def scale_eigenvalue(ev):
    return math.sqrt(abs(ev)) / (2 * math.pi)

if dimensions == 1:
    lattice = pc.tBravaisLattice([num.array([1])])
    rl = lattice.ReciprocalLattice
    k_track = [0*rl[0], rl[0]]

elif dimensions == 2:
    lattice = pc.tBravaisLattice([num.array([1,0]), num.array([0, 1])])
    rl = lattice.ReciprocalLattice
    k_track = [0*rl[0],
               0.5*rl[0],
               0.5*(rl[0]+rl[1]),
               0*rl[0]]

k_track = tools.interpolateVectorList(k_track, 30)

bdfile = file(",,band_diagram_analytic.data", "w")
evlists = [getEigenvalues(k) for k in k_track]
for i in range(len(evlists[0])):
    for j in range(len(k_track)):
        bdfile.write("%f\t%f\n" % (k_track[j][0], scale_eigenvalue(evlists[j][i])))
    bdfile.write("\n")

