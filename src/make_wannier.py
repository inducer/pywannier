import cPickle
# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.stopwatch

# Local imports ---------------------------------------------------------------
import eigenmodes
import lattice

job = fempy.stopwatch.tJob("loading")
lattice, eigenmodes = cPickle.load(file(",,eigenmodes.pickle", "r"))
job.done()


print len(eigenmodes.Mesh.dofManager())
