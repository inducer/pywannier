import cPickle
# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.stopwatch

# Local imports ---------------------------------------------------------------
import eigenmodes

job = fempy.stopwatch.tJob("loading")
results = cPickle.load(file(",,eigenmodes.pickle", "r"))
job.done()
print results

print len(results.Mesh.dofManager())
