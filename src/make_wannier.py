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
my_eigenmodes = cPickle.load(file(",,eigenmodes.pickle", "rb"))
job.done()

eigenmodes.writeBandDiagram(",,band_diagram.data", my_eigenmodes)

  if False:
    raw_ks = [0 * rl[0], 0.5 * rl[0], 0.5 * (rl[0]+rl[1]), 0 * rl[0]]
    ks_of_keys = tools.interpolateVectorList(raw_ks, 20)
    keys = range(len(list_of_ks))
