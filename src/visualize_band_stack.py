import cPickle as pickle

# fempy -----------------------------------------------------------------------
import fempy.stopwatch

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc




job = fempy.stopwatch.tJob("loading")
crystals = pickle.load(file(",,crystal.pickle", "rb"))
job.done()

crystal = crystals[0]

job = fempy.stopwatch.tJob("localizing bands")
bands = pc.findBands(crystal)
job.done()

pc.visualizeBandsVTK(",,bands.vtk", crystal, bands[0:8])
