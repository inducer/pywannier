import cPickle as pickle

# fempy -----------------------------------------------------------------------
import fempy.stopwatch

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc




job = fempy.stopwatch.tJob("loading")
crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
job.done()

crystal = crystals[-1]

pc.visualizeBandsVTK(",,bands.vtk", crystal, crystal.Bands)
