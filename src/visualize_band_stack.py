import cPickle as pickle

import pytools

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc




job = pytools.stopwatch.Job("loading")
crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
job.done()

crystal = crystals[-1]

pc.visualize_bands_vtk(",,bands.vtk", crystal, crystal.Bands)
