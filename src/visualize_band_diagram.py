import math, sys
import cmath
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.stopwatch
import fempy.solver
import fempy.tools as tools

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc



job = fempy.stopwatch.tJob("loading")
crystals = pickle.load(file(",,crystal.pickle", "rb"))
job.done()

crystal = crystals[0]

job = fempy.stopwatch.tJob("localizing bands")
bands = pc.findBands(crystal)
job.done()

rl = crystal.Lattice.ReciprocalLattice
k_track = [0*rl[0],
           0.5*rl[0],
           0.5*(rl[0]+rl[1]),
           0*rl[0]]
pc.writeBandDiagram(",,band_diagram.data", crystal, bands,
                    tools.interpolateVectorList(k_track, 30))
