import math, cmath, sys, operator, random
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.algorithms as algo
import pylinear.matrix_tools as mtools
import pylinear.iteration as iteration

import scipy.optimize

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.stopwatch
import fempy.solver
import fempy.eoc
import fempy.tools as tools
import fempy.integration
import fempy.mesh_function
import fempy.visualization as visualization

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc




def verifyBandOrthogonality(crystal, spc, bands, threshold):
    for key in crystal.KGrid:
        for index1 in range(len(bands)):
            for index2 in range(index1, len(bands)):
                emode1 = bands[index1][key][1]
                emode2 = bands[index2][key][1]
                sp = spc(emode1, emode2) 
                assert abs(sp-mtools.delta(index1, index2)) < threshold

def run():
    job = fempy.stopwatch.tJob("loading")
    crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
    job.done()

    crystal = crystals[0]
    node_number_assignment = crystal.Modes[0,0][0][1].numberAssignment()

    spc = fempy.mesh_function.tScalarProductCalculator(node_number_assignment,
                                                       crystal.ScalarProduct)

    job = fempy.stopwatch.tJob("checking bloch functions")
    verifyBandOrthogonality(crystal, spc, crystal.Bands, 1e-6)
    job.done()

    job = fempy.stopwatch.tJob("checking periodicized bloch functions")
    verifyBandOrthogonality(crystal, spc, crystal.PeriodicBands, 1e-2)
    job.done()
    print "OK"

if __name__ == "__main__":
    run()


