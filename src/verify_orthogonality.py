import math, cmath, sys, operator, random
import cPickle as pickle

# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.toybox as toybox

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




def verify_band_orthogonality(crystal, spc, bands, threshold):
    violations = 0
    total = 0
    for k_index in crystal.KGrid:
        for index1 in range(len(bands)):
            for index2 in range(index1, len(bands)):
                total += 1
                emode1 = bands[index1][k_index][1]
                emode2 = bands[index2][k_index][1]
                sp = spc(emode1, emode2) 
                err = abs(sp-toybox.delta(index1, index2))
                if err >= threshold:
                    print k_index, index1, index2, err, "is over the threshold"
                    violations += 1
    return violations, total

def run():
    job = fempy.stopwatch.tJob("loading")
    crystals = pickle.load(file(",,crystal_bands.pickle", "rb"))
    job.done()

    crystal = crystals[0]

    for crystal in crystals:
        spc = fempy.mesh_function.tScalarProductCalculator(crystal.NodeNumberAssignment,
                                                           crystal.MassMatrix)

        job = fempy.stopwatch.tJob("checking bloch functions")
        vio, tot = verify_band_orthogonality(crystal, spc, crystal.Bands, 2e-3)
        job.done()
        print vio, "violations out of", tot

        job = fempy.stopwatch.tJob("checking periodicized bloch functions")
        vio, tot = verify_band_orthogonality(crystal, spc, crystal.PeriodicBands, 3e-3)
        job.done()
        print vio, "violations out of", tot

if __name__ == "__main__":
    run()


