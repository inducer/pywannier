import photonic_crystal as pc
import fempy.stopwatch
import fempy.tools as tools
import cPickle as pickle

def projectionOnto(basis, spc, vec):
    return tools.linearCombination([spc(vec, bv) for bv in basis], basis)

def run():
    job = fempy.stopwatch.tJob("loading")
    crystals = pickle.load(file(",,crystal.pickle", "rb"))
    job.done()

    for crystal in crystals:
        spc = fempy.mesh_function.tScalarProductCalculator(crystal.NodeNumberAssignment,
                                                           crystal.MassMatrix)
        job = fempy.stopwatch.tJob("calculating periodic modes")
        crystal.PeriodicModes = pc.periodicizeModes(crystal, 
                                                    crystal.Modes)
        job.done()

        job = fempy.stopwatch.tJob("unifying phases: bloch modes")
        pc.unifyPhases(crystal.Modes)
        job.done()

        job = fempy.stopwatch.tJob("unifying phases: periodic modes")
        pc.unifyPhases(crystal.PeriodicModes)
        job.done()

        job = fempy.stopwatch.tJob("normalizing bloch modes")
        pc.normalizeModes(crystal.KGrid, crystal.Modes, spc)
        job.done()

        job = fempy.stopwatch.tJob("normalizing periodic modes")
        pc.normalizeModes(crystal.KGrid, crystal.PeriodicModes, spc)
        job.done()

        print "degeneracies:", pc.findDegeneracies(crystal)

        job = fempy.stopwatch.tJob("finding bands")
        crystal.Bands = pc.findBands(crystal, 
                                     crystal.Modes, 
                                     spc)
        job.done()

        crystal.PeriodicBands = [tools.tDependentDictionary(
            pc.tKPeriodicLookerUpper(crystal.KGrid),
                                     band.copy(new_modes = crystal.PeriodicModes))
            for band in crystal.Bands]

    job = fempy.stopwatch.tJob("saving")
    pickle.dump(crystals, file(",,crystal_bands.pickle", "wb"), pickle.HIGHEST_PROTOCOL)
    job.done()

if __name__ == "__main__":
    run()
