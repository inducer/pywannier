import photonic_crystal as pc
import fempy.stopwatch
import fempy.tools as tools
import cPickle as pickle

def run():
    job = fempy.stopwatch.Job("loading")
    crystals = pickle.load(file(",,crystal.pickle", "rb"))
    job.done()

    for crystal in crystals:
        spc = fempy.mesh_function.tScalarProductCalculator(crystal.NodeNumberAssignment,
                                                           crystal.MassMatrix)
        job = fempy.stopwatch.Job("calculating periodic modes")
        crystal.PeriodicModes = pc.periodicize_modes(crystal, 
                                                    crystal.Modes)
        job.done()

        job = fempy.stopwatch.Job("unifying phases: bloch modes")
        pc.unify_phases(crystal.Modes)
        job.done()

        job = fempy.stopwatch.Job("unifying phases: periodic modes")
        pc.unify_phases(crystal.PeriodicModes)
        job.done()

        job = fempy.stopwatch.Job("normalizing bloch modes")
        pc.normalize_modes(crystal.KGrid, crystal.Modes, spc)
        job.done()

        job = fempy.stopwatch.Job("normalizing periodic modes")
        pc.normalize_modes(crystal.KGrid, crystal.PeriodicModes, spc)
        job.done()

        print "degeneracies:", pc.find_degeneracies(crystal)

        job = fempy.stopwatch.Job("finding bands")
        crystal.Bands = pc.find_bands(crystal, crystal.Modes, spc)
        job.done()

        crystal.PeriodicBands = [tools.DependentDictionary(
            pc.KPeriodicLookerUpper(crystal.KGrid),
            band.copy(new_modes = crystal.PeriodicModes))
                                 for band in crystal.Bands]

    job = fempy.stopwatch.Job("saving")
    pickle.dump(crystals, file(",,crystal_bands.pickle", "wb"), pickle.HIGHEST_PROTOCOL)
    job.done()

if __name__ == "__main__":
    run()
