import photonic_crystal as pc
import fempy.stopwatch
import cPickle as pickle

def run():
    job = fempy.stopwatch.tJob("loading")
    crystals = pickle.load(file(",,crystal.pickle", "rb"))
    job.done()

    for crystal in crystals:
        node_number_assignment = crystal.Modes[0,0][0][1].numberAssignment()

        spc = fempy.mesh_function.tScalarProductCalculator(node_number_assignment,
                                                           crystal.ScalarProduct)

        job = fempy.stopwatch.tJob("normalizing modes")
        pc.normalizeModes(crystal, spc)
        job.done()

        job = fempy.stopwatch.tJob("finding bands")
        crystal.Bands = pc.findBands(crystal, spc)
        job.done()

        job = fempy.stopwatch.tJob("periodicizing bands")
        crystal.PeriodicBands = pc.periodicizeBands(crystal, 
                                                    crystal.Bands)
        job.done()

    job = fempy.stopwatch.tJob("saving")
    pickle.dump(crystals, file(",,crystal_bands.pickle", "wb"), pickle.HIGHEST_PROTOCOL)
    job.done()

if __name__ == "__main__":
    run()
