import math
import cmath
import fempy.tools as tools
import fempy.stopwatch
import fempy.mesh_function
import fempy.visualization

# Numeric imports -------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools




class tStepFunction:
    def __init__(self, left_value, step_at, right_value):
        self.LeftValue = left_value
        self.StepAt = step_at
        self.RightValue = right_value

    def __call__(self, x):
        if x < self.StepAt:
            return self.LeftValue
        else:
            return self.RightValue




class tCircularFunctionRemapper:
    def __init__(self, f):
        self.Function = f

    def __call__(self, x):
        return self.Function(tools.norm2(x))




class tBravaisLattice:
    def __init__(self, direct_lattice_basis):
        self.DirectLatticeBasis = direct_lattice_basis
                         
        # compute reciprocal lattice
        d = len(self.DirectLatticeBasis[0])
        mat = num.zeros((d*d, d*d), num.Float)
        rhs = num.zeros((d*d,), num.Float)

        equation_no = 0
        for direct_vector_number, direct_vec in enumerate(self.DirectLatticeBasis):
            for indirect_vector_number in range(d):
                for indirect_vector_coordinate in range(d):
                    mat[equation_no, indirect_vector_number * d + indirect_vector_coordinate] = \
                                     direct_vec[indirect_vector_coordinate]
                rhs[equation_no] = math.pi * 2 * tools.delta(direct_vector_number,
                                                             indirect_vector_number)
                equation_no += 1

        sol = la.solve_linear_equations(mat, rhs)
        self.ReciprocalLattice = []
        for indirect_vector_number in range(d):
            rec_vec = num.zeros((d,), num.Float)
            for indirect_vector_coordinate in range(d):
                rec_vec[indirect_vector_coordinate] = sol[indirect_vector_number * d + indirect_vector_coordinate]
            self.ReciprocalLattice.append(rec_vec)




class tPhotonicCrystal:
    def __init__(self, lattice, mesh, k_grid, has_inversion_symmetry, 
                 epsilon, modes_start = None):
        self.Lattice = lattice
        self.Mesh = mesh
        self.KGrid = k_grid
        self.HasInversionSymmetry = has_inversion_symmetry
        self.Modes = modes_start.copy()
        self.Epsilon = epsilon
        self.ScalarProduct = None
        self.Bands = None
        self.PeriodicBands = None



    
class tInvertedModeLookerUpper:
    def __init__(self, grid_interval_counts):
        self._GridIntervalCounts = grid_interval_counts

    def __call__(self, dictionary, failed_key):
        new_key = tuple(map(lambda (idx, count): count-idx, 
                            zip(failed_key, self._GridIntervalCounts)))
        eigenvalue, eigenmode = dictionary[new_key]
        return (eigenvalue.conjugate(), eigenmode.conjugate())




class tInvertedModeListLookerUpper:
    def __init__(self, grid_interval_counts):
        self._GridIntervalCounts = grid_interval_counts

    def __call__(self, dictionary, failed_key):
        new_key = tuple(map(lambda (idx, count): count-idx, 
                            zip(failed_key, self._GridIntervalCounts)))
        modelist = dictionary[new_key]
        return tools.tFakeList(lambda i: (modelist[i][0].conjugate(),
                                          modelist[i][1].conjugate()),
                               len(modelist))




class tInvertedIdenticalLookerUpper:
    def __init__(self, grid_interval_counts):
        self._GridIntervalCounts = grid_interval_counts

    def __call__(self, dictionary, failed_key):
        new_key = tuple(map(lambda (idx, count): count-idx, 
                            zip(failed_key, self._GridIntervalCounts)))
        return dictionary[new_key]




class tKPeriodicLookerUpper:
    def __init__(self, k_grid):
        self._KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        return dictionary[self._KGrid.chopUpperBoundary()
                          .reducePeriodically(failed_key)]




def makeKPeriodicLookupStructure(k_grid, dictionary = {}):
    return tools.tDependentDictionary(tKPeriodicLookerUpper(k_grid), 
                                      dictionary)




def findPeriodicityNodes(mesh, grid_vectors):
    bnodes = [node 
              for node in mesh.dofManager()
              if node.TrackingId == "floquet"]
    
    job = fempy.stopwatch.tJob("periodicity")

    periodicity_nodes = {}
    for node in bnodes:
        for gv in grid_vectors:
            ideal_point = node.Coordinates + gv
            dist_threshold = tools.norm2(gv) * 0.3

            close_nodes_with_dists = filter(
                lambda (n, dist): dist <= dist_threshold, 
                tools.decorate(
                lambda other_node: tools.norm2(ideal_point - other_node.Coordinates), bnodes))
            close_nodes_with_dists.sort(lambda (n1,d1), (n2,d2): cmp(d1,d2))
            
            if not close_nodes_with_dists:
                continue
            
            if close_nodes_with_dists[0][1] < 1e-5 * dist_threshold:
                # Found one node that is close enough, bet our
                # butt on it.
                other_node, dist = close_nodes_with_dists[0]
                periodicity_nodes[node] = (gv, [(other_node, 1)])
            else:
                if len(close_nodes_with_dists) == 1:
                    print "WARNING: Found only one node that's near enough."
                    continue

                other_node_a, dist_a = close_nodes_with_dists[0]
                other_node_b = None

                for candidate_b, dist in close_nodes_with_dists[1:]:
                    if tools.angleCosineBetweenVectors(other_node_a.Coordinates - ideal_point,
                                                       candidate_b.Coordinates - ideal_point) < -0.1:
                        other_node_b = candidate_b
                        dist_b = dist
            
                if other_node_b is None:
                    continue

                for candidate_b, dist in close_nodes_with_dists[1:]:
                    if tools.angleCosineBetweenVectors(other_node_a.Coordinates - ideal_point,
                                                       candidate_b.Coordinates - ideal_point) < -0.1:
                        other_node_b = candidate_b
                        dist_b = dist
                        break
                total_dist = dist_a + dist_b
                periodicity_nodes[node] = \
                                        (gv, 
                                         [(other_node_a, dist_a/total_dist),
                                          (other_node_b, dist_b/total_dist)])
    job.done()

    return periodicity_nodes




def getFloquetConstraints(periodicity_nodes, k):
    constraints = {}
    for dependent_node, (gv, independent_nodes) in periodicity_nodes.iteritems():
        floquet_factor = cmath.exp(-1j * num.innerproduct(gv, k))
        lincomb_specifier = []
        for independent_node, factor in independent_nodes:
            lincomb_specifier.append((factor * floquet_factor, independent_node))
        constraints[dependent_node] = 0, lincomb_specifier
    return constraints




class tBand:
    def __init__(self, crystal, structure):
        self.Structure = structure

        ev_abs = [abs(self[k_index][0]) for k_index in crystal.KGrid]
        self.MaxAbsolute = max(ev_abs)
        self.MinAbsolute = min(ev_abs)

    def __getitem__(self, index):
        return self.Structure[index]




def findBands(crystal):
    k_grid = crystal.KGrid
    k_grid_point_counts = k_grid.gridPointCounts()
    modes = crystal.Modes
    all_dirs = tools.enumerateBasicDirections(2)
    
    taken_eigenvalues = tools.tDictionaryWithDefault(lambda key: [])

    def findBand(band_index):
        def findClosestAt(key, eigenvalues):
            mk_values = map(lambda x: x[0], modes[key])
            mk_values_with_indices = zip(range(len(mk_values)), mk_values)

            # erase taken indices from possible choices
            taken = taken_eigenvalues[key]
            taken.sort()
            for i in taken[-1::-1]:
                mk_values_with_indices.pop(i)

            distances = [(index, sum([abs(item-ev)**2 for ev in eigenvalues]))
                         for index, item in mk_values_with_indices]
            list_index = tools.argmin(distances, lambda (i, ev): ev)
            return mk_values_with_indices[list_index][0]

        def findNeighbors((i, j), (di, dj), max_count = 2):
            result = []
            for step_count in range(max_count):
                i += di
                j += dj
                if 0 <= i < k_grid_point_counts[0] and \
                     0 <= j < k_grid_point_counts[1] and \
                     (i,j) in band:
                    result.append((i,j))
                else:
                    return result
            return result

        if crystal.HasInversionSymmetry:
            band = tools.tDependentDictionary(
                tInvertedModeLookerUpper(k_grid.gridIntervalCounts()))
        else:
            band = {}

        first = True

        for k_index in k_grid:
            k = k_grid[k_index]
            if crystal.HasInversionSymmetry and k[0] < 0:
                continue

            if first:
                band[k_index] = modes[k_index][band_index]
                continue
            
            neighbor_sets = []
            guessed_eigenvalues = []
            for direction in all_dirs:
                neighbor_set = findNeighbors(k_index, tuple(direction))
            if len(neighbor_set):
                if len(neighbor_set) == 1:
                    guessed_eigenvalues.append(band[neighbor_set[0]][0])
                elif len(neighbor_set) == 2:
                    closer_eigenvalue = band[neighbor_set[0]][0]
                    further_eigenvalue = band[neighbor_set[1]][0]
                    guessed_eigenvalues.append(2*closer_eigenvalue - further_eigenvalue)
                elif len(neighbor_set) == 3:
                    # quadratic approximation
                    closer_eigenvalue = band[neighbor_set[0]][0]
                    further_eigenvalue = band[neighbor_set[1]][0]
                    furthest_eigenvalue = band[neighbor_set[2]][0]
                    guessed_eigenvalues.append(3*closer_eigenvalue - 3*further_eigenvalue + furthest_eigenvalue)
                else:
                    raise RuntimeError, "unexpected neighbor set length"
                index = findClosestAt(k_index, guessed_eigenvalues)
                band[k_index] = modes[k_index][index]
                taken_eigenvalues[k_index].append(index)
        return band

    return [tBand(crystal, findBand(i)) for i in range(len(modes[0,0]))]


    

def visualizeBandsGnuplot(filename, crystal, bands):
    k_grid = crystal.KGrid
    out_file = file(filename, "w")

    def scale_eigenvalue(ev):
        return math.sqrt(abs(ev)) / (2 * math.pi)

    def writePoint(key):
        spot = k_grid[key]
        out_file.write("%f\t%f\t%f\n" % (spot[0], spot[1], scale_eigenvalue(band[key][0])))

    for band in bands:
        for i,j in k_grid.chopUpperBoundary():
            writePoint((i,j))
            writePoint((i+1,j))
            writePoint((i+1,j+1))
            writePoint((i,j+1))
            writePoint((i,j))
            out_file.write("\n\n")




def visualizeBandsVTK(filename, crystal, bands):
    import pyvtk
    k_grid = crystal.KGrid

    nodes = []
    quads = []

    def scale_eigenvalue(ev):
        return math.sqrt(abs(ev)) / (2 * math.pi)

    def makeNode(key):
        node_number = len(nodes)
        spot = k_grid[key]
        nodes.append((spot[0], spot[1], 5*scale_eigenvalue(band[key][0])))
        node_lookup[key] = node_number

    for band in bands:
        node_lookup = {}
        for k_index in k_grid:
            makeNode(k_index)

        for i,j in k_grid.chopUpperBoundary():
            quads.append((
                node_lookup[i,j],
                node_lookup[i+1,j],
                node_lookup[i+1,j+1],
                node_lookup[i,j+1],
                node_lookup[i,j]))
            
    structure = pyvtk.PolyData(points = nodes, polygons = quads)
    vtk = pyvtk.VtkData(structure, "Bands")
    vtk.tofile(filename, "ascii")




def writeEigenvalueLocusPlot(filename, crystal, bands, k_points):
    locus_plot_file = file(filename, "w")
    for band in bands:
        for k in k_points:
            closest_index = crystal.KGrid.findClosestGridPointIndex(k)
            eigenvalue = band[closest_index][0]
            locus_plot_file.write("%f\t%f\n" % (eigenvalue.real, eigenvalue.imag))
        locus_plot_file.write("\n")


  

def writeBandDiagram(filename, crystal, bands, k_vectors):
    def scale_eigenvalue(ev):
        return math.sqrt(abs(ev)) / (2 * math.pi)

    k_grid = crystal.KGrid
    band_diagram_file = file(filename, "w")
    for i,band in enumerate(bands):
        for index, k in enumerate(k_vectors):
            value = 0.j
            k_interp_info = k_grid.interpolateGridPointIndex(k)
            for weight, neighbor in k_interp_info:
                value += weight * band[neighbor][0]

            next_band_value = 0.j
            if i + 1 < len(bands):
                for weight, neighbor in k_interp_info:
                    next_band_value += weight * bands[i+1][neighbor][0]
            dist = 0.5 / (2*math.pi) * \
                   abs(cmath.sqrt(value) - cmath.sqrt(next_band_value))

            band_diagram_file.write("%d\t%f\t%f\n" % 
                                        (index, 
                                         scale_eigenvalue(value),
                                         dist))
        band_diagram_file.write("\n")




def analyzeBandStructure(bands):
    endpoints = []
    for idx, band in enumerate(bands):
        endpoints.append((band.MinAbsolute, "MIN", idx))
        endpoints.append((band.MaxAbsolute, "MAX", idx))
    endpoints.sort(lambda x,y: cmp(x[0], y[0]))

    # sweepline algorithm
    active_bands = []
    active_cluster = []
    clusters = []
    gaps = []
    last_evalue = 0.
    for (evalue, minmax, band_idx) in endpoints:
        if minmax == "MIN":
            if not active_bands:
                gaps.append((last_evalue, evalue))
            assert band not in active_bands
            active_bands.append(band_idx)
            if band_idx not in active_cluster:
                active_cluster.append(band_idx)
        else:
            active_bands.remove(band_idx)
            if not active_bands:
                clusters.append(active_cluster)
                active_cluster = []
        last_evalue = evalue

    return gaps, clusters




def normalizeModes(crystal, scalar_product_calculator):
    for key in crystal.KGrid:
        norms = []
        for index, (evalue, emode) in enumerate(crystal.Modes[key]):
            norm_squared = scalar_product_calculator(emode, emode)
            assert abs(norm_squared.imag) < 1e-10
            emode *= 1 / math.sqrt(norm_squared.real)




def periodicizeMeshFunction(mf, k, exponent = -1):
    vec = mf.vector()
    pvec = num.zeros(vec.shape, num.Complex)
    na = mf.numberAssignment()
    for node in mf.mesh().dofManager():
        pvec[na[node]] = vec[na[node]] * cmath.exp(1j * exponent *
                                                   mtools.sp(node.Coordinates, k))
    return mf.copy(vector = pvec)




def periodicizeBands(crystal, bands, exponent = -1):
    pbands = []
    for band in bands:
        pband = {}
        for ki in crystal.KGrid:
            if crystal.HasInversionSymmetry and crystal.KGrid[ki][0] < 0:
                continue
            pband[ki] = band[ki][0], \
                        periodicizeMeshFunction(band[ki][1],
                                                crystal.KGrid[ki],
                                                exponent)
        if crystal.HasInversionSymmetry:
            pband = tools.tDependentDictionary(
                tInvertedModeLookerUpper(crystal.KGrid.gridIntervalCounts()),
                pband)
        pbands.append(pband)
    return pbands




def visualizeGridFunction(multicell_grid, func_on_multicell_grid):
    offsets_and_mesh_functions = []
    for multicell_index in multicell_grid:
        R = multicell_grid[multicell_index]
        offsets_and_mesh_functions.append((R, func_on_multicell_grid[multicell_index]))
    fempy.visualization.visualizeSeveralMeshes("vtk", 
                                               (",,result.vtk", ",,result_grid.vtk"), 
                                               offsets_and_mesh_functions)

