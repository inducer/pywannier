import math
import cmath
import fempy.tools as tools
import fempy.stopwatch

# Numeric imports -------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la




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
    def __init__(self, lattice, mesh, k_grid, has_inversion_symmetry, modes_start = {}):
        self.Lattice = lattice
        self.Mesh = mesh
        self.KGrid = k_grid
        self.HasInversionSymmetry = has_inversion_symmetry
        self.Modes = modes_start.copy()



    
class tInvertedModeLookerUpper:
    def __init__(self, grid_interval_counts):
        self._GridIntervalCounts = grid_interval_counts

    def __call__(self, dictionary, failed_key):
        new_key = tuple(map(lambda (idx, count): count-idx, 
                            zip(failed_key, self._GridIntervalCounts)))
        eigenvalue, eigenvector = dictionary[new_key]
        return (eigenvalue.conjugate(), num.conjugate(eigenvector))




class tInvertedModeListLookerUpper:
    def __init__(self, grid_interval_counts):
        self._GridIntervalCounts = grid_interval_counts

    def __call__(self, dictionary, failed_key):
        new_key = tuple(map(lambda (idx, count): count-idx, 
                            zip(failed_key, self._GridIntervalCounts)))
        modelist = dictionary[new_key]
        return tools.tFakeList(lambda i: (modelist[i][0].conjugate(),
                                          num.conjugate(modelist[i][1])),
                               len(modelist))




def findPeriodicityNodes(mesh, grid_vectors):
    bnodes = filter(lambda node: node.ConstraintId == "floquet",
                    mesh.dofManager().constrainedNodes())
    
    job = fempy.stopwatch.tJob("periodicity")

    periodicity_nodes = []
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
                periodicity_nodes.append((gv, node, [(other_node, 1)]))
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
                periodicity_nodes.append(
                    (gv, node,
                     [(other_node_a, dist_a/total_dist),
                      (other_node_b, dist_b/total_dist)]))
    job.done()

    return periodicity_nodes




def computeFloquetBCs(periodicity_nodes, k):
    result = []
    for gv, node, other_nodes in periodicity_nodes:
        floquet_factor = cmath.exp(-1j * num.innerproduct(gv, k))
        my_condition = [(node,1)]
        for other_node, factor in other_nodes:
            my_condition.append((other_node, -factor*floquet_factor))
        result.append(my_condition)
    return result




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

        def findNeighbors(i, j, di, dj, max_count = 2):
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

        for i,j in k_grid:
            k = k_grid[i,j]
            if crystal.HasInversionSymmetry and k[0] < 0:
                continue

            if first:
                band[i,j] = modes[i,j][band_index]
                continue
            
            neighbor_sets = []
            guessed_eigenvalues = []
            for direction in all_dirs:
                di = direction[0]
                dj = direction[1]
                neighbor_set = findNeighbors(i, j, di, dj)
            if len(neighbor_set):
                if len(neighbor_set) == 1:
                    ni, nj = neighbor_set[0]
                    guessed_eigenvalues.append(band[ni,nj][0])
                elif len(neighbor_set) == 2:
                    ni0, nj0 = neighbor_set[0]
                    ni1, nj1 = neighbor_set[1]
                    closer_eigenvalue = band[ni0,nj0][0]
                    further_eigenvalue = band[ni1,nj1][0]
                    guessed_eigenvalues.append(2*closer_eigenvalue - further_eigenvalue)
                elif len(neighbor_set) == 3:
                    # quadratic approximation
                    ni0, nj0 = neighbor_set[0]
                    ni1, nj1 = neighbor_set[1]
                    ni2, nj2 = neighbor_set[2]
                    closer_eigenvalue = band[ni0,nj0][0]
                    further_eigenvalue = band[ni1,nj1][0]
                    furthest_eigenvalue = band[ni2,nj2][0]
                    guessed_eigenvalues.append(3*closer_eigenvalue - 3*further_eigenvalue + furthest_eigenvalue)
                else:
                    raise RuntimeError, "unexpected neighbor set length"
                index = findClosestAt((i,j), guessed_eigenvalues)
                band[i,j] = modes[i,j][index]
                taken_eigenvalues[i,j].append(index)
        return band

    return [findBand(i) for i in range(len(modes[0,0]))]


    

def visualizeBandsGnuplot(filename, crystal, bands):
    k_grid = crystal.KGrid
    out_file = file(filename, "w")

    def scale_eigenvalue(ev):
        return math.sqrt(ev.real) / (2 * math.pi)

    def writePoint(key):
        spot = k_grid[key]
        out_file.write("%f\t%f\t%f\n" % (spot[0], spot[1], scale_eigenvalue(band[key][0])))

    def writeBlock((i,j)):
        writePoint((i,j))
        writePoint((i+1,j))
        writePoint((i+1,j+1))
        writePoint((i,j+1))
        writePoint((i,j))
        out_file.write("\n\n")
    
    for band in bands:
        k_grid.forEachBlock(writeBlock)




def visualizeBandsVTK(filename, crystal, bands):
    import pyvtk
    k_grid = crystal.KGrid

    nodes = []
    quads = []

    def scale_eigenvalue(ev):
        return math.sqrt(ev.real) / (2 * math.pi)

    def makeNode(key):
        node_number = len(nodes)
        spot = k_grid[key]
        nodes.append((spot[0], spot[1], 5*scale_eigenvalue(band[key][0])))
        node_lookup[key] = node_number

    def makeQuad((i,j)):
        quads.append((
            node_lookup[i,j],
            node_lookup[i+1,j],
            node_lookup[i+1,j+1],
            node_lookup[i,j+1],
            node_lookup[i,j]))
    
    for band in bands:
        node_lookup = {}
        for k_index in k_grid:
            makeNode(k_index)
        k_grid.forEachBlock(makeQuad)

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
        return math.sqrt(ev.real) / (2 * math.pi)

    k_grid = crystal.KGrid
    band_diagram_file = file(filename, "w")
    for band in bands:
        for index, k in enumerate(k_vectors):
            value = 0.j
            for weight, neighbor in k_grid.interpolateGridPointIndex(k):
                value += weight * band[neighbor][0]

            band_diagram_file.write("%d\t%f\n" % 
                                        (index, scale_eigenvalue(value)))
        band_diagram_file.write("\n")
