import math, cmath, sets
import fempy.tools as tools
import fempy.stopwatch
import fempy.mesh_function
import fempy.mesh
import fempy.geometry
import fempy.visualization

# Numeric imports -------------------------------------------------------------
import pylinear.matrices as num
import pylinear.linear_algebra as la
import pylinear.matrix_tools as mtools




class tConstantFunction:
    def __init__(self, value):
        self.Value = value

    def __call__(self, x):
        return self.Value




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
                 epsilon):
        self.Lattice = lattice
        self.Mesh = mesh
        self.KGrid = k_grid
        self.HasInversionSymmetry = has_inversion_symmetry
        self.Epsilon = epsilon
        self.NodeNumberAssignment = None
        self.MassMatrix = None

        self.Modes = None
        self.PeriodicModes = None

        self.Bands = None
        self.PeriodicBands = None



    
def invertKIndex(k_grid, k_index):
    return  tuple([(high-1-(idx-low))+low
                   for (idx, (low, high)) in zip(k_index, k_grid.limits())])




class tInvertedModeLookerUpper:
    def __init__(self, k_grid):
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        new_key = invertKIndex(self.KGrid, failed_key)
        eigenvalue, eigenmode = dictionary[new_key]
        return (eigenvalue.conjugate(), eigenmode.conjugate())




class tInvertedModeListLookerUpper:
    def __init__(self, k_grid):
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        new_key = invertKIndex(self.KGrid, failed_key)
        modelist = dictionary[new_key]
        return tools.tFakeList(lambda i: (modelist[i][0].conjugate(),
                                          modelist[i][1].conjugate()),
                               len(modelist))




class tInvertedIdenticalLookerUpper:
    def __init__(self, k_grid):
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        new_key = invertKIndex(self.KGrid, failed_key)
        return dictionary[new_key]




class tReducedBrillouinModeListLookerUpper:
    """This class is meant as lookup function for a tools.tDependentDictionary.
    It will map all of k-space to the left (k[0]<0) half of the Brillouin zone,
    excluding the top rim.

    This is the version for the Modes array.
    """

    def __init__(self, k_grid):
        self.HSize = k_grid.gridPointCounts()[0]
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        reduced_key = self.KGrid.reducePeriodically(failed_key)
        if self.KGrid[reduced_key][0] > 0:
            inverted_key = invertKIndex(self.KGrid, reduced_key)
            modelist = dictionary[inverted_key]
            return tools.tFakeList(lambda i: (modelist[i][0].conjugate(),
                                              modelist[i][1].conjugate()),
                                   len(modelist))
        else:
            # only needs reduction
            return dictionary[reduced_key]




class tReducedBrillouinLookerUpper:
    """This class is meant as lookup function for a tools.tDependentDictionary.
    It will map all of k-space to the left (k[0]<0) half of the Brillouin zone,
    excluding the top rim.

    This is the version for the bands array.
    """

    def __init__(self, k_grid):
        self.HSize = k_grid.gridPointCounts()[0]
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        reduced_key = self.KGrid.reducePeriodically(failed_key)
        if self.KGrid[reduced_key][0] > 0:
            inverted_key = invertKIndex(self.KGrid, reduced_key)
            evalue, mode = dictionary[inverted_key]
            return evalue.conjugate(), mode.conjugate()
        else:
            # only needs reduction
            return dictionary[reduced_key]




class tKPeriodicLookerUpper:
    def __init__(self, k_grid):
        self._KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        return dictionary[self._KGrid.reducePeriodically(failed_key)]




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
    def __init__(self, crystal, modes, indices):
        self.Crystal = crystal
        self.Modes = modes
        self.Indices = indices

        ev_abs = [abs(self[k_index][0]) for k_index in crystal.KGrid]
        self.MaxAbsolute = max(ev_abs)
        self.MinAbsolute = min(ev_abs)

    def copy(self, new_modes = None):
        return tBand(self.Crystal, new_modes or self.Modes, self.Indices)

    def __getitem__(self, k_index):
        return self.Modes[k_index][self.Indices[k_index]]




def findDegeneracies(crystal, threshold = 1e-3):
    result = {}
    for k_index in crystal.KGrid:
        indices = range(len(crystal.Modes[k_index]))
        indices.sort(lambda i1, i2: cmp(abs(crystal.Modes[k_index][i1][0]), 
                                        abs(crystal.Modes[k_index][i2][0])))
        
        for i in range(len(indices) -1 ):
            evalue_this = crystal.Modes[k_index][indices[i]][0]
            evalue_next = crystal.Modes[k_index][indices[i+1]][0]
            if abs(evalue_this - evalue_next) < threshold:
                degeneracies_here = result.setdefault(k_index, [])
                found = False
                for deg in degeneracies_here:
                    if indices[i] in deg:
                        found = True
                        print deg
                        deg.append(indices[i+1])

                if not found:
                    degeneracies_here.append([indices[i],indices[i+1]])
    return result

        
   

def findBands(crystal, modes, scalar_product_calculator):
    """Requires the eigenmodes to have norm 1.

    Returns a list of tBand objects.
    """

    all_dirs = tools.enumerateBasicDirections(2)
    spc = scalar_product_calculator
    k_grid = crystal.KGrid
    
    taken_eigenvalues = {}

    def findNeighbors(k_index, k_index_increment, max_count, band):
        result = []
        for step_count in range(max_count):
            k_index = k_grid.reducePeriodically(
                tools.addTuples(k_index, k_index_increment))

            if k_index in band:
                result.append(k_index)
            else:
                return result
        return result

    def findClosestAt(k_index, eigenvalues, eigenmodes):
        indices = [i 
                   for i in range(len(modes[k_index]))
                   if i not in taken_eigenvalues[k_index]]

        distances = {}
        sps = {}
        joint_scores = {}
        for index in indices:
            evalue, emode = modes[k_index][index]
            distances[index] = sum([abs(evalue-ref_evalue) for ref_evalue in eigenvalues])
            sps[index] = tools.average([abs(spc(emode, ref_emode)) for ref_emode in eigenmodes])

            # FIXME inherently bogus
            # or, rather, needs parameter adjustment...
            joint_scores[index] = len(eigenmodes)/(1e-20+sps[index]) + distances[index]

        best_index = indices[tools.argmin(indices, lambda i: distances[i])]
        return best_index

        best_value = distances[best_index]

        tied_values = [index
                       for index in indices
                       if distances[index] < 2 * best_value]

        if len(tied_values) == 1:
            # no other tied values, push out result
            return best_index
        else:
            # use "joint score" as a tie breaker
            best_index_joint = tied_values[tools.argmin(tied_values, 
                                                        lambda i: joint_scores[i])]

            return best_index_joint

    def findBand(band_index):
        band_indices = makeKPeriodicLookupStructure(k_grid)

        # reset taken_eigenvalues
        for k_index in k_grid:
            taken_eigenvalues[k_index] = sets.Set()
            band_indices[k_index] = band_index
            
        if False:
            print "WARNING: No-op findBands"
            return tBand(crystal, modes, band_indices)

        for k_index in k_grid:
            k = k_grid[k_index]

            guessed_eigenvalues = []
            close_eigenmodes = []
            for direction in all_dirs:
                neighbor_set = findNeighbors(k_index, tuple(direction), 2, band_indices)
                if len(neighbor_set) == 0:
                    pass
                elif len(neighbor_set) == 1:
                    n0 = neighbor_set[0]
                    close_eigenmodes.append(modes[n0][band_indices[n0]][1])

                    guessed_eigenvalues.append(modes[n0][band_indices[n0]][0])
                elif len(neighbor_set) == 2:
                    n0 = neighbor_set[0]
                    n1 = neighbor_set[1]
                    close_eigenmodes.append(modes[n0][band_indices[n0]][1])

                    # linear approximation
                    n0_eigenvalue = modes[n0][band_indices[n0]][0]
                    n1_eigenvalue = modes[n0][band_indices[n1]][0]
                    guessed_eigenvalues.append(2*n0_eigenvalue 
                                               - n1_eigenvalue)
                elif len(neighbor_set) == 3:
                    n0 = neighbor_set[0]
                    n1 = neighbor_set[1]
                    n2 = neighbor_set[2]
                    close_eigenmodes.append(modes[n0][band_indices[n0]][1])

                    # quadratic approximation
                    n0_eigenvalue = modes[n0][band_indices[n0]][0]
                    n1_eigenvalue = modes[n0][band_indices[n1]][0]
                    n2_eigenvalue = modes[n0][band_indices[n2]][0]
                    guessed_eigenvalues.append(3*n0_eigenvalue 
                                               - 3*n1_eigenvalue 
                                               + n2_eigenvalue)
                else:
                    raise RuntimeError, "unexpected neighbor set length %d at %s" % (
                        len(neighbor_set), k_index)

            assert len(guessed_eigenvalues) > 0
            assert len(close_eigenmodes) > 0

            index = findClosestAt(k_index, guessed_eigenvalues, close_eigenmodes)
            band_indices[k_index] = index
            taken_eigenvalues[k_index].add(index)
        return tBand(crystal, modes, band_indices)

    return [findBand(i) for i in range(len(modes[0,0]))]


    

def visualizeBandsGnuplot(filename, crystal, bands):
    k_grid = crystal.KGrid
    out_file = file(filename, "w")

    def scale_eigenvalue(ev):
        return math.sqrt(abs(ev)) / (2 * math.pi)

    def writePoint(key):
        spot = k_grid[key]
        out_file.write("%f\t%f\t%f\n" % (spot[0], spot[1], scale_eigenvalue(band[key][0])))

    for band in bands:
        for i,j in k_grid:
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

            band_diagram_file.write("%d\t%f\n" % 
                                        (index, 
                                         scale_eigenvalue(value)))
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




def normalizeModes(k_grid, modes, scalar_product_calculator):
    for k_index in k_grid:
        for index, (evalue, emode) in enumerate(modes[k_index]):
            norm_squared = scalar_product_calculator(emode, emode)
            assert abs(norm_squared.imag) < 1e-10
            emode /= math.sqrt(norm_squared.real)




def periodicizeMeshFunction(mf, k, exponent = -1):
    vec = mf.vector()
    pvec = num.zeros(vec.shape, num.Complex)
    na = mf.numberAssignment()

    exponent *= 1j

    for node in mf.mesh().dofManager():
        pvec[na[node]] = vec[na[node]] * cmath.exp(exponent *
                                                   mtools.sp(node.Coordinates, k))
    return mf.copy(vector = pvec)




def periodicizeModes(crystal, modes, exponent = -1, verify = False):
    if crystal.HasInversionSymmetry:
        pmodes = tools.tDependentDictionary(
            tInvertedModeListLookerUpper(crystal.KGrid))
    else:
        pmodes = {}

        
    for k_index in crystal.KGrid.enlargeAtBothBoundaries():
        k = crystal.KGrid[k_index]
        if crystal.HasInversionSymmetry and k[0] > 0:
            continue

        pmodes[k_index] = []
        for evalue, emode in modes[k_index]:
            pmodes[k_index].append((evalue,
                                    periodicizeMeshFunction(emode, k, exponent)))

    if verify:
        for k_index in crystal.KGrid.enlargeAtBothBoundaries():
            k = crystal.KGrid[k_index]
            for (evalue, emode), (pevalue, pemode) in zip(modes[k_index], pmodes[k_index]):
                this_pmode = periodicizeMeshFunction(emode, k, exponent)
                assert tools.norm2((this_pmode - pemode).vector()) < 1e-10

    return pmodes




def visualizeGridFunction(multicell_grid, func_on_multicell_grid):
    offsets_and_mesh_functions = []
    for multicell_index in multicell_grid:
        R = multicell_grid[multicell_index]
        offsets_and_mesh_functions.append((R, func_on_multicell_grid[multicell_index]))
    fempy.visualization.visualizeSeveralMeshes("vtk", 
                                               (",,result.vtk", ",,result_grid.vtk"), 
                                               offsets_and_mesh_functions)




def generateSquareMeshWithRodCenter(lattice, inner_radius, coarsening_factor = 1, 
                                    constraint_id = "floquet",
                                    use_exact = True):
    def needsRefinement( vert_origin, vert_destination, vert_apex, area ):
        bary_x = ( vert_origin.x() + vert_destination.x() + vert_apex.x() ) / 3
        bary_y = ( vert_origin.y() + vert_destination.y() + vert_apex.y() ) / 3
        
        dist_center = math.sqrt( bary_x**2 + bary_y**2 )
        if dist_center < inner_radius * 1.2:
            return area >= 2e-3 * coarsening_factor
        else:
            return area >= 1e-2 * coarsening_factor

    geometry = [fempy.mesh.tShapeSection(
        fempy.geometry.getParallelogram(lattice.DirectLatticeBasis), constraint_id),
                fempy.mesh.tShapeSection(
        fempy.geometry.getCircle(inner_radius, use_exact), None)]

    return fempy.mesh.tTwoDimensionalMesh(geometry, refinement_func = needsRefinement)
