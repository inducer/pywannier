import math, cmath, sets
import pytools
import pytools.stopwatch as stopwatch

import fempy.mesh_function
import fempy.mesh
import fempy.geometry
import fempy.visualization

# Numeric imports -------------------------------------------------------------
import pylinear.array as num
import pylinear.linear_algebra as la
import pylinear.operation as op
import pylinear.toybox as toybox




class ConstantFunction:
    def __init__(self, value):
        self.Value = value

    def __call__(self, x):
        return self.Value




class StepFunction:
    def __init__(self, left_value, step_at, right_value):
        self.LeftValue = left_value
        self.StepAt = step_at
        self.RightValue = right_value

    def __call__(self, x):
        if x < self.StepAt:
            return self.LeftValue
        else:
            return self.RightValue




class CircularFunctionRemapper:
    def __init__(self, f):
        self.Function = f

    def __call__(self, x):
        return self.Function(op.norm_2(x))




class BravaisLattice:
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
                    rhs[equation_no] = math.pi * 2 * pytools.delta(direct_vector_number,
                                                               indirect_vector_number)
                equation_no += 1

        sol = la.solve_linear_equations(mat, rhs)
        self.ReciprocalLattice = []
        for indirect_vector_number in range(d):
            rec_vec = num.zeros((d,), num.Float)
            for indirect_vector_coordinate in range(d):
                rec_vec[indirect_vector_coordinate] = sol[indirect_vector_number * d + indirect_vector_coordinate]
            self.ReciprocalLattice.append(rec_vec)




class PhotonicCrystal:
    def __init__(self, lattice, mesh, boundary, k_grid, has_inversion_symmetry, 
                 epsilon):
        self.Lattice = lattice
        self.Mesh = mesh
        self.BoundaryShapeSection = boundary
        self.KGrid = k_grid
        self.HasInversionSymmetry = has_inversion_symmetry
        self.Epsilon = epsilon
        self.NodeNumberAssignment = None
        self.MassMatrix = None

        self.Modes = None
        self.PeriodicModes = None

        self.Bands = None
        self.PeriodicBands = None



    
def invert_k_index(k_grid, k_index):
    return  tuple([(high-1-(idx-low))+low
                   for (idx, (low, high)) in zip(k_index, k_grid.limits())])




class InvertedModeLookerUpper:
    def __init__(self, k_grid):
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        new_key = invert_k_index(self.KGrid, failed_key)
        eigenvalue, eigenmode = dictionary[new_key]
        return (eigenvalue.conjugate(), eigenmode.conjugate())




class InvertedModeListLookerUpper:
    def __init__(self, k_grid):
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        new_key = invert_k_index(self.KGrid, failed_key)
        modelist = dictionary[new_key]
        return pytools.FakeList(lambda i: (modelist[i][0].conjugate(),
                                           modelist[i][1].conjugate()),
                                len(modelist))




class InvertedIdenticalLookerUpper:
    def __init__(self, k_grid):
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        new_key = invert_k_index(self.KGrid, failed_key)
        return dictionary[new_key]




class ReducedBrillouinModeListLookerUpper:
    """This class is meant as lookup function for a pytools.DependentDictionary.
    It will map all of k-space to the left (k[0]<0) half of the Brillouin zone,
    excluding the top rim.

    This is the version for the Modes array.
    """

    def __init__(self, k_grid):
        self.HSize = k_grid.grid_point_counts()[0]
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        reduced_key = self.KGrid.reduce_periodically(failed_key)
        if self.KGrid[reduced_key][0] > 0:
            inverted_key = invert_k_index(self.KGrid, reduced_key)
            modelist = dictionary[inverted_key]
            return pytools.FakeList(lambda i: (modelist[i][0].conjugate(),
                                               modelist[i][1].conjugate()),
                                    len(modelist))
        else:
            # only needs reduction
            return dictionary[reduced_key]




class ReducedBrillouinLookerUpper:
    """This class is meant as lookup function for a pytools.DependentDictionary.
    It will map all of k-space to the left (k[0]<0) half of the Brillouin zone,
    excluding the top rim.

    This is the version for the bands array.
    """

    def __init__(self, k_grid):
        self.HSize = k_grid.gridPointCounts()[0]
        self.KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        reduced_key = self.KGrid.reduce_periodically(failed_key)
        if self.KGrid[reduced_key][0] > 0:
            inverted_key = invert_k_index(self.KGrid, reduced_key)
            evalue, mode = dictionary[inverted_key]
            return evalue.conjugate(), mode.conjugate()
        else:
            # only needs reduction
            return dictionary[reduced_key]




class KPeriodicLookerUpper:
    def __init__(self, k_grid):
        self._KGrid = k_grid

    def __call__(self, dictionary, failed_key):
        return dictionary[self._KGrid.reduce_periodically(failed_key)]




def make_k_periodic_lookup_structure(k_grid, dictionary = {}):
    return pytools.DependentDictionary(KPeriodicLookerUpper(k_grid), 
                                       dictionary)




def find_periodicity_nodes(mesh, boundary, grid_vectors, order = 2):
    bnodes = [node 
              for node in mesh.dof_manager()
              if node.TrackingId == "floquet"]
    
    job = stopwatch.Job("periodicity")

    periodicity_nodes = {}
    for node in bnodes:
        for gv in grid_vectors:
            ideal_point = node.Coordinates + gv
            if not boundary.contains_point(ideal_point):
                continue

            dist_threshold = op.norm_2(gv) * 0.3

            close_nodes_with_dists = filter(
                lambda (n, dist): dist <= dist_threshold, 
                pytools.decorate(
                lambda other_node: op.norm_2(ideal_point - other_node.Coordinates), bnodes))
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
                    print "WARNING: Found just one near node. Bummer.."
                    continue

                first_other_node, dummy = close_nodes_with_dists[0]
                first_other_co = first_other_node.Coordinates
                direction = first_other_co - ideal_point
                
                other_nodes_with_alpha = [(first_other_node, 1.)]
                for candidate, dist in close_nodes_with_dists[1:]:
                    dtl, alpha = pytools.distanceToLine(ideal_point, direction, candidate.Coordinates)
                    if dtl < 1e-5:
                        other_nodes_with_alpha.append((candidate, alpha))
                    if len(other_nodes_with_alpha) >= order+1:
                        break

                if len(other_nodes_with_alpha) < order+1:
                    print "WARNING: Found an insufficient number of near nodes, degraded approximation."

                icoeffs = toybox.find_interpolation_coefficients(
                    num.array([alpha for dummy, alpha in other_nodes_with_alpha], num.Float),
                    0.)

                nodes_with_icoeffs = [(other_node, icoeff) 
                                      for icoeff, (other_node, alpha) 
                                      in zip(icoeffs, other_nodes_with_alpha)]

                periodicity_nodes[node] = (gv, nodes_with_icoeffs)
    job.done()

    return periodicity_nodes




def get_floquet_constraints(periodicity_nodes, k):
    constraints = {}
    for dependent_node, (gv, independent_nodes) in periodicity_nodes.iteritems():
        lincomb_specifier = []
        floquet_factor = cmath.exp(-1j*gv*k)
        for independent_node, factor in independent_nodes:
            lincomb_specifier.append((factor * floquet_factor, independent_node))
        constraints[dependent_node] = 0, lincomb_specifier
    return constraints




class Band:
    def __init__(self, crystal, modes, indices):
        self.Crystal = crystal
        self.Modes = modes
        self.Indices = indices

        ev_abs = [abs(self[k_index][0]) for k_index in crystal.KGrid]
        self.MaxAbsolute = max(ev_abs)
        self.MinAbsolute = min(ev_abs)

    def copy(self, new_modes = None):
        return Band(self.Crystal, new_modes or self.Modes, self.Indices)

    def __getitem__(self, k_index):
        return self.Modes[k_index][self.Indices[k_index]]




def find_degeneracies(crystal, threshold = 1e-3):
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

        
   

def find_bands(crystal, modes, scalar_product_calculator):
    """Requires the eigenmodes to have norm 1.

    Returns a list of Band objects.
    """

    all_dirs = pytools.enumerate_basic_directions(2)
    spc = scalar_product_calculator
    k_grid = crystal.KGrid
    
    taken_eigenvalues = {}
    for k_index in k_grid:
        taken_eigenvalues[k_index] = sets.Set()

    def find_neighbors(k_index, k_index_increment, max_count, band):
        result = []
        for step_count in range(max_count):
            k_index = k_grid.reduce_periodically(
                pytools.add_tuples(k_index, k_index_increment))

            if k_index in band:
                result.append(k_index)
            else:
                return result
        return result

    def find_closest_at(k_index, eigenvalues, eigenmodes):
        indices = [i 
                   for i in range(len(modes[k_index]))
                   if i not in taken_eigenvalues[k_index]]

        distances = {}
        sps = {}
        joint_scores = {}
        for index in indices:
            evalue, emode = modes[k_index][index]
            distances[index] = sum([abs(evalue-ref_evalue) for ref_evalue in eigenvalues])
            sps[index] = pytools.average([abs(spc(emode, ref_emode)) for ref_emode in eigenmodes])

            # FIXME inherently bogus
            # or, rather, needs parameter adjustment...
            joint_scores[index] = len(eigenmodes)/(1e-20+sps[index]) + distances[index]

        best_index = indices[pytools.argmin(indices, lambda i: distances[i])]
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
            best_index_joint = tied_values[pytools.argmin(tied_values, 
                                                        lambda i: joint_scores[i])]

            return best_index_joint

    def find_band(band_index):
        band_indices = make_k_periodic_lookup_structure(k_grid)

        # reset taken_eigenvalues
        for k_index in k_grid:
            band_indices[k_index] = band_index
            
        if False:
            print "WARNING: No-op findBands"
            return Band(crystal, modes, band_indices)

        for k_index in k_grid:
            k = k_grid[k_index]

            guessed_eigenvalues = []
            close_eigenmodes = []
            for direction in all_dirs:
                neighbor_set = find_neighbors(k_index, tuple(direction), 2, band_indices)
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

            index = find_closest_at(k_index, guessed_eigenvalues, close_eigenmodes)
            band_indices[k_index] = index
            taken_eigenvalues[k_index].add(index)
        return Band(crystal, modes, band_indices)

    return [find_band(i) for i in range(len(modes[0,0]))]



    

def visualize_bands_gnuplot(filename, crystal, bands):
    k_grid = crystal.KGrid
    out_file = file(filename, "w")

    def scale_eigenvalue(ev):
        return math.sqrt(abs(ev)) / (2 * math.pi)

    def write_point(key):
        spot = k_grid[key]
        out_file.write("%f\t%f\t%f\n" % (spot[0], spot[1], scale_eigenvalue(band[key][0])))

    for band in bands:
        for i,j in k_grid:
            write_point((i,j))
            write_point((i+1,j))
            write_point((i+1,j+1))
            write_point((i,j+1))
            write_point((i,j))
            out_file.write("\n\n")




def visualize_bands_vtk(filename, crystal, bands):
    import pyvtk
    k_grid = crystal.KGrid

    nodes = []
    quads = []

    def scale_eigenvalue(ev):
        return math.sqrt(abs(ev)) / (2 * math.pi)

    node_lookup = {}
    for i,j in k_grid:
        spot = k_grid[i,j]
        node_lookup[i,j] = len(nodes)
        nodes.append((spot[0], spot[1], 0))

    for i,j in k_grid.chop_upper_boundary():
        quads.append((
            node_lookup[i,j],
            node_lookup[i+1,j],
            node_lookup[i+1,j+1],
            node_lookup[i,j+1],
            node_lookup[i,j]))


    datasets = []
    for i, band in enumerate(bands):
        values = []
        for k_index in k_grid:
            values.append(scale_eigenvalue(band[k_index][0]))
        datasets.append(pyvtk.Scalars(values, "band%d" % i, "default"))
            
    structure = pyvtk.PolyData(points = nodes, polygons = quads)
    vtk = pyvtk.VtkData(structure, "Bands", pyvtk.PointData(*datasets))
    vtk.tofile(filename, "ascii")




def write_eigenvalue_locus_plot(filename, crystal, bands, k_points):
    locus_plot_file = file(filename, "w")
    for band in bands:
        for k in k_points:
            closest_index = crystal.KGrid.find_closest_grid_point_index(k)
            eigenvalue = band[closest_index][0]
            locus_plot_file.write("%f\t%f\n" % (eigenvalue.real, eigenvalue.imag))
        locus_plot_file.write("\n")


  

def write_band_diagram(filename, crystal, bands, k_vectors):
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




def analyze_band_structure(bands):
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




def normalize_modes(k_grid, modes, scalar_product_calculator):
    for k_index in k_grid:
        for index, (evalue, emode) in enumerate(modes[k_index]):
            norm_squared = scalar_product_calculator(emode, emode)
            assert abs(norm_squared.imag) < 1e-10
            emode /= math.sqrt(norm_squared.real)




def get_exp_ikr_mesh_function(mesh, number_assignment, k, exponent=1):
    mf = fempy.mesh_function.MeshFunction(mesh, number_assignment,
                                          typecode=num.Complex)
    vec = mf.vector()
    na = number_assignment
    exponent *= 1j

    for node in mesh.dof_manager():
        vec[na[node]] = cmath.exp(exponent * k * node.Coordinates)
    return mf




def periodicize_mesh_function(mf, k, exponent=-1):
    vec = mf.vector()
    pvec = num.zeros(vec.shape, num.Complex)
    na = mf.numberAssignment()

    exponent *= 1j

    for node in mf.mesh().dof_manager():
        pvec[na[node]] = vec[na[node]] * cmath.exp(exponent*node.Coordinates*k)
    return mf.copy(vector = pvec)




def periodicize_modes(crystal, modes, exponent=-1, verify=False):
    if crystal.HasInversionSymmetry:
        pmodes = pytools.DependentDictionary(
            InvertedModeListLookerUpper(crystal.KGrid))
    else:
        pmodes = {}

        
    for k_index in crystal.KGrid.enlarge_at_both_boundaries():
        k = crystal.KGrid[k_index]

        exp_ikr = get_exp_ikr_mesh_function(crystal.Mesh,
                                            crystal.NodeNumberAssignment,
                                            k, exponent)
        if crystal.HasInversionSymmetry and k[0] > 0:
            continue

        pmodes[k_index] = []
        for evalue, emode in modes[k_index]:
            pmodes[k_index].append((evalue, exp_ikr * emode))

    if verify:
        for k_index in crystal.KGrid.enlarge_at_both_boundaries():
            k = crystal.KGrid[k_index]
            for (evalue, emode), (pevalue, pemode) in zip(modes[k_index], pmodes[k_index]):
                this_pmode = periodicize_mesh_function(emode, k, exponent)
                assert op.norm_2((this_pmode - pemode).vector()) < 1e-10

    return pmodes




def visualize_grid_function(multicell_grid, func_on_multicell_grid):
    offsets_and_mesh_functions = []
    for multicell_index in multicell_grid:
        R = multicell_grid[multicell_index]
        offsets_and_mesh_functions.append((R, func_on_multicell_grid[multicell_index]))
    fempy.visualization.visualizeSeveralMeshes("vtk", 
                                               (",,result.vtk", ",,result_grid.vtk"), 
                                               offsets_and_mesh_functions)




def generate_square_mesh_with_rod_center(lattice, inner_radius, coarsening_factor = 1, 
                                         constraint_id = "floquet",
                                         use_exact = True):
    def needs_refinement( vert_origin, vert_destination, vert_apex, area ):
        bary_x = ( vert_origin.x() + vert_destination.x() + vert_apex.x() ) / 3
        bary_y = ( vert_origin.y() + vert_destination.y() + vert_apex.y() ) / 3
        
        dist_center = math.sqrt( bary_x**2 + bary_y**2 )
        if dist_center < inner_radius * 1.2:
            return area >= 2e-3 * coarsening_factor
        else:
            return area >= 1e-2 * coarsening_factor

    boundary = fempy.mesh.ShapeSection(
        fempy.geometry.get_parallelogram(lattice.DirectLatticeBasis), constraint_id)
    geometry = [boundary,
                fempy.mesh.ShapeSection(
        fempy.geometry.get_circle(inner_radius, use_exact), None)]

    return fempy.mesh.TwoDimensionalMesh(
        geometry, refinement_func = needs_refinement), boundary




def unify_phases(modes):
    def find_abs_max(mesh_funcs):
        max_val = 0
        max_idx = None

        for i in range(len(mesh_funcs[0].vector())):
            this_val = 0
            for mf in mesh_funcs:
                this_val += abs(mf.vector()[i])
            if this_val > max_val:
                max_idx = i
                max_val = this_val
        return max_idx

    mfs = [mode[1]
           for key, modelist in modes.iteritems()
           for mode in modelist
           ]
    max_i = find_abs_max(mfs)
    for key, modelist in modes.iteritems():
        for mode in modelist:
            mf = mode[1]
            mf /= (mf.vector()[max_i] / abs(mf.vector()[max_i]))
            assert abs(mode[1].vector()[max_i].imag) < 1e-13




