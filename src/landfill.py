
fempy.stopwatch.HIDDEN_JOBS.append("bcs, periodic")

def transformSolution(mesh, transform, solution):
    txinv = la.inverse(transform)
    new_solution = solution.copy()
    for node_number in range(len(mesh.dofManager())):
        node = mesh.dofManager()[node_number]
        before_tx_coord = num.matrixmultiply(txinv, node.Coordinates)
        el = mesh.findElement(before_tx_coord)
        relevant_part = num.take(solution, el.nodeNumbers())
        new_solution[node_number] = el.getFormFunctionCombination(relevant_part)(before_tx_coord)
    return new_solution
    
def checkSymmetry(mapped_k, eigenpairs, eigenvalue_map, eigenvector_map):
    p_nodes = pc.computeFloquetBCs(periodicity_nodes, mapped_k)
    eigensolver.addPeriodicBoundaryConditions(p_nodes)

    res_sum = 0.
    for value, vector in eigenpairs:
        res_sum += eigensolver.computeEigenpairResidual(eigenvalue_map(value),
                                                        eigenvector_map(vector))
                                                    
    return res_sum

def checkTransform(k_space_matrix, eigenvalue_map, eigenvector_map):
    res_sum = 0.
    for index in crystal.KGrid.asSequence().getAllIndices():
        k = crystal.KGrid[index]
        transformed_k = num.matrixmultiply(k_space_matrix, k)
        this_res = checkSymmetry(transformed_k, crystal.Modes[index], eigenvalue_map, eigenvector_map)
        print "for k=", k, ":", this_res
        res_sum += this_res
    return res_sum

print "coord swap:"
coord_swap_mat = num.array([[0.,1.],[1.,0]])
print checkTransform(coord_swap_mat, lambda x: x, 
                     lambda vec: transformSolution(crystal.Mesh, coord_swap_mat, vec))
print "inversion:"
inversion_mat = num.array([[-1.,0],[0,-1.]])
print checkTransform(inversion_mat, lambda x: x.conjugate(), num.conjugate)
# ---------------------------------------------------------

raw_ks = [0 * rl[0], 0.5 * rl[0], 0.5 * (rl[0]+rl[1]), 0 * rl[0]]
ks_of_keys = tools.interpolateVectorList(raw_ks, 20)
keys = range(len(list_of_ks))

# ---------------------------------------------------------

pc.writeEigenvalueLocusPlot(",,ev_locus.data", crystal, bands, 
                            [0*rl[0], 0.25 * rl[0], 0.5 * rl[0]])

# ---------------------------------------------------------
def generateHierarchicGaussians(crystal, typecode):
    for l in tools.generateAllPositiveIntegerTuples(2,1):
        div_x, div_y = tuple(l)
        dlb = crystal.Lattice.DirectLatticeBasis
        h_x = dlb[0] / div_x
        h_y = dlb[1] / div_y

        def gaussian(point):
            result = 0
            for idx_x in range(div_x):
                y_result = 0
                for idx_y in range(div_y):
                    y_result += math.exp(-20*div_y**2*mtools.sp(dlb[1], point-(idx_y+.5)*h_y+dlb[1]/2)**2)
                result += y_result * \
                          math.exp(-20*div_x**2*mtools.sp(dlb[0], point-(idx_x+.5)*h_x+dlb[0]/2)**2)
            return result
        yield fempy.mesh_function.discretizeFunction(crystal.Mesh, 
                                                     gaussian, 
                                                     typecode,
                                                     crystal.NodeNumberAssignment)


