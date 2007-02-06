import math, cmath, random
import cPickle as pickle

import pytools
import pytools.grid
import pytools.stopwatch as stopwatch

# Numerics imports ------------------------------------------------------------
import pylinear.array as num
import pylinear.linear_algebra as la
import pylinear.computation as comp
import pylinear.toybox as toybox
import pylinear.randomized as randomized
import pylinear.iteration as iteration

import scipy.optimize

# fempy -----------------------------------------------------------------------
import fempy.mesh
import fempy.solver
import fempy.eoc
import fempy.integration
import fempy.mesh_function

# Local imports ---------------------------------------------------------------
import photonic_crystal as pc




from localizer_tools import *




def generate_random_gaussians(crystal, typecode):
    dlb = crystal.Lattice.DirectLatticeBasis
    while True:
        i = random.randint(0,len(dlb)-1)
        center_coords = num.zeros((len(dlb),), num.Float)
        center_coords[i] = random.uniform(-0.4, 0.4)
        center = dlb[i] * pytools.linear_combination(center_coords, dlb)

        sigma = num.zeros((len(dlb), len(dlb)), num.Float)
        for i in range(len(dlb)):
            max_width = min(1-center_coords[i], center_coords[i])
            sigma[i,i] = random.uniform(0.1, max_width)
        sigma_inv = la.inverse(sigma)
            
        # FIXME this is dependent on dlb actually being unit vectors
        def gaussian(point):
            arg = sigma_inv*(point - center)
            return math.exp(-comp.norm_2_squared(arg))

        yield fempy.mesh_function.discretize_function(crystal.Mesh, 
                                                      gaussian, 
                                                      typecode,
                                                      crystal.NodeNumberAssignment)

def generate_axial_gaussians(crystal, typecode):
    dlb = crystal.Lattice.DirectLatticeBasis
    dim = len(dlb)
    points = [num.zeros((dim,))] + \
            [0.25*pytools.linear_combination(dir, dlb)
                for dir in pytools.enumerate_basic_directions(dim)]

    for center in points:
        sigma = 1./6

        def gaussian(point):
            return math.exp(-comp.norm_2_squared(point-center)/sigma**2)

        result = fempy.mesh_function.discretize_function(crystal.Mesh, 
                                                      gaussian, 
                                                      typecode,
                                                      crystal.NodeNumberAssignment)
        #from fempy.visualization import visualize
        #visualize("vtk", (",,result.vtk", ",,result_grid.vtk"),
            #result.absolute())
        #raw_input("[Enter]")
        yield result



    
def guess_initial_mix_matrix(crystal, bands, sp):
    # generate the gaussians
    gaussians = []
    #gaussian_it = generate_random_gaussians(crystal, num.Complex)
    gaussian_it = generate_axial_gaussians(crystal, num.Complex)
    for n in range(len(bands)):
        gaussians.append(gaussian_it.next())

    # project the gaussians
    projected_bands = []
    projected_bands_co = []
    for n in range(len(bands)):
        projected_band = {}
        projected_band_co = {}

        for k_index in crystal.KGrid:
            mf = fempy.mesh_function.discretize_function(
                crystal.Mesh, lambda x: 0., num.Complex, 
                number_assignment = crystal.NodeNumberAssignment)
            coordinates = num.zeros((len(bands),), num.Complex)
            for m in range(len(bands)):
                coordinates[m] = sp(gaussians[n], bands[m][k_index][1])
                mf += coordinates[m] * bands[m][k_index][1]
            projected_band[k_index] = mf
            projected_band_co[k_index] = coordinates
        projected_bands.append(projected_band)
        projected_bands_co.append(projected_band_co)

    # orthogonalize the projected gaussians
    mix_matrix = DictionaryOfMatrices()
    for k_index in crystal.KGrid:
        # calculate scalar products
        my_sps = num.zeros((len(bands), len(bands)), num.Complex)
        for n in range(len(bands)):
            for m in range(m+1):
                my_sp = sp(projected_bands[n][k_index], projected_bands[m][k_index])
                my_sps[n,m] = my_sp
                my_sps[m,n] = my_sp.conjugate()

        #inv_sqrt_my_sps = 1/comp.cholesky(my_sps)
        inv_sqrt_my_sps = toybox.apply_f_to_symmetric(
                lambda x: 1/math.sqrt(x), 
                my_sps)

        mix_matrix[k_index] = num.zeros((len(bands), len(bands)), num.Complex)
        for n in range(len(bands)):
            # determine and compute correct mixture of projected bands
            mix_matrix[k_index][n] = pytools.linear_combination(
                inv_sqrt_my_sps[n], 
                [projected_bands_co[i][k_index] 
                 for i in range(len(bands))])
                
    return mix_matrix





# band mixing -----------------------------------------------------------------
def compute_mixed_bands(crystal, bands, mix_matrix):
    # WARNING! Don't be tempted to insert symmetry code in here, since
    # mix_matrix is of potentially unknown symmetry.

    result = []
    for n in range(len(bands)):
        band = {}

        for k_index in crystal.KGrid:
            # set eigenvalue to 0 since there is no meaning attached to it
            band[k_index] = 0, pytools.linear_combination(mix_matrix[k_index][n],
                                                        [bands[i][k_index][1] 
                                                         for i in range(len(bands))])
        result.append(band)
    return result




def compute_mixed_periodic_bands(crystal, pbands, mix_matrix):
    # WARNING! Don't be tempted to insert symmetry code in here, since
    # mix_matrix is of potentially unknown symmetry.

    result = []
    for n in range(len(pbands)):
        pband = {}

        for k_index in crystal.KGrid.enlarge_at_both_boundaries():
            reduced_k_index = crystal.KGrid.reduce_periodically(k_index)

            # set eigenvalue to 0 since there is no meaning attached to it
            pband[k_index] = 0.j, pytools.linear_combination(mix_matrix[reduced_k_index][n],
                                                           [pbands[i][k_index][1] 
                                                            for i in range(len(pbands))])
        result.append(pband)
    return result





