import math
import fempy.tools as tools




class tEigenmodes:
  def __init__(self, lattice, mesh, k_grid):
    self.Lattice = lattice
    self.Mesh = mesh
    self.KGrid = k_grid
    self.Modes = {}

  def add(self, key, values, vectors):
    self.Modes[key] = values,vectors



    
def findBands(eigenmodes):
  grid = eigenmodes.KGrid
  grid_subdivs = grid.subdivisions()
  modes = eigenmodes.Modes
  all_dirs = tools.enumerateDirections(2)

  def findBand(index):
    taken_eigenvalues = tools.tDictionaryWithDefault(lambda key: [])

    def findClosestAt(key, eigenvalues):
      mk_values = modes[key][0]

      # erase taken indices from possible choices
      taken = taken_eigenvalues[key]
      taken.sort()
      for i in taken[-1::-1]:
        mk_values.pop(i)

      index, minimum = tools.argmin(lambda item: 
                                    sum([abs(item-ev) for ev in eigenvalues]), 
                                    mk_values)
      return index

    def findNeighbors(i, j, di, dj, max_count = 2):
      result = []
      for i in range(max_count):
        i += di
        j += dj
        if 0 <= i <= grid_subdivs[0] and \
           0 <= j <= grid_subdivs[1]:
          result.append((i,j))
        else:
          return result
      return result
          
    band = {}
    band{0,0} = modes[0,0][index][0]
    all_indices = grid.asSequence().getAllIndices()
    all_indices.remove((0,0))
    for i,j in all_indices:
      neighbor_sets = []
      for direction in all_dirs:
        di = direction[0]
        dj = direction[1]
        neighbors.append(findNeighbors(i, j, di, dj))
      for neighbor_set in neighbor_sets:
        if len(neighbor_set) == 1:
          pass
        elif len(neighbor_set) == 2:
          pass
        else:
          raise RuntimeError, "unexpected neighbor set length"

  return [findBand(i) for i in range(len(modes[0,0][0]))]


    

def writeBandDiagram(filename, eigenmodes, k_grid):
  band_diagram_file = file(filename, "w")
  for index, key in tools.indexAnd(eigenmodes.Modes.keys()):
    values, vectors = eigenmodes.Modes[key]
    for val in values:
      band_diagram_file.write("%d\t%f\n" % (index, math.sqrt(val.real)))
