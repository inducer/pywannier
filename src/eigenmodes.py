import math
import fempy.tools as tools




class tEigenmodes:
  def __init__(self, mesh):
    self.Mesh = mesh
    self.Modes = {}

  def add(self, key, values, vectors):
    self.Modes[key] = values,vectors



    
def writeBandDiagram(filename, eigenmodes, k_grid):
  band_diagram_file = file(filename, "w")
  for index, key in tools.indexAnd(eigenmodes.Modes.keys()):
    values, vectors = eigenmodes.Modes[key]
    for val in values:
      band_diagram_file.write("%d\t%f\n" % (index, math.sqrt(val.real)))
