class tEigenmodes:
  def __init__(self, mesh):
    self.Mesh = mesh
    self.KValues = []

  def addKValue(self, k, values, vectors):
    self.KValues.append((k, values, vectors))

