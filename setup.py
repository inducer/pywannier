#!/usr/bin/env python

from distutils.core import setup,Extension
import glob
import os
import os.path

setup(name="pywannier",
      version="0.10",
      description="A Wannier function computation package",
      author="Andreas Kloeckner",
      author_email="ak@ixion.net",
      license = "GNU GPL",
      url="http://pywannier.sf.net",
      packages=["pywannier"],
      package_dir={"pywannier": "src"}
     )
