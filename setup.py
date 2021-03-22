from setuptools import setup, find_packages

# What packages are required for this module to be executed?
REQUIRED = ["scikit-learn", "deepchem>=2.4.0"]
setup(name="nas_gcn", packages=find_packages(), install_requires=REQUIRED)
