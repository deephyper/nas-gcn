from setuptools import setup, find_packages

# What packages are required for this module to be executed?
REQUIRED = [
    "scikit-learn==0.21.3",
    "deepchem==2.4.0rc1.dev20200819015415"
]
setup(
    name='nas_gcn',
    packages=find_packages(),
    install_requires=REQUIRED
)