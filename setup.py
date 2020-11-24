import os
from setuptools import setup, find_packages
from setuptools_scm import get_version

__author__ = "Johannes Hörmann"
__copyright__ = "Copyright 2020, IMTEK Simulation, University of Freiburg"
__maintainer__ = "Johannes Hörmann"
__email__ = "johannes.hoermann@imtek.uni-freiburg.de"
__date__ = "Mar 18, 2020"

module_dir = os.path.dirname(os.path.abspath(__file__))
readme = open(os.path.join(module_dir, 'README.md')).read()
version = get_version(root='.', relative_to=__file__)


def local_scheme(version):
    """Skip the local version (eg. +xyz of 0.6.1.dev4+gdf99fe2)
    to be able to upload to Test PyPI"""
    return ""

setup(
    name='jlhpy',
    use_scm_version={
        "root": '.',
        "relative_to": __file__,
        "write_to": os.path.join("jlhpy", "version.py"),
        "local_scheme": local_scheme},
    description='Nanotribology of surfactants',
    long_description=readme,
    url='https://github.com/jotelha/N_surfactant_on_substrate_template',
    author='Johannes Hörmann',
    author_email='johannes.hoermann@imtek.uni-freiburg.de',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6.5',
    zip_safe=False,
    install_requires=[
        'fireworks>=1.9.5',
    ],
    setup_requires=['setuptools_scm'],
)
