from setuptools import setup, find_packages
import os
import versioneer

__author__     = "Johannes Hörmann"
__copyright__  = "Copyright 2020, IMTEK Simulation, University of Freiburg"
__maintainer__ = "Johannes Hörmann"
__email__      = "johannes.hoermann@imtek.uni-freiburg.de"
__date__       = "Mar 18, 2020"

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='jlhpy',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description='Nanotribology of surfactants',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
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
        entry_points={
            'console_scripts': [
             #   'fwrlm = imteksimfw.fireworks.scripts.fwrlm_run:main',
             #   'render = imteksimfw.fireworks.scripts.render_run:main',
            ]
        },
    )
