from __future__ import unicode_literals

import os
import sys

from setuptools import find_packages, setup

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (OSError, IOError, ImportError):
    long_description = open('README.md').read()


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner>=2.9'] if needs_pytest else []

    setup_requires = ["cython", "numpy"] + pytest_runner
    install_requires = [
        'scikit-learn', 'limix-core>=1.0.1',
        'dask[array,bag,dataframe,delayed]>=0.14', 'h5py',
        'pandas-plink>=1.2.1', 'limix-legacy>=0.8.12', 'glimix-core>=1.2.19',
        'joblib>=0.11', 'tqdm>=4.10', 'scipy>=0.19', 'distributed',
        'numpy-sugar>=1.0.47', 'ncephes>=1.0.40', 'asciitree>=0.3.3',
        'scipy>=0.13', 'numpy>=1.6', 'matplotlib>=1.2', 'nose', 'pandas',
        'limix>=1.0.12', 'scipy-sugar', 'bottleneck', 'pp'
    ]

    tests_require = ['pytest', 'pytest-console-scripts', 'pytest-pep8']

    console_scripts = [
        'runLiMMBo=limmbo.bin.runLiMMBo:entry_point',
        'runSimpleVD=limmbo.bin.runSimpleVD:entry_point'
    ]

    metadata = dict(
        name='limmbo',
        version='0.1',
        maintainer="Hannah Meyer",
        maintainer_email="hannah@ebi.ac.uk",
        author="Hannah Meyer, Francesco Paolo Casale",
        author_email="hannah@ebi.ac.uk, casale@ebi.ac.u",
        description=('Linear mixed model bootstrapping'),
        url="https://github.com/HannahVMeyer/limmbo",
        long_description=long_description,
        license="Apache License 2.0",
        keywords='linear mixed models, covariance estimation',
        packages=find_packages(),
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        include_package_data=True,
        entry_points={
            'console_scripts': console_scripts
        })

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)


if __name__ == '__main__':
    setup_package()
