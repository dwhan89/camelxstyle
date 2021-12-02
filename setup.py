#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import setuptools
from numpy.distutils.core import setup, Extension, build_ext, build_src
import versioneer
import os, sys

requirements =  ['numpy>=1.16',
                 'astropy>=2.0',
                 'setuptools>=39',
                 'scipy>=1.0',
                 'matplotlib>=2.0'
                 ]
    
fcflags = os.getenv('FCFLAGS')

setup(
    author="Dongwon 'DW' Han",
    author_email='dwhan89@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="camel",
    package_dir={"camelxstyle": "camelxstype"},
    entry_points={
    },
    ext_modules=[
    ],
    include_dirs = [],
    library_dirs = [],
    install_requires=requirements,
    extras_require = {},
    license="BSD license",
    package_data={},
    include_package_data=True,    
    data_files=[],
    keywords='camelxstyle',
    name='camelxstyle',
    url='https://github.com/simonsobs/camelxstyle',
    version=versioneer.get_version(),
)

print('\n[setup.py request was successful.]')

