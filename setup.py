#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='sphericalSampling',
        version='0.1.0',

        description='Spherical Sampling',
        author='Niklas Winter',
        author_email='niklas.winter@audiolabs-erlangen.de',

        license='open source',
        packages=setuptools.find_packages(
            where='src',
            include=['sphericalSampling']
        ),
        package_dir={"": "src"},
        install_requires=[
            'numpy>=1.20.2',
            'scipy>=1.8.0',
            'mayavi>=4.7.2',
            'matplotlib >=3.4.1',
            'PyQt5 >= 5.15.6'
        ],

        extras_require={
            'tests': [
                'pytest>=6.2.3',
            ],
            'notebooks' : [
                'notebook>=6.4.8'
            ]
            # 'docs': [
            #     'matplotlib>=3.3.4',
            #     'sphinx>=4.3.0',
            #     'sphinxcontrib-napoleon>=0.7',
            #     'sphinx_rtd_theme>=1.0.0',
            #     'numpydoc>=1.1.0',
            # ],
        },

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.8',
            'Topic :: Mathmatics  :: Function Synthesis'
        ],

        zip_safe=True,
        include_package_data=True,
    )
