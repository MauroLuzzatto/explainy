#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Mauro Luzzatto",
    author_email='mauroluzzatto@hotmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="explainy is a library for generating explanations for machine learning models in Python. It uses methods from Machine Learning Explainability and provides a standardized API to create feature importance explanations for samples. The explanations are generated in the form of plots and text.",
    entry_points={
        'console_scripts': [
            'explainy=explainy.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='explainy',
    name='explainy',
    packages=find_packages(include=['explainy', 'explainy.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MauroLuzzatto/explainy',
    version='0.1.1',
    zip_safe=False,
)
