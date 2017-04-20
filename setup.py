from setuptools import setup, find_packages

setup(
    name='hmm',
    version='0.0.1',
    description='Functions for working with HMMs',
    url='https://github.com/n-s-f/hmm',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'examples']),
)
