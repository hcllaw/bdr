from setuptools import setup, find_packages

setup(
    name='bdr',
    version='0.1.0',
    author='Ho Chung Leon Law',
    author_email='ho.law@spc.ox.ac.uk',
    url='https://github.com/hlaw/bdr/',
    install_requires=[
        'scipy',
        'numpy',
        'scikit-learn',
        'six',
        'scipy',
        'tqdm',
        'glob',
#        'matplotlib',
#        'sealion'
     ],
    description='Code for bayesian approaches to distribution regression.',
    license='MIT'
)
