# bdr
This package implements the following paper: 

H. Law, D. Sutherland, D. Sejdinovic, S. Flaxman, __Bayesian Approaches to Distribution Regression__, in _Artificial Intelligence and Statistics (AISTATS)_, 2018. [arxiv](https://arxiv.org/abs/1705.04293)


To setup as a package, clone the repository and run
```
python setup.py develop
```

The directory is organised as follows:
* __bdr__: contains the main code
* __experiment_code__: contains the API code for all the networks
* __experiment_config__: contains the experimental setup/configurations described in the paper
* __results__: contains the trained networks of the results described in the paper, and the corresponding plot functions.

There are two main datasets found in this paper:
* __gamma synthetic data__: Simulated by `bdr/data/toy.py`, seed provided in `experiment_code`
* __IMDb-WIKI__: The reprensentation in <img src="https://latex.codecogs.com/gif.latex?R^{4096}" />



Representaion of the last layer of a VGG-16 CNN architecture
The preprocessed IMDB-wiki dataset can be, run bdr/data/astro/get.sh and bdr/data/aerosol/get.sh. The main code, including data, is organized in a package in the bdr folder.
