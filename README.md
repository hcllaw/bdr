# bdr
This package implements the following paper: 

H. Law, D. Sutherland, D. Sejdinovic, S. Flaxman, __Bayesian Approaches to Distribution Regression__, in _Artificial Intelligence and Statistics (AISTATS)_, 2018. [arxiv](https://arxiv.org/abs/1705.04293)


To setup as a package, clone the repository and run
```
python setup.py develop
```
## Structure
The directory is organised as follows:
* __bdr__: contains the main code, including data scripts
* __experiment_code__: contains the API code for all the networks
* __experiment_config__: contains the experimental setup/configurations described in the paper
* __results__: contains the results of the experiments described in the paper, and the corresponding plot functions for them.

## Data
There are two main datasets found in this paper:
* __gamma synthetic data__: Simulated by `bdr/data/toy.py`, seed provided in `experiment_code`
* __IMDb-WIKI features__: Features of celebrity images taken from the output layer of a VGG-16 CNN architecture

The IMDb-WIKI features can be download by running
```
bdr/data/imdb_faces/all-feats/get.sh
bdr/data/imdb_faces/grouped-4/get.sh
```
The python notebook `/bdr/data/imdb_faces/group_data.ipynb` provides the data cleaning process, analysis and grouping process.

## Main API
The main API can be found in `/experiment_code`, where:
* `train_test.py`: contains API code for RBF network, shrinkage, shrinkageC and also fourier features network.
* `blr.py`: contains API code for bayesian linear regression.
* `chi2_make_optimal.py`: contains API code for the Bayes-optimal for the varying bag size experiment.

All API make use of the `argparse` package, i.e. parameters can be specified in command line. To bring up the manual (and see the default options), run in command line:
```
python train_test.py --help
```
An example of constructing a shrinkage network and training it with simulated data using `argparse` is shown here:
```
 python train_test.py chi2 -n shrinkage --learning-rate 0.001 --n-landmarks 50 /folder/to/save/to
 ```
 This would train a shrinkage network on simulated data with learning rate 0.001, 50 landmarks and save results to `/folder/to/save/to`. For other parameters not specified, it would use the default options. Likewise, the `blr.py` and `chi2_make_optimal.py` can be used in a similar fashion. It is also noted by default, it will parallise, unless specified otherwise.
 
 ## Reproducing the experiments
There are 3 main experiments in the paper, namely _Varying bag size: Uncertainty in the inputs_, _Fixed bag size: Uncertainty in the regression model_ and _IMDb-WIKI: Age Estimation_. The experimental setup and the exact grid for each of these experiments can be found in `/experiment_config`. Note that the each network on the IMDb-WIKI dataset will take around 4 hours to train roughly (depending on the parameters) on four 2 - E5-2690 v4 @ 2.60GHz	CPUs.

The `/results` folder contains the results of the models that performed best on the validation set for our experiments, each experiment results folder also contain corresponding notebooks for baseline or plotting purposes.
