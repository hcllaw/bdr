# bdr
NOT IN PRODUCTION YET
Bayesian Distribution Regression

To download the data, run `bdr/data/astro/get.sh` and `bdr/data/aerosol/get.sh`.
The main code, including data, is organized in a package in the [`bdr`](bdr) folder.

Experiment-specific stuff is in [`experiments`](experiments).
Code in `experiments` assumes that `import bdr` works;
run `python setup.py develop` so that's true.
(If you prefer, you could also put this folder on your `PYTHONPATH`.)
