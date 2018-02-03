#!/bin/bash

cd $(dirname $0)
wget -c http://www.gatsby.ucl.ac.uk/\~dougals/data/imdb-faces/grouped-4/{data.csv.gz,feats.npy,emp_cov{.npy,_eigs.npz}}
