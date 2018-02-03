#!/bin/bash

cd $(dirname $0)
wget -c http://www.gatsby.ucl.ac.uk/\~dougals/data/imdb-faces/all-feats/{{fc7s,probs}.npy,meta.npz}
