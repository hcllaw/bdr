#!/bin/bash
set -e
cd $(dirname $0)
py=$HOME/anaconda/envs/mmd/bin/python

n_cpus=4

# sleep random amount to avoid iniital race condition if started simultaneously
# python -c 'import time, random; time.sleep(random.uniform(.1, 3))'

function handle {
    # inputs: min_size, max_size, name, file, data_args, rest_of_args
    out=shrinkage_experiment_smallman5_fixed/$min_size/$max_size/$name
    if [[ ! -e $out/results.npz && ! -e ${out}.running ]]; then
        echo $out
        # warning: this can have a fun race condition
        mkdir -p $out
        touch ${out}.running
        # ./run_slurm.py -n $out -- OMP_NUM_THREADS=1 \
        $py -u $file $data_args $rest_of_args $out 2>&1 | tee ${out}/output
        rm ${out}.running
    fi
}

#mean_size=20
#for std_size in 5 10 15 20 25 30 35 40 45 50; do
for min_size in 0 5 10 15 20 25 30 35 40 45 50; do
    max_size=$(echo 50 - $min_size | bc)
    data_args="chi2 --data-seed 47 --size-type manual \
               --min-size $min_size --max-size $max_size \
               --n-train 150"
               #--size-mean $mean_size --size-std $std_size"
    common_args="--no-opt-landmarks --dtype-double --n-cpus $n_cpus \
                 --n-landmarks 60 --init-obs-var-mult .2"

    name=radialblr
    file=blr.py
    rest_of_args="--type radial $common_args"
    handle

    name=optimal
    file=bayes_optimal.py
    rest_of_args=""
    handle


    file=train_test.py

    name=radial
    rest_of_args="-n radial $common_args"
    handle

    base_args="-n shrinkage $common_args"
    for R in rbfR realR; do
        if [[ $R == rbfR ]]; then
            R_arg="--use-rbf-R"
        else
            R_arg="--use-real-R"
        fi

        for shrink in zero mean; do
            if [[ $shrink == mean ]]; then
                shrink_name="_tomean"
                shrink_arg="--shrink-towards-mean"
            else
                shrink_name=""
                shrink_arg=""
            fi

            for tau in fix opt; do
                name=shrinkage_${R}_${tau}tau.01${shrink_name}
                rest_of_args="$base_args $R_arg $shrink_arg \
                              --${tau}-prior-feat-var --init-prior-feat-var .01"
                handle
            done

            name=shrinkage_${R}_empcov${shrink_name}
            rest_of_args="$base_args $R_arg $shrink_arg --use-empirical-cov"
            handle
        done
    done
done
