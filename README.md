>📋  A README.md for code accompanying paper

# Revisit Operation Selection in Differentiable Architecture Search: A Perspective from Influence-Directed Explanations

This repository is the implementation of Revisit Operation Selection in Differentiable Architecture Search: A Perspective from Influence-Directed Explanations 


## Requirements

To install requirements:

```setup
Following NAS-Bench-201 install requirements, download NAS-Bench-201-v1_1-096897.pth
```


## A example of DARTS-IM on NAS-Bench-201

The results can be reproduced in NAS_Bench201/exps/algos/run_batches_sherman.sh

The results can be found in NAS_Bench201/exps/algos/ZZZZ_INTER_RESULT.zip

For example, the results in the fold "run_example_bathces_neumann_240" means that we run the algorithm with Neurmann series approximation with random seed 2 and number of batches 40.

The results in the fold "run_track_example_bathces_sherman_130" means that we run the algorithm with Sherman-Mprrison approximation with random seed 1 and number of batches 30, with tracking every epoch.


## Trained supernt

Before reproducing our results, place the trained supernet in an approriate path.


## Trained model in DARTS space

The trained model in DARTS space can be found in CIFAR10_DARTS_IM and IMAGENET_DARTS_IM.
