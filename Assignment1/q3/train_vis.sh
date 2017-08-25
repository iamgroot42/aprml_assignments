#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python vae.py --epochs 1000 --hdim 500 --outfile mnist_model.pk --dset mnist --zdim $1
THEANO_FLAGS=device=gpu,floatX=float32 python manifold.py mnist_model.pk mnist --zdim $1
