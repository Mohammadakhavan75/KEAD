#!/bin/bash

cat /storage/users/rkashefi/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-Train-R-C/noises.txt | while read -r noise
do
    python clip_embeding2.py --aug $noise > outputs2/$noise.out
done
