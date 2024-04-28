#!/bin/bash

cat /storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-Train-R-C/noises.txt | while read -r noise
do
    python clip_embeding.py --aug $noise --dataset $1 #> outputs/$noise.out
done
