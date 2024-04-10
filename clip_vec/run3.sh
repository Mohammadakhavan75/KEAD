#!/bin/bash

cat /storage/users/rkashefi/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-Train-R-C/noises.txt | while read -r noise
do
    python clip_embeding3.py --aug $noise > outputs3/$noise.out
done
