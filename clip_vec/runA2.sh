#!/bin/bash

cat /storage/users/rkashefi/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-Train-R-A/noises.txt | while read -r noise
do
    python clip_embeding2.py --aug $noise --transform 1 > outputs2/$noise.out
done
