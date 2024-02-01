#!/bin/bash
mkdir outputs
cat /storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-R-C/noises.txt | while read -r noise
do
    python eval.py --model_path ./run/best.pt --aug $noise > outputs/$noise.out
done

cat /storage/users/makhavan/CSI/finals/datasets/generalization_repo_dataset/CIFAR-10-R-A/noises.txt | while read -r noise
do
    python eval.py --model_path ./run/best.pt --aug $noise --transform 1 > outputs/$noise.out
done

