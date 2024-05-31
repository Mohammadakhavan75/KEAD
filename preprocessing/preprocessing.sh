#!/bin/bash

dataset=$1
config=$2
backbone=$3
# creating train augmentations datatset
python create_augmentations.py --dataset $dataset --config $config --severity 1 --train 

# creating test augmentations datatset
python create_augmentations.py --dataset $dataset --config $config --severity 6

# creating representation from selected backbone and dataset
cat ./selected_noises.txt | while read -r noise
do
    echo "running on $noise"
    python create_representations.py --aug $noise --dataset $dataset --config $config --backbone $backbone --batch_size 16 --save_rep_norm --save_rep_aug --gpu 0 
done

# creating wasserstein distances on dataset and augmented dataset
cat ./selected_noises.txt | while read -r noise
do
    echo "running on $noise"
    python wasser_dist.py --aug $noise --dataset $dataset --backbone $backbone --config $config
done

# combine distances for dataset into a dictionary
python dist_combination.py --dataset $dataset --backbone $backbone --config $config
