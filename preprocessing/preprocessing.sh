#!/bin/bash

dataset=$1
config=$2
backbone=$3
batch_size=$4
# creating train augmentations datatset
cat ./preprocessing/selected_noises.txt | while read -r noise
do
    echo "running on $noise"
    python  ./preprocessing/create_augmentation.py --dataset $dataset --config $config --severity 1 --train --aug $noise
done

# creating test augmentations datatset
cat ./preprocessing/selected_noises.txt | while read -r noise
do
    echo "running on $noise"
    python ./preprocessing/create_augmentation.py --dataset $dataset --config $config --severity 5 --aug $noise
done

# creating representation from selected backbone and dataset
C=0
cat ./preprocessing/selected_noises.txt | while read -r noise
do
    echo "running on $noise"
    if [ $C -eq 0 ]
    then
        python ./preprocessing/create_representations.py --aug $noise --dataset $dataset --config $config --backbone $backbone --batch_size $batch_size --save_rep_norm --save_rep_aug --gpu 0
        C=1
    else
        python ./preprocessing/create_representations.py --aug $noise --dataset $dataset --config $config --backbone $backbone --batch_size $batch_size --save_rep_aug --gpu 0 
    fi
done

# creating wasserstein distances on dataset and augmented dataset
cat ./preprocessing/selected_noises.txt | while read -r noise
do
    echo "running on $noise"
    python ./preprocessing/wasser_dist.py --aug $noise --dataset $dataset --backbone $backbone --config $config --one_class &
done
# sleep 600
# # combine distances for dataset into a dictionary
python ./preprocessing/dist_combination.py --dataset $dataset --backbone $backbone --config $config --one_class
