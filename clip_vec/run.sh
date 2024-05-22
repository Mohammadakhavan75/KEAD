#!/bin/bash

cat /exp10/new_contrastive/selected_noises.txt | while read -r noise
do
    echo "running on $noise"
    python clip_embeding.py --aug $noise --dataset svhn --batch_size 16 --save_rep_norm --save_rep_aug --gpu 0 > logs/cifar100/$noise.out
done
