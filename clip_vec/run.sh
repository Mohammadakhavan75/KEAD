#!/bin/bash

cat /storage/users/makhavan/CSI/exp10/new_contrastive/selected_noises.txt | while read -r noise
do
    python clip_embeding.py --aug $noise --dataset $1 #> outputs/$noise.out
done
