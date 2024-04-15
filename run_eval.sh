# cc=0
# for folder in $(ls ./run/)
# do
#     echo "Class $cc"
#     python evaluation.py --model_path ./run/$folder/best_params.pt --one_class_idx $cc
#     cc=$(($cc + 1))
# done

for noise in $(cat ./noises.txt)
do
    echo "$noise"
    python evaluation.py --model_path ./run/exp-2024-04-09-15-13-56-141768_0.005_4.0_0.5_sgd_0.0_one_class_idx_3/best_params.pt --one_class_idx 3 --noise $noise
done