# type=$1
# type="cifar10_t2"
# dataset="cifar10"
k=$1
for idx in {0..9}
do
    echo "Class $idx"
    python evaluation.py --dataset cifar10 --score 'svm' --one_class_idx $idx --model_path ./run/cifar10_pairs/exp_cifar10_0.01_3_0.1_sgd_epochs_50_one_class_idx_$idx\_temprature_0.2_shift_normal_False_preprocessing_wasser_seed_1_linear_layer_False_k_pairs_$k/last_params.pt #models/model_params_epoch_19.pt 
done
# for idx in {0..9}
# do
#     echo "Class $idx"
#     python evaluation.py --dataset $dataset --gpu 0 --score 'svm' --one_class_idx $idx --model_path ./run/cifar10_pairs/exp_cifar10_0.01_3_0.1_sgd_epochs_50_one_class_idx_$idx\_temprature_0.2_shift_normal_False_preprocessing_wasser_seed_1_linear_layer_False_k_pairs_2/last_params.pt 
# done

# for noise in $(cat ./noises.txt)
# do
#     echo "$noise"
#     modelpath=''
#     python evaluation.py --model_path ./run/$modelpath/last_params.pt --one_class_idx 1 --noise $noise --csv
# done

# for modelpath in $(ls run/ | grep 'exp_svhn_')
# do
#     for i in {0..19}
#     do
#         python evaluation.py --model_path ./run/$modelpath/last_params.pt --one_class_idx $i --dataset 'svhn' --score 'svm' --csv
#         # python evaluation.py --model_path ./run/exp_cifar10_0.1_3_0.1_sgd_epochs_20_one_class_idx_$i\_temprature_0.2_shift_normal_False_preprocessing_wasser_seed_1_linear_layer_True_multi_pair_True/last_params.pt --one_class_idx $i --dataset 'anomaly' --score 'svm' --linear
#     done
# done