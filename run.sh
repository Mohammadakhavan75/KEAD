# for i in {0..9}
# do
#     echo "starting run class #$i cifar10 dataset"
#     # python train.py --seed 1 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'clip' &
#     # python train.py --seed 1 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'clip' --multi_neg
#     python train.py --seed 1 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'clip' 
#     # python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'cosine'
#     sleep 5
# done
# for i in {0..9}
# do
#     echo "starting run class #$i cifar10 dataset"
#     python train.py --seed 2 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'clip' &
#     python train.py --seed 2 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'clip'
#     # python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'cosine'
#     sleep 5
# done
# for i in {0..9}
# do
#     echo "starting run class #$i cifar10 dataset"
#     python train.py --seed 3 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'clip' &
#     python train.py --seed 3 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'clip'
#     # python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'cosine'
#     sleep 5
# done
# for i in {0..9}
# do
#     echo "starting run class #$i cifar10 dataset"
#     python train.py --seed 4 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'clip' &
#     python train.py --seed 4 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'clip'
#     # python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'cosine'
#     sleep 5
# done
# for i in {0..9}
# do
#     echo "starting run class #$i cifar10 dataset"
#     python train.py --seed 5 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'clip' &
#     python train.py --seed 5 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'clip'
#     # python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'cosine'
#     sleep 5
# done






# for i in {0..9}
# do
#     echo "starting run class #$i svhn dataset"
#     python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --tail_positive 5 --tail_negative 95 --dataset 'svhn' > outputs/svhn_class_$i.log
#     sleep 5
# done

# CIFAR 10 RUN
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 1 --tail_negative 50 --dataset 'cifar10' & 
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 1 --tail_negative 75 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 1 --tail_negative 95 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 1 --tail_positive 50 --dataset 'cifar10' 
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 1 --tail_positive 25 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 1 --tail_positive 5 --dataset 'cifar10' &

# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.5 --tail_negative 50 --dataset 'cifar10' & 
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.5 --tail_negative 75 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.5 --tail_negative 95 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.5 --tail_positive 50 --dataset 'cifar10' 
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.5 --tail_positive 25 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.5 --tail_positive 5 --dataset 'cifar10' &

# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.1 --tail_negative 50 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.1 --tail_negative 75 --dataset 'cifar10' 
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.1 --tail_negative 95 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.1 --tail_positive 50 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.1 --tail_positive 25 --dataset 'cifar10' &
# python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx 1 --optimizer 'sgd' --temperature 0.1 --tail_positive 5 --dataset 'cifar10' &


# for i in {6..9}
# do
#     echo "starting run class #$i cifar10 dataset"
#     python train.py --seed 1 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'wasser' 
#     sleep 5
# done

# for i in {0..9}
# do
#     echo "starting run class #$i cifar10 dataset"
#     # python train.py --seed 1 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'clip' &
#     # python train.py --seed 1 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'clip' --multi_neg
#     python train.py --seed 1 --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'wasser' --multi_pair --gpu 0
#     # python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --shift_normal --preprocessing 'cosine'
#     sleep 5
# done

# i=9 ; conda activate Torch; cd ~/CSI/exp10/new_contrastive/ ; python train.py --learning_rate 0.1 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'wasser' --multi_pair --milestones  30 100
# i=9 ; conda activate Torch; cd ~/CSI/exp10/new_contrastive/ ; python train.py --learning_rate 0.01 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'wasser' --multi_pair --milestones  30 100
# i=9 ; conda activate Torch; cd ~/CSI/exp10/new_contrastive/ ; python train.py --learning_rate 0.001 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'wasser' --multi_pair --milestones  30 100
# i=9 ; conda activate Torch; cd ~/CSI/exp10/new_contrastive/ ; python train.py --learning_rate 0.0001 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'wasser' --multi_pair --milestones  30 100
# i=9 ; conda activate Torch; cd ~/CSI/exp10/new_contrastive/ ; python train.py --learning_rate 0.00001 --epochs 20 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.5 --dataset 'cifar10' --preprocessing 'wasser' --multi_pair --milestones  30 100

for i in {10..19}
do
    echo ""RUN $i
    python train.py --learning_rate 0.01 --epochs 50 --batch_size 64 --one_class_idx $i --optimizer 'sgd' --temperature 0.2 --dataset $1 --preprocessing 'wasser' --k_pairs $2
done