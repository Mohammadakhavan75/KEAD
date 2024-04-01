for i in {0..9}
do
    python train.py --learning_rate 0.001 --lr_update_rate 2 --lr_gamma 0.5 --epochs 10 --batch_size 16 --one_class_idx $i --lamb 1 > outputs/class_$i.log &
    sleep 5
done