for i in {0..9}
do
    echo "starting run #$i"
    python train.py --learning_rate 0.001 --lr_update_rate 5 --lr_gamma 0.5 --epochs 20 --batch_size 16 --one_class_idx $i --optimizer 'sgd' --lamb 0 --tail > outputs/class_$i.log
    sleep 5
done